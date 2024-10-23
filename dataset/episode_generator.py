import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class EpisodeGenerator(Dataset):
    """
    A Dataset class for generating episodes for few-shot learning tasks.

    This class creates episodes consisting of support and query sets for N-way,
    K-shot classification tasks. It applies data augmentation to the images and
    organizes them into episodes suitable for meta-learning algorithms like
    Prototypical Networks.

    Attributes:
        dataset (Dataset): The original dataset to sample from.
        config (object): Configuration object containing task parameters.
        mode (str): The mode of operation ('train', 'val', or 'test').
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        class_images (dict): A dictionary mapping class labels to image indices.
        classes (list): List of all available class labels.
    """

    def __init__(self, dataset, config, mode='train'):
        """
        Initialize the EpisodeGenerator.

        Args:
            dataset (Dataset): The original dataset to sample from.
            config (object): Configuration object with task parameters.
            mode (str, optional): The mode of operation. Defaults to 'train'.
        """
        self.dataset = dataset
        self.config = config
        self.mode = mode
        self.n_way = config.n_way
        self.k_shot = config.k_shot

        # Group images by class
        self.class_images = {}
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label not in self.class_images:
                self.class_images[label] = []
            self.class_images[label].append(idx)

        # Add new data reduction logic
        if mode == 'train' and hasattr(config, 'data_percentage') and config.data_percentage < 1.0:
            self._reduce_dataset()
            print(f"Using {config.data_percentage*100}% of training data")

        # Dynamically adjust n_query based on available samples
        min_samples_per_class = min(len(imgs) for imgs in self.class_images.values())
        self.n_query = min(config.n_query, min_samples_per_class - self.k_shot)
        print(f"Adjusted query size to {self.n_query} based on available data")

        # Filter out classes that don't have enough samples
        min_samples_required = self.k_shot + self.n_query
        self.valid_classes = [
            c for c in self.class_images.keys()
            if len(self.class_images[c]) >= min_samples_required
        ]

        if not self.valid_classes:
            raise ValueError(
                f"No classes have enough samples for {self.k_shot}-shot "
                f"with {self.n_query} query images after data reduction"
            )

        print(f"Number of valid classes: {len(self.valid_classes)}")
        print(f"Samples per class after reduction: {min_samples_per_class}")

    def _reduce_dataset(self):
        """Reduce the dataset while maintaining class balance"""
        for label in self.class_images:
            indices = self.class_images[label]
            num_samples = len(indices)
            num_keep = max(1, int(num_samples * self.config.data_percentage))
            self.class_images[label] = np.random.choice(
                indices, num_keep, replace=False).tolist()

        total_samples = sum(len(indices) for indices in self.class_images.values())
        print(f"Reduced dataset size: {total_samples} images")

    def __len__(self):
        """
        Return the number of episodes per epoch.

        Returns:
            int: Number of episodes per epoch (fixed at 1000).
        """
        return 1000  # Number of episodes per epoch

    def __getitem__(self, idx):
        """
        Generate a single episode for few-shot learning.

        This method creates an episode by sampling N classes and K+Q images per class,
        splitting them into support and query sets.

        Args:
            idx (int): Index of the episode (not used, but required by PyTorch).

        Returns:
            dict: A dictionary containing:
                - 'support_images': Tensor of support set images.
                - 'support_labels': Tensor of support set labels.
                - 'query_images': Tensor of query set images.
                - 'query_labels': Tensor of query set labels.
        """
        # Sample N classes from valid classes
        episode_classes = np.random.choice(
            self.valid_classes,
            self.n_way,
            replace=False
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for label_idx, class_label in enumerate(episode_classes):
            # Get images for this class
            class_images = self.class_images[class_label]

            # Sample K+Q images with replacement if necessary
            n_required = self.k_shot + self.n_query
            if len(class_images) < n_required:
                selected_images = np.random.choice(
                    class_images,
                    n_required,
                    replace=True  # Allow replacement if not enough samples
                )
            else:
                selected_images = np.random.choice(
                    class_images,
                    n_required,
                    replace=False
                )

            # Split into support and query
            for i, img_idx in enumerate(selected_images):
                img, _ = self.dataset[img_idx]  # Image is already transformed
                if i < self.k_shot:
                    support_images.append(img)
                    support_labels.append(label_idx)
                else:
                    query_images.append(img)
                    query_labels.append(label_idx)

        # Stack images and convert labels to tensors
        episode = {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long)
        }

        return episode

def mixup_episode(episode, alpha=0.2):
    """
    Apply mixup data augmentation to an episode.

    This function performs mixup on both support and query images within an episode.
    Mixup creates new virtual examples by linearly interpolating between pairs of images
    and their labels.

    Args:
        episode (dict): A dictionary containing support and query images and labels.
        alpha (float, optional): Parameter for the beta distribution. Defaults to 0.2.

    Returns:
        dict: A dictionary containing mixed-up support and query data, including:
            - 'support_images': Mixed support images.
            - 'support_labels_a': Original support labels.
            - 'support_labels_b': Mixed support labels.
            - 'support_lambda': Mixup coefficient for support set.
            - 'query_images': Mixed query images.
            - 'query_labels_a': Original query labels.
            - 'query_labels_b': Mixed query labels.
            - 'query_lambda': Mixup coefficient for query set.
    """
    def mixup(images, labels):
        """
        Apply mixup to a batch of images and labels.

        Args:
            images (Tensor): Batch of images.
            labels (Tensor): Corresponding labels.

        Returns:
            tuple: Mixed images, original labels, mixed labels, and mixup coefficient.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = len(images)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam

    # Apply mixup to support set
    mixed_support, sa_labels, sb_labels, s_lam = mixup(
        episode['support_images'],
        episode['support_labels']
    )

    # Apply mixup to query set
    mixed_query, qa_labels, qb_labels, q_lam = mixup(
        episode['query_images'],
        episode['query_labels']
    )

    return {
        'support_images': mixed_support,
        'support_labels_a': sa_labels,
        'support_labels_b': sb_labels,
        'support_lambda': s_lam,
        'query_images': mixed_query,
        'query_labels_a': qa_labels,
        'query_labels_b': qb_labels,
        'query_lambda': q_lam
    }

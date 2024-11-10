import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone="resnet50", feature_dim=1024, n_heads=8, n_layers=1):
        super(PrototypicalNetwork, self).__init__()
        # Initialize ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final FC layer
        layers = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*layers)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Transformer encoder layer
        encoder_layers = TransformerEncoderLayer(
            d_model=2048,  # ResNet50 outputs 2048 features
            nhead=n_heads,
            dim_feedforward=feature_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,  # Set batch_first=True for batch dimension first
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=n_layers
        )

        # Layer normalization after Transformer
        self.layer_norm = nn.LayerNorm(2048)

        # Final projection and normalization
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Layer groups for discriminative fine-tuning
        self.layer_groups = [
            list(self.encoder[:4].parameters()),  # Early layers
            list(self.encoder[4:6].parameters()),  # Middle layers
            list(self.encoder[6:].parameters()),  # Late layers
            list(self.transformer_encoder.parameters()),  # Transformer encoder
            list(self.projection.parameters()),  # Projection head
        ]

    def forward(self, x, support_set=False):
        x = self.encoder(x)  # (batch_size, 2048, 1, 1)
        x = self.flatten(x)  # (batch_size, 2048)

        if support_set:
            # Add sequence dimension for Transformer
            x = x.unsqueeze(0)  # (1, batch_size, 2048)
            x = self.transformer_encoder(x)  # (1, batch_size, 2048)
            x = x.squeeze(0)  # (batch_size, 2048)
            x = self.layer_norm(x)
        else:
            # For query embeddings, no adaptation
            pass

        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings


def compute_prototypes(support_embeddings, support_labels):
    """
    Compute class prototypes from support set embeddings.
    Args:
        support_embeddings: Tensor of shape (n_way * k_shot, embedding_dim)
        support_labels: Tensor of shape (n_way * k_shot)
    Returns:
        prototypes: Tensor of shape (n_way, embedding_dim)
    """
    n_way = len(torch.unique(support_labels))
    prototypes = []

    for i in range(n_way):
        mask = support_labels == i
        class_embeddings = support_embeddings[mask]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes)


def prototypical_loss(prototypes, query_embeddings, query_labels, temperature=0.5):
    """
    Compute prototypical networks loss.
    Args:
        prototypes: Tensor of shape (n_way, embedding_dim)
        query_embeddings: Tensor of shape (n_way * n_query, embedding_dim)
        query_labels: Tensor of shape (n_way * n_query)
        temperature: Temperature for scaling logits
    Returns:
        loss: Scalar loss value
        accuracy: Batch accuracy
    """
    # Compute squared euclidean distances
    distances = torch.cdist(query_embeddings, prototypes) / temperature

    # Compute log probabilities
    log_p_y = F.log_softmax(-distances, dim=1)

    # Compute cross entropy loss
    loss = F.nll_loss(log_p_y, query_labels)

    # Compute accuracy
    _, predictions = log_p_y.max(1)
    accuracy = torch.eq(predictions, query_labels).float().mean()

    return loss, accuracy


def mixup_data(x, y, alpha=0.2):
    """
    Performs MixUp augmentation.
    Args:
        x: Input tensor
        y: Target tensor
        alpha: MixUp interpolation strength
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Performs CutMix augmentation.
    Args:
        x: Input tensor
        y: Target tensor
        alpha: CutMix interpolation strength
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Get random box dimensions
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Get random box position
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Perform CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class EpisodeSampler:
    """Samples episodes for few-shot training with limited samples per class"""

    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # Group samples by class
        self.class_samples = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_samples:
                self.class_samples[label] = []
            self.class_samples[label].append(idx)

        # Filter out classes with too few samples
        min_samples_required = k_shot + n_query
        self.valid_classes = [
            cls
            for cls, samples in self.class_samples.items()
            if len(samples) >= min_samples_required
        ]

        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes with {min_samples_required} samples. "
                f"Found {len(self.valid_classes)} valid classes, need {n_way}."
            )

        print(
            f"Found {len(self.valid_classes)} valid classes "
            f"with >= {min_samples_required} samples each"
        )

    def sample_episode(self):
        """Sample a single episode ensuring enough samples per class"""
        # Sample n_way classes from valid classes
        episode_classes = random.sample(self.valid_classes, self.n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for label_idx, cls in enumerate(episode_classes):
            # Get available samples for this class
            available_samples = self.class_samples[cls]

            # Determine how many query samples we can use
            n_available = len(available_samples)
            n_query_actual = min(self.n_query, n_available - self.k_shot)

            # Sample without replacement
            selected_indices = random.sample(
                available_samples, self.k_shot + n_query_actual
            )

            # Split into support and query
            support_idx = selected_indices[: self.k_shot]
            query_idx = selected_indices[self.k_shot : self.k_shot + n_query_actual]

            # Add to support set
            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(label_idx)

            # Add to query set
            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(label_idx)

        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)

        return {
            "support_images": support_images,
            "support_labels": support_labels,
            "query_images": query_images,
            "query_labels": query_labels,
        }

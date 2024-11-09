# evaluate_proto_net.py

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from torchvision.datasets import Flowers102  # type: ignore
from torch.utils.data import Dataset  # type: ignore
import numpy as np  # type: ignore
import random
from tqdm import tqdm  # type: ignore
import os
import matplotlib.pyplot as plt  # type: ignore
from typing import Dict, Tuple, List
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


from src.models.prototypical_network import (
    PrototypicalNetwork,
    compute_prototypes,
    prototypical_loss,
)


class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # Group samples by label
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

        # Remove minimum samples per class requirement
        if len(self.labels) < n_way:
            raise ValueError(
                f"Not enough classes. Found {len(self.labels)}, need {n_way}"
            )

    def sample_episode(self):
        # Randomly select n_way classes
        episode_classes = random.sample(self.labels, self.n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_label in episode_classes:
            # Allow resampling of indices
            class_indices = self.label_to_indices[class_label]

            # Sample with replacement if needed
            support_idx = np.random.choice(
                class_indices, size=self.k_shot, replace=True
            )
            query_idx = np.random.choice(class_indices, size=self.n_query, replace=True)

            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(episode_classes.index(class_label))

            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(episode_classes.index(class_label))

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


# Define MappedSubset class and split_dataset function
class MappedSubset(Dataset):
    """Custom Dataset that applies class mapping"""

    def __init__(self, dataset, indices, class_to_idx):
        self.dataset = dataset
        self.indices = indices
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        mapped_label = self.class_to_idx[label]
        return image, mapped_label

    def __len__(self):
        return len(self.indices)


def split_dataset(dataset):
    """Split dataset into train and test sets based on classes"""
    all_classes = list(range(102))
    random.shuffle(all_classes)

    train_classes = all_classes[:60]
    test_classes = all_classes[60:]

    train_indices = []
    test_indices = []

    for idx, (_, label) in enumerate(dataset):
        if label in train_classes:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    train_class_to_idx = {label: idx for idx, label in enumerate(sorted(train_classes))}
    test_class_to_idx = {label: idx for idx, label in enumerate(sorted(test_classes))}

    return (
        (train_indices, test_indices),
        (train_class_to_idx, test_class_to_idx),
        (train_classes, test_classes),
    )


def evaluate_on_test(model, test_sampler, device, n_episodes=100):
    """Evaluate model on test set"""
    model.eval()
    total_acc = 0
    accuracies = []

    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Test Evaluation"):
            batch = test_sampler.sample_episode()

            support_images = batch["support_images"].to(device)
            support_labels = batch["support_labels"].to(device)
            query_images = batch["query_images"].to(device)
            query_labels = batch["query_labels"].to(device)

            support_features = model(support_images)
            query_features = model(query_images)
            prototypes = compute_prototypes(support_features, support_labels)
            _, acc = prototypical_loss(
                prototypes, query_features, query_labels, temperature=0.5
            )

            accuracies.append(acc.item())
            total_acc += acc.item()

    mean_acc = total_acc / n_episodes
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)

    return mean_acc, ci95, accuracies


def main():
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print("Loading dataset...")
    dataset = Flowers102(root="./data", download=True, transform=transform)

    # Split dataset
    print("Splitting dataset...")
    (
        (train_indices, test_indices),
        (train_mapping, test_mapping),
        (train_classes, test_classes),
    ) = split_dataset(dataset)

    # Create mapped datasets
    test_dataset = MappedSubset(dataset, test_indices, test_mapping)

    print(f"Number of test classes: {len(test_classes)}")

    # Load the model
    print("Loading model...")
    model = PrototypicalNetwork(backbone="resnet50", feature_dim=1024).to(device)
    checkpoint = torch.load(
        "/home/zrgong/data/BloomLens/checkpoints/stage_4/best_model.pt"
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Configurations to evaluate
    configs = [
        (5, 1, 15),  # 5-way-1-shot with 15 queries per class
        (5, 5, 15),  # 5-way-5-shot with 15 queries per class
        (10, 1, 10),  # 10-way-1-shot with 10 queries per class
        (10, 5, 10),  # 10-way-5-shot with 10 queries per class
        (20, 1, 5),  # 20-way-1-shot with 5 queries per class
        (20, 5, 5),  # 20-way-5-shot with 5 queries per class
    ]

    # Open log file
    os.makedirs("results", exist_ok=True)
    with open("results/proto_net_evaluation_log.txt", "w") as log_file:
        # Evaluation loop
        for n_way, k_shot, n_query in configs:
            print(f"\nEvaluating {n_way}-way {k_shot}-shot:")
            log_file.write(f"\n{n_way}-way {k_shot}-shot:\n")
            try:
                test_sampler = EpisodeSampler(test_dataset, n_way, k_shot, n_query)
            except ValueError as e:
                print(f"Cannot evaluate {n_way}-way {k_shot}-shot: {e}")
                log_file.write(f"Cannot evaluate {n_way}-way {k_shot}-shot: {e}\n")
                continue

            mean_acc, ci95, _ = evaluate_on_test(
                model, test_sampler, device, n_episodes=100
            )

            result_str = f"Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%\n"
            print(result_str)
            log_file.write(result_str)

    print("\nResults have been saved to 'results/proto_net_evaluation_log.txt'")


if __name__ == "__main__":
    main()

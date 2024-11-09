# src/evaluation/evaluate_proto_net.py
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from src.models.prototypical_network import (
    PrototypicalNetwork,
    compute_prototypes,
)
from src.data.dataset import MappedSubset, split_dataset
from torchvision.datasets import Flowers102  # type: ignore
from torchvision import transforms  # type: ignore
import random  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore


def evaluate_prototypical_network(
    model, test_dataset, device, n_way=5, k_shot=5, q_queries=5, n_episodes=100
):
    model.eval()
    accuracies = []

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes"):
        # Sample an episode
        episode_classes = random.sample(list(test_dataset.class_to_idx.values()), n_way)
        support_indices = []
        query_indices = []

        for cls in episode_classes:
            cls_indices = [
                idx for idx, (_, label) in enumerate(test_dataset) if label == cls
            ]
            if len(cls_indices) < k_shot + q_queries:
                # Not enough samples in this class, skip this episode
                break
            selected = random.sample(cls_indices, k_shot + q_queries)
            support_indices.extend(selected[:k_shot])
            query_indices.extend(selected[k_shot:])

        if (
            len(support_indices) != n_way * k_shot
            or len(query_indices) != n_way * q_queries
        ):
            continue  # Skip this episode if not enough samples

        # Create a mapping from original class labels to episode class labels (0 to n_way-1)
        class_mapping = {cls: i for i, cls in enumerate(episode_classes)}

        # Get support and query samples
        support_set = torch.utils.data.Subset(test_dataset, support_indices)
        query_set = torch.utils.data.Subset(test_dataset, query_indices)

        support_loader = DataLoader(
            support_set, batch_size=k_shot * n_way, shuffle=False
        )
        query_loader = DataLoader(
            query_set, batch_size=q_queries * n_way, shuffle=False
        )

        support_images, support_labels = next(iter(support_loader))
        query_images, query_labels = next(iter(query_loader))

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        # Map labels to episode-specific labels
        support_labels = torch.tensor(
            [class_mapping[label.item()] for label in support_labels]
        ).to(device)
        query_labels = torch.tensor(
            [class_mapping[label.item()] for label in query_labels]
        ).to(device)

        # Forward pass
        with torch.no_grad():
            support_embeddings = model(support_images)
            query_embeddings = model(query_images)

            # Compute prototypes
            prototypes = compute_prototypes(support_embeddings, support_labels)

            # Compute distances and predict labels
            distances = torch.cdist(query_embeddings, prototypes)
            log_p_y = torch.log_softmax(-distances, dim=1)
            _, y_hat = log_p_y.max(1)

            # Compute accuracy
            acc = torch.eq(y_hat, query_labels).float().mean().item()
            accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
    print(
        f"Accuracy over {len(accuracies)} episodes: {mean_acc*100:.2f}% ± {ci95*100:.2f}%"
    )
    return mean_acc, ci95, accuracies


if __name__ == "__main__":
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    transform = transforms.Compose(
        [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Using ImageNet statistics
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Load dataset
    print("Loading dataset...")
    dataset = Flowers102(root="./data", download=True, transform=transform)

    # Split dataset
    print("Splitting dataset...")
    (
        (train_indices, val_indices, test_indices),
        (train_mapping, val_mapping, test_mapping),
        (train_classes, val_classes, test_classes),
    ) = split_dataset(dataset)

    # Create mapped test dataset
    test_dataset = MappedSubset(dataset, test_indices, test_mapping)

    # Load model
    model = PrototypicalNetwork().to(device)
    checkpoint = torch.load(
        "/home/zrgong/BloomLens/results/checkpoints/stage_3/best_model.pt"
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )  # Extract just the model state

    # Evaluate the model
    print("Evaluating Prototypical Network...")
    configs = [
        (5, 1),  # 5-way 1-shot
        (5, 5),  # 5-way 5-shot
        (10, 1),  # 10-way 1-shot
        (10, 5),  # 10-way 5-shot
        (20, 1),  # 20-way 1-shot
        (20, 5),  # 20-way 5-shot
    ]

    results = {}
    for n_way, k_shot in configs:
        print(f"\nEvaluating {n_way}-way {k_shot}-shot:")
        mean_acc, ci95, accuracies = evaluate_prototypical_network(
            model,
            test_dataset,
            device,
            n_way=n_way,
            k_shot=k_shot,
            q_queries=5,
            n_episodes=100,
        )
        results[(n_way, k_shot)] = {
            "mean_acc": mean_acc,
            "ci95": ci95,
            "accuracies": accuracies,
        }
        print(f"Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")

    # Optionally, save the results or plot them
    # Save the results to a JSON file
    import json

    with open("./results/proto_net_results.json", "w") as f:
        json.dump(results, f)

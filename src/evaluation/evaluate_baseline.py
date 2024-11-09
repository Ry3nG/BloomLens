# src/evaluation/evaluate_baseline.py

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from src.models.flower_classifier import FlowerClassifier  # type: ignore
from src.data.dataset import MappedSubset, split_dataset  # type: ignore
from torchvision.datasets import Flowers102  # type: ignore
from torchvision import transforms  # type: ignore
import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import random  # type: ignore


def extract_features(model, dataloader, device):
    """Extract features from the model before the final FC layer"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            batch_features = model.get_features(inputs)
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)


def run_few_shot_evaluation(
    features, labels, n_way=5, k_shot=5, n_query=15, n_episodes=100
):
    """Run multiple episodes of few-shot evaluation"""
    accuracies = []
    unique_classes = np.unique(labels)

    for _ in tqdm(range(n_episodes), desc="Evaluating episodes"):
        # Randomly sample n_way classes
        selected_classes = np.random.choice(unique_classes, n_way, replace=False)

        support_features_list = []
        support_labels_list = []
        query_features_list = []
        query_labels_list = []

        for class_idx in selected_classes:
            # Get all samples for this class
            class_features = features[labels == class_idx]
            class_indices = np.random.permutation(len(class_features))

            # Split into support and query sets
            support_idx = class_indices[:k_shot]
            query_idx = class_indices[k_shot : k_shot + n_query]

            support_features_list.append(class_features[support_idx])
            support_labels_list.extend([class_idx] * k_shot)
            query_features_list.append(class_features[query_idx])
            query_labels_list.extend([class_idx] * len(query_idx))

        # Combine all support and query samples
        support_features = np.vstack(support_features_list)
        support_labels = np.array(support_labels_list)
        query_features = np.vstack(query_features_list)
        query_labels = np.array(query_labels_list)

        # Evaluate using k-NN
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(support_features, support_labels)
        predictions = knn.predict(query_features)
        accuracy = accuracy_score(query_labels, predictions)
        accuracies.append(accuracy)

    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * np.std(accuracies) / np.sqrt(n_episodes)
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
        (train_indices, val_indices, test_indices),
        (train_mapping, val_mapping, test_mapping),
        (train_classes, val_classes, test_classes),
    ) = split_dataset(dataset)

    # Create mapped datasets
    test_dataset = MappedSubset(dataset, test_indices, test_mapping)

    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = FlowerClassifier(num_classes=len(train_classes)).to(device)
    model.load_state_dict(torch.load("./results/models/baseline_model.pth"))

    # Extract features for test set
    print("Extracting features for few-shot evaluation...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # Run few-shot evaluation
    print("Running few-shot evaluation...")
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
        mean_acc, ci95, accuracies = run_few_shot_evaluation(
            test_features, test_labels, n_way=n_way, k_shot=k_shot
        )
        results[(n_way, k_shot)] = {
            "mean_acc": mean_acc,
            "ci95": ci95,
            "accuracies": accuracies,
        }
        print(f"Accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%")

    # Plot results
    plt.figure(figsize=(10, 6))
    for (n_way, k_shot), result in results.items():
        plt.hist(result["accuracies"], alpha=0.5, label=f"{n_way}-way {k_shot}-shot")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Few-Shot Classification Accuracies")
    plt.legend()
    plt.savefig("./results/figures/baseline_results.png")
    plt.close()

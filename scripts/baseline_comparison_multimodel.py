import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.models as models  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from torchvision.datasets import Flowers102  # type: ignore
from torch.utils.data import DataLoader, Subset, Dataset  # type: ignore
import numpy as np  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore
import random
import os
from tabulate import tabulate  # type: ignore
import pathlib
import json
from typing import Dict, Tuple, List, Any
from src.data.dataset import MappedSubset, split_dataset


class BaselineModel(nn.Module):
    """Generic baseline model wrapper for different architectures"""

    def __init__(self, architecture: str, num_classes: int = 60):
        super().__init__()
        self.architecture = architecture

        model_constructors = {
            "alexnet": (models.alexnet, models.AlexNet_Weights.DEFAULT),
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT),
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "densenet201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "efficientnet_b0": (
                models.efficientnet_b0,
                models.EfficientNet_B0_Weights.DEFAULT,
            ),
        }

        if architecture not in model_constructors:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model_fn, weights = model_constructors[architecture]
        self.model = model_fn(weights=None)

        # Modify the final layer based on architecture
        if architecture.startswith("resnet"):
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif architecture.startswith("densenet"):
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        elif architecture in ["vgg16", "vgg19"]:
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        elif architecture == "mobilenet_v2":
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        elif architecture == "googlenet":
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif architecture == "alexnet":
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        elif architecture.startswith("efficientnet"):
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        """Extract features before the final classification layer"""
        if self.architecture.startswith("resnet"):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.architecture.startswith("densenet"):
            features = self.model.features(x)
            x = torch.nn.functional.relu(features)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif self.architecture in ["vgg16", "vgg19"]:
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            # Only run through part of the classifier, stopping before the last layer
            for layer in self.model.classifier[:-1]:
                x = layer(x)
        elif self.architecture == "mobilenet_v2":
            x = self.model.features(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif self.architecture == "alexnet":
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier[:-1](x)
        elif self.architecture.startswith("efficientnet"):
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

        return x


def train_model(model, train_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_loss = float("inf")
    training_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        training_history.append(
            {"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc}
        )

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

    return best_state, training_history


def extract_features(model, dataloader, device):
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
    # Match the prototypical network evaluation settings
    accuracies = []
    for _ in range(n_episodes):
        # Sample n_way classes
        classes = np.random.choice(np.unique(labels), n_way, replace=False)

        episode_accuracies = []
        for _ in range(n_query):
            support_features = []
            support_labels = []
            query_features = []
            query_labels = []

            # Sample support and query sets
            for label in classes:
                class_features = features[labels == label]
                indices = np.random.permutation(len(class_features))

                # Support set
                support_features.append(class_features[indices[:k_shot]])
                support_labels.extend([label] * k_shot)

                # Query set
                query_features.append(class_features[indices[k_shot : k_shot + 1]])
                query_labels.extend([label])

            support_features = np.vstack(support_features)
            query_features = np.vstack(query_features)

            # Use cosine similarity instead of Euclidean distance
            support_features = support_features / np.linalg.norm(
                support_features, axis=1, keepdims=True
            )
            query_features = query_features / np.linalg.norm(
                query_features, axis=1, keepdims=True
            )

            # Compute prototypes
            prototypes = []
            for c in classes:
                class_support = support_features[np.array(support_labels) == c]
                prototype = np.mean(class_support, axis=0)
                prototypes.append(prototype)
            prototypes = np.stack(prototypes)

            # Compute similarities
            similarities = query_features @ prototypes.T
            predictions = classes[np.argmax(similarities, axis=1)]

            accuracy = np.mean(predictions == query_labels)
            episode_accuracies.append(accuracy)

        accuracies.append(np.mean(episode_accuracies))

    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * np.std(accuracies) / np.sqrt(n_episodes)
    return mean_acc, ci95, accuracies


def plot_training_history(histories: Dict[str, List], save_path: str):
    """Plot training histories for all models"""
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    for model_name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        plt.plot(epochs, losses, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss by Model")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 1, 2)
    for model_name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        accuracies = [h["accuracy"] for h in history]
        plt.plot(epochs, accuracies, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy by Model")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_few_shot_results(results: Dict[str, Dict], save_path: str):
    """Plot few-shot results for all models"""
    # Create subplots for different n_way configurations
    n_ways = sorted(list({config[0] for config in next(iter(results.values())).keys()}))
    fig, axes = plt.subplots(len(n_ways), 1, figsize=(15, 5 * len(n_ways)))

    for idx, n_way in enumerate(n_ways):
        ax = axes[idx]

        # Get results for this n_way
        x_positions = []
        x_labels = []
        accuracies = []
        errors = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for model_idx, (model_name, model_results) in enumerate(results.items()):
            model_positions = []
            model_accuracies = []
            model_errors = []

            for (way, shot), result in model_results.items():
                if way == n_way:
                    pos = model_idx + len(results) * len(x_positions)
                    model_positions.append(pos)
                    model_accuracies.append(result["mean_acc"] * 100)
                    model_errors.append(result["ci95"] * 100)
                    if model_idx == 0:
                        x_positions.append(pos)
                        x_labels.append(f"{shot}-shot")

            ax.bar(
                model_positions,
                model_accuracies,
                yerr=model_errors,
                label=model_name,
                color=colors[model_idx],
                alpha=0.7,
                width=0.8,
            )

        ax.set_xticks([p + (len(results) - 1) / 2 for p in x_positions])
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{n_way}-way Classification")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")


def main():
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    architectures = [
        "alexnet",
        "resnet18",
        "resnet50",
        "vgg16",
        "vgg19",
        "densenet121",
        "densenet201",
        "mobilenet_v2",
        "efficientnet_b0",
    ]

    # !todo: Comment below line to test all architectures
    # Overwrite architectures to test
    # architectures = ["alexnet"]

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
    train_dataset = MappedSubset(dataset, train_indices, train_mapping)
    test_dataset = MappedSubset(dataset, test_indices, test_mapping)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Open log file
    with open("results/evaluation_log.txt", "w") as log_file:
        # Training and evaluation loop for each architecture
        for arch in architectures:
            print(f"\nProcessing {arch}...")
            log_file.write(f"\n{arch}:\n")

            # Initialize model
            model = BaselineModel(arch, num_classes=len(train_classes)).to(device)

            # Train model
            print(f"Training {arch}...")
            model_state, history = train_model(
                model, train_loader, device, num_epochs=30
            )

            # Extract features for test set
            print(f"Extracting features for {arch}...")
            test_features, test_labels = extract_features(model, test_loader, device)

            # Run few-shot evaluation
            print(f"Running few-shot evaluation for {arch}...")
            configs = [
                (5, 1, 15),  # 5-way-1-shot with 15 queries per class
                (5, 5, 15),  # 5-way-5-shot with 15 queries per class
                (10, 1, 10),  # 10-way-1-shot with 10 queries per class
                (10, 5, 10),  # 10-way-5-shot with 10 queries per class
                (20, 1, 5),  # 20-way-1-shot with 5 queries per class
                (20, 5, 5),  # 20-way-5-shot with 5 queries per class
            ]

            for n_way, k_shot, n_query in configs:
                print(f"\nEvaluating {n_way}-way {k_shot}-shot:")
                mean_acc, ci95, _ = run_few_shot_evaluation(
                    test_features,
                    test_labels,
                    n_way=n_way,
                    k_shot=k_shot,
                    n_query=n_query,
                )
                result_str = f"{n_way}-way {k_shot}-shot: accuracy: {mean_acc*100:.2f}% ± {ci95*100:.2f}%\n"
                print(result_str)
                log_file.write(result_str)

    print("\nResults have been saved to 'results/evaluation_log.txt'")


if __name__ == "__main__":
    main()

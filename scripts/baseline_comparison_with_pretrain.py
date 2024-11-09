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


class MappedSubset(Dataset):
    """Custom Dataset that applies class mapping"""

    def __init__(self, dataset, indices, class_to_idx):
        self.dataset = dataset
        self.indices = indices
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # Map the original label to the new index
        mapped_label = self.class_to_idx[label]
        return image, mapped_label

    def __len__(self):
        return len(self.indices)


def split_dataset(dataset):
    """Split dataset into train and test sets based on classes"""
    all_classes = list(range(102))
    random.shuffle(all_classes)

    # Split classes into train (60) and test (42)
    train_classes = all_classes[:60]
    test_classes = all_classes[60:]

    # Create indices for each split
    train_indices = []
    test_indices = []

    for idx, (_, label) in enumerate(dataset):
        if label in train_classes:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    # Create class mappings
    train_class_to_idx = {label: idx for idx, label in enumerate(sorted(train_classes))}
    test_class_to_idx = {label: idx for idx, label in enumerate(sorted(test_classes))}

    return (
        (train_indices, test_indices),
        (train_class_to_idx, test_class_to_idx),
        (train_classes, test_classes),
    )


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        """Extract features before the final FC layer"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def train_model(model, train_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

    return model.state_dict()


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


def main():
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

    # Train model
    print("Training model...")
    model = FlowerClassifier(num_classes=len(train_classes)).to(device)
    model_state = train_model(model, train_loader, device)

    # Save model
    torch.save(model_state, "baseline_model.pth")

    # Extract features for test set
    print("Extracting features for few-shot evaluation...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # Run few-shot evaluation
    print("Running few-shot evaluation...")
    configs = [
        (5, 1),  # 5-way 1-shot
        (5, 5),  # 5-way 5-shot
        (5, 8),  # 5-way 8-shot
        (10, 1),  # 10-way 1-shot
        (10, 5),  # 10-way 5-shot
        (10, 8),  # 10-way 8-shot
        (20, 1),  # 20-way 1-shot
        (20, 5),  # 20-way 5-shot
        (20, 8),  # 20-way 8-shot
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
    plt.savefig("baseline_results.png")
    plt.close()


if __name__ == "__main__":
    main()

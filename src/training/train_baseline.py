# src/training/train_baseline.py

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore
from src.models.flower_classifier import FlowerClassifier
from src.data.dataset import MappedSubset, split_dataset
from torchvision.datasets import Flowers102  # type: ignore
from torchvision import transforms  # type: ignore
import numpy as np  # type: ignore
import random  # type: ignore


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
    train_dataset = MappedSubset(dataset, train_indices, train_mapping)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train model
    print("Training model...")
    model = FlowerClassifier(num_classes=len(train_classes)).to(device)
    model_state = train_model(model, train_loader, device)

    # Save model
    torch.save(model_state, "./results/models/baseline_model.pth")

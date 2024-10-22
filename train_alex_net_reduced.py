import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset with official splits
root_dir = './data/flowers102'
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=transform, download=True)

# Function to create reduced dataset
def create_reduced_dataset(dataset, reduction_percentage):
    num_samples = len(dataset)
    num_reduced_samples = int(num_samples * reduction_percentage)
    indices = random.sample(range(num_samples), num_reduced_samples)
    return Subset(dataset, indices)

# Create reduced datasets
reduction_percentage = 0.9  # Change this to experiment with different percentages
reduced_train_dataset = create_reduced_dataset(train_dataset, reduction_percentage)
reduced_val_dataset = create_reduced_dataset(val_dataset, reduction_percentage)
reduced_test_dataset = create_reduced_dataset(test_dataset, reduction_percentage)

# Create data loaders
batch_size = 32
train_loader = DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(reduced_val_dataset, batch_size=batch_size)
test_loader = DataLoader(reduced_test_dataset, batch_size=batch_size)

# Load pre-trained AlexNet model
model = models.alexnet(pretrained=True)
num_classes = 102
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Calculate and print validation accuracy
    val_accuracy = calculate_accuracy(val_loader, model)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Evaluate the model on the test set
test_accuracy = calculate_accuracy(test_loader, model)
print(f"Test Accuracy (with {reduction_percentage*100}% of data): {test_accuracy:.2f}%")

# Optional: Save the model
# torch.save(model.state_dict(), f'alexnet_reduced_{reduction_percentage*100}percent.pth')

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet takes 224x224 inputs
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pre-trained AlexNet
])

# Load the training, validation, and test datasets
root_dir = './data/flowers102'
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=transform, download=True)

# DataLoader for batching
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# Modify the classifier to fit 102 classes instead of 1000 (ImageNet)
model.classifier[6] = nn.Linear(4096, 102)  # Replace the last FC layer

# Move the model to the appropriate device (GPU/CPU)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Use tqdm for a progress bar
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print statistics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Calculate and print validation accuracy
    val_accuracy = calculate_accuracy(val_loader, model)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Add early stopping (optional)
    # if val_accuracy > best_accuracy:
    #     best_accuracy = val_accuracy
    #     torch.save(model.state_dict(), 'best_model.pth')

# Evaluate the model on the test set
test_accuracy = calculate_accuracy(test_loader, model)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the final model (optional)
# torch.save(model.state_dict(), 'final_model.pth')

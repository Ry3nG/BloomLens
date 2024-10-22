import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import random
import os
import logging
from datetime import datetime

# Set up logging
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'baseline_comparison_{timestamp}.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the dataset with official splits
root_dir = './data/flowers102'
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=transform, download=True)

# Function to create reduced dataset
def create_reduced_dataset(dataset, reduction_percentage):
    if reduction_percentage >= 1.0:
        return dataset
    num_samples = len(dataset)
    num_reduced_samples = int(num_samples * reduction_percentage)
    indices = random.sample(range(num_samples), num_reduced_samples)
    logging.info(f"Created reduced dataset with {num_reduced_samples} samples")
    return Subset(dataset, indices)

# Define dataset sizes
dataset_sizes = {
    '100%': 1.0,
    '50%': 0.5,
    '25%': 0.25,
    '10%': 0.10
}

# Define models to compare
model_names = [
    'alexnet',
    'vgg16',
    'vgg19',
    'resnet18',
    'resnet50',
    'resnet101',
    'densenet121',
    'densenet201',
    'shufflenet_v2_x1_0',
    'mobilenet_v2',
    'googlenet'
]

# Function to load and modify model
def get_model(model_name, num_classes=102):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        if model.aux1:
            model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
            model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    logging.info(f"Loaded and modified {model_name} for {num_classes} classes")
    return model.to(device)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):  # For GoogLeNet which has auxiliary outputs
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logging.info(f"Calculated accuracy: {accuracy:.2f}%")
    return accuracy

# Function to train and evaluate a model
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_accuracy = 0
    patience = 10
    counter = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, tuple):  # For GoogLeNet
                outputs = outputs[0]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_accuracy = calculate_accuracy(val_loader, model)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load the best model before final evaluation
    if best_model is not None:
        model.load_state_dict(best_model)

    # Evaluate on test set
    test_accuracy = calculate_accuracy(test_loader, model)
    return test_accuracy

# Prepare a directory to save results
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Initialize a dictionary to store results
results = {size: {} for size in dataset_sizes.keys()}

# Iterate over dataset sizes
for size_name, reduction in dataset_sizes.items():
    logging.info(f"\n=== Dataset Size: {size_name} ({int(reduction*100)}%) ===")

    # Create reduced datasets
    reduced_train = create_reduced_dataset(train_dataset, reduction)
    reduced_val = create_reduced_dataset(val_dataset, reduction)
    reduced_test = create_reduced_dataset(test_dataset, reduction)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(reduced_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(reduced_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(reduced_test, batch_size=batch_size, shuffle=False)

    # Iterate over models
    for model_name in model_names:
        logging.info(f"\nTraining model: {model_name}")
        model = get_model(model_name)

        test_acc = train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs=100, learning_rate=0.001)
        logging.info(f"Test Accuracy for {model_name} with {size_name} data: {test_acc:.2f}%")

        # Store the result
        results[size_name][model_name] = test_acc

        # Optionally, save the model
        # model_path = os.path.join(results_dir, f"{model_name}_{size_name}.pth")
        # torch.save(model.state_dict(), model_path)

# Convert results to DataFrames and save as CSV
for size_name in dataset_sizes.keys():
    df = pd.DataFrame(list(results[size_name].items()), columns=['Model', 'Test Accuracy (%)'])
    csv_path = os.path.join(results_dir, f"performance_{size_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nSaved performance table for {size_name} dataset to {csv_path}")

# Optionally, create a single CSV with all results
all_results = []
for size_name, models in results.items():
    for model_name, acc in models.items():
        all_results.append({'Dataset Size': size_name, 'Model': model_name, 'Test Accuracy (%)': acc})

all_df = pd.DataFrame(all_results)
all_csv_path = os.path.join(results_dir, f"performance_all_sizes_{timestamp}.csv")
all_df.to_csv(all_csv_path, index=False)
logging.info(f"\nSaved all performance results to {all_csv_path}")

# Optionally, visualize the results using plots
import matplotlib.pyplot as plt
import seaborn as sns

# Create a pivot table for visualization
pivot_df = all_df.pivot(index='Model', columns='Dataset Size', values='Test Accuracy (%)')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Performance on Flowers102 Dataset")
plt.ylabel("Model")
plt.xlabel("Dataset Size")
plt.tight_layout()
plt_path = os.path.join(results_dir, f"performance_heatmap_{timestamp}.png")
plt.savefig(plt_path)
plt.close()
logging.info(f"Saved performance heatmap to {plt_path}")

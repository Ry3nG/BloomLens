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
from collections import defaultdict

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
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the dataset with official splits
root_dir = './data/flowers102'
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=test_transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=test_transform, download=True)

# Function to create few-shot dataset
def create_few_shot_dataset(dataset, k_shot):
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) >= k_shot:
            selected_indices.extend(random.sample(indices, k_shot))
        else:
            logging.warning(f"Class {label} has less than {k_shot} samples. Using all available samples.")
            selected_indices.extend(indices)

    logging.info(f"Created {k_shot}-shot dataset with {len(selected_indices)} samples.")
    return Subset(dataset, selected_indices)

# Define few-shot settings
few_shot_settings = {
    '1-shot': 1,
    '5-shot': 5,
    '10-shot': 10,
    'full-data': None  # Use all available data
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
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        # MobileNetV2's classifier is simpler - it only has one linear layer
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes)
        )
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
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
            if isinstance(outputs, tuple):  # For models with multiple outputs
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Function to train and evaluate a model
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0
    patience = 5
    counter = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
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
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy

# Prepare a directory to save results
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Initialize a dictionary to store results
results = {setting: {} for setting in few_shot_settings.keys()}

# Iterate over few-shot settings
for setting_name, k_shot in few_shot_settings.items():
    logging.info(f"\n=== Few-Shot Setting: {setting_name} ===")

    # Create few-shot datasets
    if k_shot is not None:
        few_shot_train = create_few_shot_dataset(train_dataset, k_shot)
    else:
        few_shot_train = train_dataset  # Use full training data

    # Use the entire validation and test sets
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Create data loader for training
    train_loader = DataLoader(few_shot_train, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over models
    for model_name in model_names:
        logging.info(f"\nTraining model: {model_name} with {setting_name}")
        model = get_model(model_name)

        test_acc = train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=0.0001)
        logging.info(f"Test Accuracy for {model_name} with {setting_name}: {test_acc:.2f}%")

        # Store the result
        results[setting_name][model_name] = test_acc

# Convert results to DataFrames and save as CSV
for setting_name in few_shot_settings.keys():
    df = pd.DataFrame(list(results[setting_name].items()), columns=['Model', 'Test Accuracy (%)'])
    csv_path = os.path.join(results_dir, f"performance_{setting_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nSaved performance table for {setting_name} to {csv_path}")

# Optionally, create a single CSV with all results
all_results = []
for setting_name, models in results.items():
    for model_name, acc in models.items():
        all_results.append({'Few-Shot Setting': setting_name, 'Model': model_name, 'Test Accuracy (%)': acc})

all_df = pd.DataFrame(all_results)
all_csv_path = os.path.join(results_dir, f"performance_all_settings_{timestamp}.csv")
all_df.to_csv(all_csv_path, index=False)
logging.info(f"\nSaved all performance results to {all_csv_path}")

# Optionally, visualize the results using plots
import matplotlib.pyplot as plt
import seaborn as sns

# Create a pivot table for visualization
pivot_df = all_df.pivot(index='Model', columns='Few-Shot Setting', values='Test Accuracy (%)')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Performance on Flowers102 Dataset Under Few-Shot Settings")
plt.ylabel("Model")
plt.xlabel("Few-Shot Setting")
plt.tight_layout()
plt_path = os.path.join(results_dir, f"performance_heatmap_{timestamp}.png")
plt.savefig(plt_path)
plt.close()
logging.info(f"Saved performance heatmap to {plt_path}")

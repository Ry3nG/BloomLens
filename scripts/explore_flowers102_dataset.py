import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.io

# Set the root directory for the dataset
root_dir = './data/flowers102'

# Define a simple transform (we'll just resize the images for exploration)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the training dataset
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=transform, download=True)

# Function to explore dataset statistics
def explore_dataset(dataset, split_name):
    print(f"\n{split_name} Dataset Statistics:")
    print(f"Dataset size: {len(dataset)}")

    # Get all labels
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels = set(labels)

    print(f"Number of classes: {len(unique_labels)}")

    # Count samples per class
    class_counts = Counter(labels)

    print("\nSamples per class:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"Class {class_idx}: {count}")

    # Plot class distribution
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(class_counts)), [class_counts[i] for i in sorted(class_counts.keys())])
    plt.title(f"Class Distribution - {split_name}")
    plt.xlabel("Class Index")
    plt.ylabel("Number of Samples")
    plt.savefig(f"class_distribution_{split_name.lower()}.png")
    plt.close()

# Function to display sample images
def display_sample_images(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        img, label = dataset[idx]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"Class: {label}")
        axes[i].axis('off')
    plt.savefig("sample_images.png")
    plt.close()

# Function to load category names
def load_category_names(root_dir):
    cat_file = scipy.io.loadmat(f"{root_dir}/cat_to_name.mat")
    cat_to_name = {i: name[0] for i, name in enumerate(cat_file['cat_to_name'][0], 1)}
    return cat_to_name

# Main exploration
print("Exploring the 102 Flowers Dataset:")
print("==================================")

# Explore dataset statistics for each split
explore_dataset(train_dataset, "Train")
explore_dataset(val_dataset, "Validation")
explore_dataset(test_dataset, "Test")

# Display sample images
print("\nGenerating sample images (saved as 'sample_images.png')")
display_sample_images(train_dataset)

# Load and display category names
cat_to_name = load_category_names(root_dir)
print("\nCategory Names:")
for i, name in cat_to_name.items():
    print(f"Class {i}: {name}")

print("\nExploration complete. Check the generated images for visual information.")

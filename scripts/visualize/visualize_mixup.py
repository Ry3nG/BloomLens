import torch
import torchvision
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

# Add project root to path to import from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.prototypical_network import mixup_data, cutmix_data


def denormalize(tensor):
    """Convert normalized image tensor back to displayable range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def visualize_augmentations(images, labels, alpha=0.5):
    """Visualize original images and their MixUp/CutMix versions"""
    # Ensure we're working with tensors
    images = images.clone()
    labels = labels.clone()

    # Apply augmentations
    mixed_images, labels_a, labels_b, mixup_lambda = mixup_data(
        images, labels, alpha=alpha
    )
    cutmix_images, cm_labels_a, cm_labels_b, cutmix_lambda = cutmix_data(
        images, labels, alpha=alpha
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Helper function to display image
    def show_image(ax, img, title):
        img = denormalize(img)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    # Show original images
    show_image(axes[0, 0], images[0], f"Original 1\nLabel: {labels[0].item()}")
    show_image(axes[1, 0], images[1], f"Original 2\nLabel: {labels[1].item()}")

    # Show MixUp results
    show_image(
        axes[0, 1],
        mixed_images[0],
        f"MixUp\nλ={mixup_lambda:.2f}\nLabels: {labels_a[0].item()}, {labels_b[0].item()}",
    )
    show_image(
        axes[1, 1],
        mixed_images[1],
        f"MixUp\nλ={mixup_lambda:.2f}\nLabels: {labels_a[1].item()}, {labels_b[1].item()}",
    )

    # Show CutMix results
    show_image(
        axes[0, 2],
        cutmix_images[0],
        f"CutMix\nλ={cutmix_lambda:.2f}\nLabels: {cm_labels_a[0].item()}, {cm_labels_b[0].item()}",
    )
    show_image(
        axes[1, 2],
        cutmix_images[1],
        f"CutMix\nλ={cutmix_lambda:.2f}\nLabels: {cm_labels_a[1].item()}, {cm_labels_b[1].item()}",
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Load some sample images from CIFAR-10
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = torchvision.datasets.Flowers102(
        root="./data", split="train", download=True, transform=transform
    )
    # Select a few images
    batch_size = 4
    indices = torch.randint(0, len(dataset), (batch_size,))
    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])

    # Create visualizations with different alpha values
    alphas = [0.2, 0.5, 0.8]
    for alpha in alphas:
        fig = visualize_augmentations(images, labels, alpha=alpha)
        plt.savefig(f"mixup_cutmix_alpha_{alpha}.png")
        plt.close()

    print("Visualizations saved as 'mixup_cutmix_alpha_*.png'")

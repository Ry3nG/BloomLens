import datetime
import os
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from torchvision.datasets import Flowers102
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.prototypical_network import (
    PrototypicalNetwork,
    compute_prototypes,
    prototypical_loss,
    mixup_data,
    cutmix_data,
    EpisodeSampler,
)

from torch.utils.data import Subset
from collections import defaultdict


def setup_wandb(config):
    # Create a more descriptive name that includes the stage
    run_name = f"{config['n_way']}way_{config['k_shot']}shot"
    if "stage" in config:
        run_name += f"_stage{config['stage']}"

    wandb.init(
        project="flower-proto-net",
        config=config,
        name=run_name,
        # Optional: group runs from the same progressive training session
        group=f"progressive_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        reinit=True,  # Allow multiple wandb.init() calls
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def evaluate_model(model, sampler, device, n_episodes=100):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            # Sample episode
            batch = sampler.sample_episode()

            # Move to device
            support_images = batch["support_images"].to(device)
            support_labels = batch["support_labels"].to(device)
            query_images = batch["query_images"].to(device)
            query_labels = batch["query_labels"].to(device)

            # Forward pass
            support_features = model(support_images)
            query_features = model(query_images)

            # Compute prototypes and loss
            prototypes = compute_prototypes(support_features, support_labels)
            loss, acc = prototypical_loss(
                prototypes, query_features, query_labels, temperature=0.5
            )

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / n_episodes, total_acc / n_episodes


def train_prototypical_network(config, train_dataset, val_dataset, test_sampler=None):
    # Setup
    set_seed(config["seed"])
    setup_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create samplers
    train_sampler = EpisodeSampler(
        train_dataset, config["n_way"], config["k_shot"], config["n_query"]
    )
    val_sampler = EpisodeSampler(
        val_dataset, config["n_way"], config["k_shot"], config["n_query"]
    )

    # Model setup
    model = PrototypicalNetwork(
        backbone="resnet50", feature_dim=config["feature_dim"]
    ).to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Load pretrained weights if available
    if "pretrained_model" in config and config["pretrained_model"] is not None:
        model.load_state_dict(config["pretrained_model"])
        print("Loaded pretrained weights from previous stage")

    # Optimizer setup
    optimizer = optim.AdamW(
        [
            {"params": model.layer_groups[0], "lr": config["lr"] * 0.1},
            {"params": model.layer_groups[1], "lr": config["lr"] * 0.3},
            {"params": model.layer_groups[2], "lr": config["lr"]},
            {"params": model.layer_groups[3], "lr": config["lr"] * 3},
        ],
        weight_decay=config["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config["scheduler_T0"], T_mult=2, eta_min=config["lr"] * 0.01
    )

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        train_acc = 0

        # Training episodes
        for episode in tqdm(range(config["train_episodes"]), desc=f"Epoch {epoch+1}"):
            batch = train_sampler.sample_episode()

            support_images = batch["support_images"].to(device)
            support_labels = batch["support_labels"].to(device)
            query_images = batch["query_images"].to(device)
            query_labels = batch["query_labels"].to(device)

            # Apply augmentation with probability
            if random.random() < config["aug_prob"]:
                if random.random() < 0.5:
                    query_images, labels_a, labels_b, lam = mixup_data(
                        query_images, query_labels, alpha=0.2
                    )
                else:
                    query_images, labels_a, labels_b, lam = cutmix_data(
                        query_images, query_labels, alpha=1.0
                    )

                # Forward pass with mixed data
                support_features = model(support_images)
                query_features = model(query_images)
                prototypes = compute_prototypes(support_features, support_labels)

                # Compute mixed loss
                loss_a, acc_a = prototypical_loss(
                    prototypes, query_features, labels_a, temperature=0.5
                )
                loss_b, acc_b = prototypical_loss(
                    prototypes, query_features, labels_b, temperature=0.5
                )
                loss = lam * loss_a + (1 - lam) * loss_b
                acc = lam * acc_a + (1 - lam) * acc_b
            else:
                # Regular forward pass
                support_features = model(support_images)
                query_features = model(query_images)
                prototypes = compute_prototypes(support_features, support_labels)
                loss, acc = prototypical_loss(
                    prototypes, query_features, query_labels, temperature=0.5
                )

            # Add L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += config["weight_decay"] * l2_reg

            # Optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["grad_clip"]
            )

            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        # Calculate average training metrics
        train_loss = train_loss / config["train_episodes"]
        train_acc = train_acc / config["train_episodes"]

        # Validation
        val_loss, val_acc = evaluate_model(
            model, val_sampler, device, n_episodes=config["val_episodes"]
        )

        # Update learning rate
        scheduler.step()

        # Log metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                save_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

        # Save checkpoint every few epochs
        if (epoch + 1) % config["checkpoint_freq"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                },
                save_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )
    # After training completes and best model is saved,
    # evaluate on test set
    if test_sampler is not None:
        print("\nEvaluating model on test set...")
        # Load best model
        checkpoint = torch.load(save_dir / "best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_acc, test_std = evaluate_on_test(model, test_sampler, device)

        # log test results (this will work now as we're in the same wandb session)
        wandb.log(
            {
                "test_accuracy": test_acc,
                "test_std": test_std,
                "final_test_metrics": True,  # Flag to identify these are final test metrics
            }
        )
        print(f"Test Accuracy: {test_acc:.4f} ± {test_std:.4f}")
        test_metrics = (test_acc, test_std)
    else:
        test_metrics = None
    wandb.finish()
    return best_val_acc, test_metrics


def evaluate_on_test(model, test_sampler, device, n_episodes=50):
    """Evaluate model on test set"""
    model.eval()
    total_acc = 0
    accuracies = []  # Store individual episode accuracies

    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Test Evaluation"):
            batch = test_sampler.sample_episode()

            support_images = batch["support_images"].to(device)
            support_labels = batch["support_labels"].to(device)
            query_images = batch["query_images"].to(device)
            query_labels = batch["query_labels"].to(device)

            support_features = model(support_images)
            query_features = model(query_images)
            prototypes = compute_prototypes(support_features, support_labels)
            _, acc = prototypical_loss(
                prototypes, query_features, query_labels, temperature=0.5
            )

            accuracies.append(acc.item())
            total_acc += acc.item()

    mean_acc = total_acc / n_episodes
    std_acc = np.std(accuracies)

    return mean_acc, std_acc


def split_dataset(dataset, train_ratio=0.6, seed=42):
    """
    Split dataset by classes ensuring no class overlap between train and test.

    Args:
        dataset: Flowers102 dataset
        train_ratio: Ratio of classes to use for training
        seed: Random seed for reproducibility

    Returns:
        (train_indices, test_indices): Indices for train and test sets
        (train_mapping, test_mapping): Class label mappings
        (train_classes, test_classes): Original class labels
    """
    random.seed(seed)

    # Group samples by class
    class_samples = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_samples[label].append(idx)

    # Get all unique classes and shuffle them
    all_classes = list(class_samples.keys())
    random.shuffle(all_classes)

    # Split classes into train and test
    n_train_classes = int(len(all_classes) * train_ratio)
    train_classes = all_classes[:n_train_classes]
    test_classes = all_classes[n_train_classes:]

    # Create new class mappings
    train_mapping = {c: i for i, c in enumerate(train_classes)}
    test_mapping = {c: i for i, c in enumerate(test_classes)}

    # Get indices for train and test sets
    train_indices = [idx for c in train_classes for idx in class_samples[c]]
    test_indices = [idx for c in test_classes for idx in class_samples[c]]

    return (
        (train_indices, test_indices),
        (train_mapping, test_mapping),
        (train_classes, test_classes),
    )


class MappedSubset(Subset):
    """Dataset wrapper that applies a class mapping"""

    def __init__(self, dataset, indices, class_mapping):
        super().__init__(dataset, indices)
        self.class_mapping = class_mapping

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, self.class_mapping[y]


def create_datasets(config):
    """Create train, validation, and test datasets with proper class separation"""
    # Get transforms
    train_transform, eval_transform = get_transforms()

    # Load full dataset
    full_dataset = Flowers102(root="./data", download=True)

    # Split dataset by classes
    (
        (train_indices, test_indices),
        (train_mapping, test_mapping),
        (train_classes, test_classes),
    ) = split_dataset(full_dataset, train_ratio=0.6, seed=config["seed"])

    # Further split train into train and val
    train_val_split = 0.8
    n_train = int(len(train_indices) * train_val_split)
    random.shuffle(train_indices)
    val_indices = train_indices[n_train:]
    train_indices = train_indices[:n_train]

    # Create datasets with appropriate transforms and mappings
    train_dataset = MappedSubset(
        Flowers102(root="./data", transform=train_transform, download=True),
        train_indices,
        train_mapping,
    )

    val_dataset = MappedSubset(
        Flowers102(root="./data", transform=eval_transform, download=True),
        val_indices,
        train_mapping,
    )

    test_dataset = MappedSubset(
        Flowers102(root="./data", transform=eval_transform, download=True),
        test_indices,
        test_mapping,
    )

    print(f"Number of classes - Train: {len(train_classes)}, Test: {len(test_classes)}")
    print(
        f"Number of samples - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    return train_dataset, val_dataset, test_dataset


def progressive_training(resume_stage=None):
    """
    Implement progressive training with increasing n_way
    """
    # Add dataset loading
    train_transform, eval_transform = get_transforms()
    train_dataset = Flowers102(
        root="./data", split="train", transform=train_transform, download=True
    )
    val_dataset = Flowers102(
        root="./data", split="val", transform=eval_transform, download=True
    )
    test_dataset = Flowers102(
        root="./data", split="test", transform=eval_transform, download=True
    )

    base_config = {
        # Base configuration (same as above)
        "feature_dim": 1024,
        "k_shot": 1,
        "n_query": 4,
        "epochs": 100,
        "train_episodes": 100,
        "val_episodes": 50,
        "patience": 20,
        "checkpoint_freq": 5,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "grad_clip": 10.0,
        "scheduler_T0": 10,
        "aug_prob": 0.5,
        "seed": 42,
        "save_dir": "/home/zrgong/data/BloomLens/checkpoints",
    }

    # Progressive training stages
    # Modified progressive training stages
    stages = [
        {"n_way": 5, "epochs": 30, "n_query": 5},
        {"n_way": 10, "epochs": 30, "n_query": 5},
        {"n_way": 20, "epochs": 30, "n_query": 5},
    ]

    # Create test sampler
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=20,
        k_shot=1,
        n_query=5,
    )
    test_results = []

    # Track the model state between stages
    model = None

    for stage_idx, stage in enumerate(stages):
        # Skip stages before resume_stage
        if resume_stage is not None and stage_idx + 1 < resume_stage:
            print(f"Skipping completed stage {stage_idx + 1}")
            # Load the last checkpoint from the previous stage
            last_stage_path = f"/home/zrgong/data/BloomLens/checkpoints/stage_{resume_stage-1}/best_model.pt"
            if os.path.exists(last_stage_path):
                print(f"Loading weights from stage {resume_stage-1}")
                checkpoint = torch.load(last_stage_path)
                model = PrototypicalNetwork(
                    backbone="resnet50", feature_dim=base_config["feature_dim"]
                )
                model.load_state_dict(checkpoint["model_state_dict"])
            continue

        print(f"\nStarting stage {stage_idx + 1}: {stage['n_way']}-way classification")

        current_config = base_config.copy()
        current_config.update(stage)
        current_config["save_dir"] = (
            f"/home/zrgong/data/BloomLens/checkpoints/stage_{stage_idx + 1}"
        )
        current_config["stage"] = stage_idx + 1
        # Pass the model from previous stage
        current_config["pretrained_model"] = model

        best_acc, test_metrics = train_prototypical_network(
            current_config, train_dataset, val_dataset, test_sampler=test_sampler
        )
        # Update model for next stage
        model = torch.load(f"{current_config['save_dir']}/best_model.pt")[
            "model_state_dict"
        ]


if __name__ == "__main__":
    config = {
        # Model configuration
        "feature_dim": 1024,
        "n_way": 5,  # Start with 5-way classification
        "k_shot": 1,  # 1-shot learning
        "n_query": 4,  # Reduced query samples per class (considering ~10 samples per class)
        # Training configuration
        "epochs": 100,
        "train_episodes": 100,  # Reduced episodes per epoch
        "val_episodes": 50,  # Reduced validation episodes
        "patience": 20,
        "checkpoint_freq": 5,
        # Optimization configuration
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 10.0,
        "scheduler_T0": 10,
        # Augmentation configurations
        "aug_prob": 0.5,
        # Other configuration
        "seed": 42,
        "save_dir": "/home/zrgong/data/BloomLens/checkpoints",
    }

    USE_PROGRESSIVE_TRAINING = True

    if USE_PROGRESSIVE_TRAINING:
        progressive_training()
    else:
        # Create datasets with proper class separation
        train_dataset, val_dataset, test_dataset = create_datasets(config)

        # Create test sampler with proper number of classes
        test_sampler = EpisodeSampler(
            test_dataset,
            n_way=min(
                20, len(set(y for _, y in test_dataset))
            ),  # Ensure n_way doesn't exceed available classes
            k_shot=1,
            n_query=5,
        )

        best_acc = train_prototypical_network(
            config, train_dataset, val_dataset, test_sampler
        )
        print(f"Training completed! Best validation accuracy: {best_acc:.4f}")

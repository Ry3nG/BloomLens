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
from src.data.combined_dataset import get_combined_dataset


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

    # After training completes and best model is saved
    if test_sampler is not None:
        print("\nEvaluating model on test set...")
        # Load best model
        checkpoint = torch.load(save_dir / "best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_acc, test_std = evaluate_on_test(model, test_sampler, device)

        # Log test results (this will work now as we're in the same wandb session)
        wandb.log(
            {
                "test_accuracy": test_acc,
                "test_std": test_std,
                "final_test_metrics": True,  # Flag to identify these are final test metrics
            }
        )

        print(f"Test Accuracy: {test_acc:.4f} ± {test_std:.4f}")

    wandb.finish()
    return best_val_acc, (test_acc, test_std) if test_sampler is not None else None


def evaluate_on_test(model, test_sampler, device, n_episodes=100):
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


def progressive_training():
    """Implement progressive training with increasing n_way"""
    # Load datasets once at the start
    train_dataset, val_dataset, test_dataset = get_combined_dataset()

    base_config = {
        # Base configuration (same as above)
        "feature_dim": 1024,
        "k_shot": 1,
        "n_query": 4,
        "epochs": 100,
        "train_episodes": 200,
        "val_episodes": 100,
        "patience": 20,
        "checkpoint_freq": 5,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 10.0,
        "scheduler_T0": 10,
        "aug_prob": 0.5,
        "seed": 42,
        "save_dir": "./results/checkpoints",
    }

    # Progressive training stages
    # Modified progressive training stages
    stages = [
        {"n_way": 5, "epochs": 30, "n_query": 4},
        {"n_way": 5, "epochs": 30, "n_query": 6},
        {"n_way": 10, "epochs": 35, "n_query": 6},
        {"n_way": 20, "epochs": 40, "n_query": 8},
        {"n_way": 20, "epochs": 50, "n_query": 10},
    ]

    # Create test sampler
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=20,  # We'll use maximum n_way for testing
        k_shot=1,  # Keep same as training
        n_query=10,  # Keep same as training
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_results = []

    for stage_idx, stage in enumerate(stages):
        print(f"\nStarting stage {stage_idx + 1}: {stage['n_way']}-way classification")

        # Update configuration for this stage
        current_config = base_config.copy()
        current_config.update(stage)
        current_config["save_dir"] = f"./results/checkpoints/stage_{stage_idx + 1}"
        current_config["stage"] = stage_idx + 1

        # Pass test_sampler to training function
        best_acc, test_metrics = train_prototypical_network(
            current_config, train_dataset, val_dataset, test_sampler=test_sampler
        )

        if test_metrics is not None:
            test_acc, test_std = test_metrics
            test_results.append(
                {
                    "stage": stage_idx + 1,
                    "n_way": stage["n_way"],
                    "test_acc": test_acc,
                    "test_std": test_std,
                }
            )

    # Print summary of all stages
    print("\nTest Results Summary:")
    print("--------------------")
    for result in test_results:
        print(
            f"Stage {result['stage']} ({result['n_way']}-way): "
            f"{result['test_acc']:.4f} ± {result['test_std']:.4f}"
        )


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
        "save_dir": "./results/checkpoints",
    }

    USE_PROGRESSIVE_TRAINING = True

    if USE_PROGRESSIVE_TRAINING:
        progressive_training()
    else:
        # Load datasets once
        train_dataset, val_dataset, _ = get_combined_dataset()
        best_acc = train_prototypical_network(config, train_dataset, val_dataset)
        print(f"Training completed! Best validation accuracy: {best_acc:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import wandb
import logging
from datetime import datetime
import os
import argparse
import torchvision.transforms as T

from config import Config
from models.protonet import PrototypicalNetwork
from dataset.episode_generator import EpisodeGenerator, mixup_episode

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BloomLens model')

    # Add arguments for commonly modified parameters
    parser.add_argument('--data_percentage', type=float, default=1.0,
                      help='Percentage of training data to use (default: 1.0)')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for the experiment (default: auto-generated)')
    parser.add_argument('--backbone', type=str, default='densenet201',
                      help='Backbone architecture (default: densenet201)')
    parser.add_argument('--k_shot', type=int, default=1,
                      help='Number of support examples per class (default: 1)')
    # Add other commonly modified parameters here

    return parser.parse_args()

def setup_config():
    """Create config from command line arguments."""
    args = parse_args()
    # Convert args to dictionary, removing None values
    arg_dict = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**arg_dict)

def get_criterion(config):
    """Get the loss criterion based on config"""
    if config.label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    return nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion, config, device):
    model.train()
    total_loss = 0
    total_acc = 0

    for batch_idx, episode in enumerate(dataloader):
        # Reshape batched episodes
        support_images = episode['support_images'].view(-1, 3, config.image_size, config.image_size).to(device)
        support_labels = episode['support_labels'].view(-1).to(device)
        query_images = episode['query_images'].view(-1, 3, config.image_size, config.image_size).to(device)
        query_labels = episode['query_labels'].view(-1).to(device)

        # Apply mixup if enabled
        if config.use_mixup:
            episode = mixup_episode(episode, config.mixup_alpha)

        # Forward pass
        support_features, query_features = model(support_images, query_images)
        prototypes = model.compute_prototypes(support_features, support_labels)
        logits = model.compute_logits(query_features, prototypes)

        # Compute loss
        if config.use_mixup:
            loss = criterion(logits, episode['query_labels_a'].view(-1).to(device)) * episode['query_lambda'] + \
                   criterion(logits, episode['query_labels_b'].view(-1).to(device)) * (1 - episode['query_lambda'])
        else:
            loss = criterion(logits, query_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred = logits.argmax(dim=1)
        acc = (pred == query_labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

        if batch_idx % 100 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_acc': acc.item()
            })

    return total_loss / len(dataloader), total_acc / len(dataloader)

def validate(model, dataloader, criterion, config, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for episode in dataloader:
            # Reshape batched episodes
            support_images = episode['support_images'].view(-1, 3, config.image_size, config.image_size).to(device)
            support_labels = episode['support_labels'].view(-1).to(device)
            query_images = episode['query_images'].view(-1, 3, config.image_size, config.image_size).to(device)
            query_labels = episode['query_labels'].view(-1).to(device)

            # Forward pass
            support_features, query_features = model(support_images, query_images)
            prototypes = model.compute_prototypes(support_features, support_labels)
            logits = model.compute_logits(query_features, prototypes)

            # Compute loss and accuracy
            loss = criterion(logits, query_labels)
            pred = logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(dataloader), total_acc / len(dataloader)

def setup_logging(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('logs', config.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def main():
    config = setup_config()
    timestamp = setup_logging(config)

    # Initialize wandb with updated config
    wandb.init(
        project="bloomlens",
        name=config.experiment_name,
        config=vars(config)
    )

    # Setup transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(config.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    eval_transform = T.Compose([
        T.Resize(int(config.image_size * 1.14)),
        T.CenterCrop(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    # Load datasets with transforms
    train_dataset = Flowers102(root='./data', split='train', transform=train_transform, download=True)
    val_dataset = Flowers102(root='./data', split='val', transform=eval_transform, download=True)
    test_dataset = Flowers102(root='./data', split='test', transform=eval_transform, download=True)

    # Log dataset statistics
    logging.info(f"Training with {config.data_percentage*100}% of data")
    logging.info(f"K-shot: {config.k_shot}, N-query (training): {config.n_query}")
    logging.info(f"Evaluation N-query: {config.eval_n_query}")

    # Create episode generators
    train_episodes = EpisodeGenerator(train_dataset, config, mode='train')
    val_episodes = EpisodeGenerator(val_dataset, config, mode='val')
    test_episodes = EpisodeGenerator(test_dataset, config, mode='val')

    # Create dataloaders
    train_loader = DataLoader(train_episodes, batch_size=config.batch_size,
                            num_workers=config.num_workers, shuffle=True)
    val_loader = DataLoader(val_episodes, batch_size=config.batch_size,
                          num_workers=config.num_workers)
    test_loader = DataLoader(test_episodes, batch_size=config.batch_size,
                           num_workers=config.num_workers)

    # Log actual episode sizes after adjustments
    logging.info(f"Actual training query size: {train_episodes.n_query}")
    logging.info(f"Actual validation query size: {val_episodes.n_query}")
    logging.info(f"Number of valid classes - Train: {len(train_episodes.valid_classes)}")

    # Create evaluation loaders for validation and test sets
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False
    )

    test_eval_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False
    )

    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork(config).to(device)
    criterion = get_criterion(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_val_acc = 0
    counter = 0

    # Training loop
    for epoch in range(config.num_epochs):
        # Episodic training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config, device)

        # Evaluate on full 102-way classification
        val_acc_full = model.evaluate(
            val_eval_loader,
            train_dataset,
            shots_per_class=config.k_shot
        )

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc_episode': train_acc,
            'val_acc_full': val_acc_full,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Update scheduler
        scheduler.step()

        # Early stopping and model saving
        if val_acc_full > best_val_acc:
            best_val_acc = val_acc_full
            counter = 0
            save_path = os.path.join(config.save_dir, f'best_model_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc_full,
                'config': config
            }, save_path)
            logging.info(f"Saved best model with validation accuracy: {val_acc_full:.2f}%")
        else:
            counter += 1
            if counter >= config.patience:
                logging.info("Early stopping triggered")
                break

    # Final evaluation
    model.load_state_dict(torch.load(save_path)['model_state_dict'])
    test_acc = model.evaluate(
        test_eval_loader,
        train_dataset,
        shots_per_class=config.k_shot
    )
    logging.info(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()

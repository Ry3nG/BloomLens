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

from config import Config
from models.protonet import PrototypicalNetwork
from dataset.episode_generator import EpisodeGenerator, mixup_episode

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
    config = Config()

    # Setup logging and get timestamp
    timestamp = setup_logging(config)

    # Initialize wandb
    wandb.init(project="bloomlens", config=config, name=f"{config.experiment_name}_{timestamp}")

    # Load dataset
    train_dataset = Flowers102(root='./data', split='train', download=True)
    val_dataset = Flowers102(root='./data', split='val', download=True)

    # Create episode generators
    train_episodes = EpisodeGenerator(train_dataset, config, mode='train')
    val_episodes = EpisodeGenerator(val_dataset, config, mode='val')

    # Create dataloaders
    train_loader = DataLoader(
        train_episodes,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_episodes,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork(config).to(device)

    # Initialize criterion
    criterion = get_criterion(config)

    # Initialize optimizer
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    # Initialize scheduler
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )

    # Early stopping parameters
    patience = 10
    best_val_acc = 0
    counter = 0

    # Create save directory
    save_dir = os.path.join('saved_models', config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config, device)

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            save_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, save_path)
            wandb.save(save_path)
            logging.info(f"Saved best model to {save_path}")
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break

        logging.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main()

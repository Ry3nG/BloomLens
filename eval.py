import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from dataset.episode_generator import EpisodeGenerator
from models.protonet import PrototypicalNetwork
from config import Config

def evaluate(model, dataloader, config, device, num_episodes=600):
    model.eval()
    accuracies = []

    with torch.no_grad():
        for _ in range(num_episodes):
            episode = next(iter(dataloader))

            # Move episode to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)

            # Forward pass
            support_features, query_features = model(support_images, query_images)
            prototypes = model.compute_prototypes(support_features, support_labels)
            logits = model.compute_logits(query_features, prototypes)

            # Compute accuracy
            pred = logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean()
            accuracies.append(acc.item())

    mean_acc = np.mean(accuracies)
    conf_interval = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))

    return mean_acc, conf_interval

def main():
    # Load config from saved model
    checkpoint = torch.load('best_model.pth')
    config = checkpoint['config']

    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test dataset
    test_dataset = Flowers102(root='./data', split='test', download=True)

    # Create episode generator for different configurations
    configs = [
        (5, 1),  # 5-way 1-shot
        (5, 5),  # 5-way 5-shot
        (10, 5), # 10-way 5-shot
    ]

    results = {}
    for n_way, k_shot in configs:
        config.n_way = n_way
        config.k_shot = k_shot

        test_episodes = EpisodeGenerator(test_dataset, config, mode='test')
        test_loader = DataLoader(
            test_episodes,
            batch_size=1,
            num_workers=config.num_workers
        )

        acc, conf = evaluate(model, test_loader, config, device)
        results[f"{n_way}-way {k_shot}-shot"] = {
            'accuracy': acc,
            'confidence_interval': conf
        }

        print(f"{n_way}-way {k_shot}-shot: {acc:.4f} ± {conf:.4f}")

if __name__ == '__main__':
    main()

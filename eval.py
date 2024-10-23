import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import pandas as pd
import logging
import os
from datetime import datetime

from config import Config
from models.protonet import PrototypicalNetwork

def evaluate_model(model_path, shots_list=[1, 5, 10]):
    """
    Evaluate a trained model with different numbers of shots.
    """
    # Load model and config
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    # Initialize model
    model = PrototypicalNetwork(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load datasets
    train_dataset = Flowers102(root='./data', split='train', download=True)
    test_dataset = Flowers102(root='./data', split='test', download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Evaluate with different numbers of shots
    results = []
    for shots in shots_list:
        accuracy = model.evaluate(test_loader, train_dataset, shots_per_class=shots)
        results.append({
            'Shots': shots,
            'Accuracy': accuracy
        })
        logging.info(f"{shots}-shot accuracy: {accuracy:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'results/protonet_evaluation_{timestamp}.csv', index=False)

    return results_df

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Evaluate model
    model_path = 'best_model.pth'
    results = evaluate_model(model_path)

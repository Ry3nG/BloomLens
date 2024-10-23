import torch
from config import Config
from models.protonet import PrototypicalNetwork

def test_model():
    config = Config()
    model = PrototypicalNetwork(config)

    # Create dummy data
    support_images = torch.randn(5, 3, 224, 224)  # 5-way, 1-shot
    query_images = torch.randn(25, 3, 224, 224)   # 5 query images per class

    # Forward pass
    support_features, query_features = model(support_images, query_images)

    # Check outputs
    print(f"Support features shape: {support_features.shape}")
    print(f"Query features shape: {query_features.shape}")

    # Test prototype computation
    support_labels = torch.tensor([0, 1, 2, 3, 4])
    prototypes = model.compute_prototypes(support_features, support_labels)
    print(f"Prototypes shape: {prototypes.shape}")

    # Test logits computation
    logits = model.compute_logits(query_features, prototypes)
    print(f"Logits shape: {logits.shape}")

if __name__ == '__main__':
    test_model()

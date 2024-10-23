import torch
import torch.nn as nn
import timm
from .attention import TaskAttention, SelfAttention
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Learning.

    This class implements a Prototypical Network with optional attention mechanisms
    for few-shot image classification tasks. It uses a pre-trained backbone for
    feature extraction and includes embedding projection, task attention, and
    self-attention modules.

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        backbone (nn.Module): Pre-trained backbone network for feature extraction.
        embedding (nn.Sequential): Embedding projection layers.
        task_attention (TaskAttention): Optional task-specific attention module.
        self_attention (SelfAttention): Optional self-attention module.
    """

    def __init__(self, config):
        """
        Initialize the Prototypical Network.

        Args:
            config (object): Configuration object with model settings.
        """
        super().__init__()
        self.config = config

        # Initialize backbone
        if config.backbone == 'densenet201':
            self.backbone = timm.create_model(
                'densenet201',
                pretrained=config.pretrained,
                num_classes=0,  # Remove classifier
                global_pool=''  # Remove global pooling
            )
            self.backbone_dim = 1920  # DenseNet201's feature dim
        elif config.backbone == 'resnet50':
            self.backbone = timm.create_model(
                'resnet50',
                pretrained=config.pretrained,
                num_classes=0,
                global_pool=''
            )
            self.backbone_dim = 2048  # ResNet50's feature dim
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        # Add global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

        # Optional attention mechanisms
        if config.use_task_attention:
            self.task_attention = TaskAttention(config.embedding_dim)
        if config.use_self_attention:
            self.self_attention = SelfAttention(config.embedding_dim)

    def forward_features(self, x):
        """
        Extract and process features from input images.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Normalized embeddings of shape (batch_size, embedding_dim).
        """
        # Extract backbone features
        x = self.backbone(x)  # [B, C, H, W]
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]

        # Project to embedding space
        x = self.embedding(x)  # [B, embedding_dim]

        if self.config.use_self_attention:
            x = x.unsqueeze(1)  # Add sequence dimension for attention
            x = self.self_attention(x)
            x = x.squeeze(1)

        return F.normalize(x, dim=-1)

    def forward(self, support_images, query_images):
        """
        Forward pass for the Prototypical Network.

        Args:
            support_images (torch.Tensor): Support set images.
            query_images (torch.Tensor): Query set images.

        Returns:
            tuple: A tuple containing:
                - support_features (torch.Tensor): Processed support set features.
                - query_features (torch.Tensor): Processed query set features.
        """
        # Extract features
        support_features = self.forward_features(support_images)
        query_features = self.forward_features(query_images)

        # Apply task attention if enabled
        if self.config.use_task_attention:
            attention_weights = self.task_attention(support_features, query_features)
            query_features = query_features * attention_weights

        return support_features, query_features

    def compute_prototypes(self, support_features, support_labels):
        """
        Compute class prototypes from support features.

        Args:
            support_features (torch.Tensor): Features of the support set.
            support_labels (torch.Tensor): Labels of the support set.

        Returns:
            torch.Tensor: Normalized class prototypes.
        """
        classes = torch.unique(support_labels)
        prototypes = torch.zeros(len(classes), support_features.shape[-1]).to(support_features.device)

        for idx, c in enumerate(classes):
            mask = support_labels == c
            class_features = support_features[mask]
            if self.config.use_task_attention:
                weights = self.task_attention(class_features, class_features)
                prototypes[idx] = (class_features * weights).sum(0)
            else:
                prototypes[idx] = class_features.mean(0)

        return F.normalize(prototypes, dim=-1)

    def compute_logits(self, query_features, prototypes):
        """
        Compute classification logits for query features.

        Args:
            query_features (torch.Tensor): Features of the query set.
            prototypes (torch.Tensor): Class prototypes.

        Returns:
            torch.Tensor: Classification logits.
        """
        if self.config.distance_metric == "euclidean":
            distances = torch.cdist(query_features, prototypes)
            logits = -distances * self.config.temperature
        else:  # cosine
            logits = torch.matmul(query_features, prototypes.t()) * self.config.temperature
        return logits
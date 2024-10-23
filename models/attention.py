import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAttention(nn.Module):
    """
    Task-specific attention mechanism for few-shot learning.

    This module calculates attention weights for support and query features,
    allowing the model to focus on relevant information for the current task.

    Args:
        embed_dim (int): The dimension of the input feature embeddings.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, support_features, query_features):
        """
        Compute task-specific attention weights for support and query features.

        Args:
            support_features (torch.Tensor): Features of the support set, shape (n_support, embed_dim).
            query_features (torch.Tensor): Features of the query set, shape (n_query, embed_dim).

        Returns:
            torch.Tensor: Attention weights for query features, shape (n_query, 1).
        """
        # Calculate attention weights for support set
        support_weights = self.attention(support_features).softmax(0)

        # Weight support features by attention and sum them
        weighted_support = (support_features * support_weights).sum(0, keepdim=True)

        # Calculate query attention by comparing query features with weighted support
        query_weights = torch.matmul(query_features, weighted_support.t()).softmax(1)

        return query_weights

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature refinement.

    This module applies self-attention to input features, allowing each feature
    to attend to all other features in the input.

    Args:
        in_dim (int): The dimension of the input features.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        """
        Apply self-attention to the input features.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, seq_len, in_dim).

        Returns:
            torch.Tensor: Refined features after self-attention, same shape as input.
        """
        # Generate query, key, and value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute scaled dot-product attention
        attention = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = F.softmax(attention, dim=-1)

        # Apply attention weights to values
        return torch.matmul(attention, v)

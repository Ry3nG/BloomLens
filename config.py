from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

@dataclass
class Config:
    """
    Configuration class for a Few-Shot Learning model using Prototypical Networks.

    This class contains all the hyperparameters and settings for training and
    evaluating the model, including model architecture, few-shot settings,
    training parameters, and advanced configurations.
    """

    # Experiment Configuration
    experiment_name: str = field(default_factory=lambda: f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    """Unique name for the experiment, used for logging and saving models."""

    # Model configuration
    backbone: str = "densenet201"
    """The backbone architecture for feature extraction (e.g., 'densenet201', 'resnet50')."""

    embedding_dim: int = 512
    """Dimension of the embedding space for prototypes and queries."""

    pretrained: bool = True
    """Whether to use pretrained weights for the backbone."""

    # Few-shot configuration
    n_way: int = 5
    """Number of classes in each few-shot task."""

    k_shot: int = 1
    """Number of support examples per class."""

    n_query: int = 5
    """Number of query examples per class."""

    # Training Configuration
    num_epochs: int = 100
    """Total number of training epochs."""

    batch_size: int = 1
    """Number of episodes per batch."""

    learning_rate: float = 0.0001
    """Initial learning rate for the optimizer."""

    weight_decay: float = 0.0001
    """L2 regularization factor."""

    # Prototypical Network Configuration
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    """Distance metric used for prototype comparison ('euclidean' or 'cosine')."""

    temperature: float = 64.0
    """Temperature scaling factor for softmax in the classification layer."""

    # Advanced Configuration
    use_task_attention: bool = True
    """Enable task-specific attention mechanism."""

    use_self_attention: bool = True
    """Enable self-attention mechanism in the backbone."""

    use_mixup: bool = True
    """Enable episode-level mixup for data augmentation."""

    mixup_alpha: float = 0.2
    """Alpha parameter for the beta distribution in mixup."""

    # Regularization
    dropout_rate: float = 0.1
    """Dropout rate applied in the model."""

    label_smoothing: float = 0.1
    """Label smoothing factor for loss calculation."""

    # Optimizer Configuration
    optimizer: str = "adamw"
    """Optimizer choice ('adam', 'adamw', or 'sgd')."""

    scheduler: str = "cosine"
    """Learning rate scheduler ('cosine', 'step', or 'none')."""

    warmup_epochs: int = 10
    """Number of epochs for learning rate warm-up."""

    # Data Configuration
    image_size: int = 224
    """Size of the input images (assumes square images)."""

    num_workers: int = 4
    """Number of worker processes for data loading."""

    # Logging and Saving Configuration
    log_dir: str = "logs"
    """Directory for storing log files."""

    save_dir: str = "saved_models"
    """Directory for saving model checkpoints."""

    save_frequency: int = 1
    """Frequency (in epochs) for saving model checkpoints."""

    # Early Stopping Configuration
    patience: int = 10
    """Number of epochs to wait for improvement before early stopping."""

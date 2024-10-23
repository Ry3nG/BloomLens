from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

@dataclass
class Config:
    """Configuration class for Few-Shot Learning model."""

    def __init__(self, **kwargs):
        """Initialize config with optional overrides from command line."""
        # Set defaults first
        self.set_defaults()
        # Override with any provided arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Post-initialization processing
        self._post_init()

    def set_defaults(self):
        """Set default values for all configuration parameters."""
        # Experiment Configuration
        self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backbone = "densenet201"
        self.embedding_dim = 512
        self.pretrained = True

        # Few-shot configuration
        self.n_way = 5
        self.k_shot = 1
        self.n_query = 5  # Training query size (will be automatically adjusted)
        self.eval_n_query = 5  # Evaluation query size (fixed)

        # Training Configuration
        self.num_epochs = 100
        self.batch_size = 1
        self.learning_rate = 0.00001
        self.weight_decay = 0.0001

        # Prototypical Network Configuration
        self.distance_metric = "euclidean"
        self.temperature = 64.0

        # Advanced Configuration
        self.use_task_attention = True
        self.use_self_attention = True
        self.use_mixup = True
        self.mixup_alpha = 0.2

        # Regularization
        self.dropout_rate = 0.1
        self.label_smoothing = 0.1

        # Optimizer Configuration
        self.optimizer = "adamw"
        self.scheduler = "cosine"
        self.warmup_epochs = 10

        # Data Configuration
        self.image_size = 224
        self.num_workers = 4

        # Logging and Saving Configuration
        self.log_dir = "logs"
        self.save_dir = "saved_models"
        self.save_frequency = 1

        # Early Stopping Configuration
        self.patience = 10

        # Data percentage
        self.data_percentage = 1.0
        self.min_samples_per_class = 1  # Minimum samples required per class

    def _post_init(self):
        """Validate and process configuration after initialization."""
        if self.data_percentage != 1.0:
            self.experiment_name = f"{self.experiment_name}_data{int(self.data_percentage*100)}pct"
            # Adjust n_query based on expected data reduction
            expected_samples = 10 * self.data_percentage  # 10 is the original samples per class
            self.n_query = max(1, min(self.n_query, int(expected_samples) - self.k_shot))
            print(f"Adjusted n_query to {self.n_query} for training")

        # Additional validation
        if self.k_shot < 1:
            raise ValueError("k_shot must be at least 1")
        if self.n_query < 1:
            raise ValueError("n_query must be at least 1")
        if self.eval_n_query < 1:
            raise ValueError("eval_n_query must be at least 1")

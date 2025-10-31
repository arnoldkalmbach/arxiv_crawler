"""Configuration dataclasses for citation embedding training and evaluation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data"
    hf_dataset_name: Optional[str] = None
    citations_batch_size: int = 10000

    # Collation settings
    max_length: int = 256
    pad_to_multiple_of: int = 8
    pad_value: float = 0.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    hidden_size: int = 768
    num_hidden_layers: int = 1
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    max_position_embeddings: int = 2048


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 256
    num_workers: int = 6
    num_epochs: int = 20
    learning_rate: float = 1e-3

    # Logging
    log_steps: int = 10
    save_steps: int = 500
    tensorboard_dir: str = "runs"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    batch_size: int = 256
    num_workers: int = 6
    max_batches: Optional[int] = None


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

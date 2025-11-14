"""Configuration dataclasses for citation embedding training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data"
    hf_dataset_name: Optional[str] = None
    citations_batch_size: int = 10000
    basemodel_name: str = "sentence-transformers/allenai-specter"  # SentenceTransformer base model for embeddings

    # Dataset building settings
    embedding_batch_size: int = 1000  # Batch size for processing citation embeddings
    test_size: float = 0.0  # Fraction of papers to use for test set
    random_seed: int = 42  # Random seed for train/test split

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
class RectflowConfig:
    """Rectified flow model configuration."""

    # Velocity field architecture
    velocity_field_type: str = "DiT"  # "DiT" or "CrossAttention"
    num_blocks: int = 1  # Number of transformer blocks in velocity field
    num_heads: int = 4  # Number of attention heads
    mlp_ratio: float = 4.0  # MLP expansion ratio

    # Conditioning model
    conditioning_checkpoint: str = "models/model_500.pth"  # Path to conditioning model checkpoint


@dataclass
class RectflowTrainingConfig:
    """Rectified flow training configuration."""

    batch_size: int = 256
    num_workers: int = 6
    num_epochs: int = 20
    learning_rate: float = 1e-3

    # Logging
    log_steps: int = 10
    save_steps: int = 500
    tensorboard_dir: str = "runs_rectflow"


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    rectflow: RectflowConfig = field(default_factory=RectflowConfig)
    rectflow_training: RectflowTrainingConfig = field(default_factory=RectflowTrainingConfig)


def load_config(config_file: Optional[Path] = None, cli_overrides: Optional[list[str]] = None) -> Config:
    """
    Load configuration from YAML file and merge with CLI overrides.

    This function should be used with argparse.parse_known_args() to separate
    normal CLI arguments from config-style overrides (e.g., training.lr=1e-4).

    Args:
        config_file: Path to YAML config file. Defaults to "configs/default.yaml"
        cli_overrides: List of config override strings from parse_known_args().
                      Example: ["training.lr=1e-4", "model.hidden_size=512"]

    Returns:
        Merged configuration object

    Example:
        ```python
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--device", type=str, default="cuda")
            args, unknown = parser.parse_known_args()
            return args, unknown

        def main():
            args, unknown = parse_args()
            cfg = load_config(cli_overrides=unknown)
            # Use cfg and args...
        ```
    """
    # Default config file path
    if config_file is None:
        config_file = Path("configs/default.yaml")

    # Load structured config (provides schema and defaults)
    schema = OmegaConf.structured(Config)

    # Load from YAML file
    if config_file.exists():
        yaml_conf = OmegaConf.load(config_file)
        conf = OmegaConf.merge(schema, yaml_conf)
    else:
        print(f"Warning: Config file {config_file} not found, using defaults")
        conf = schema

    # Merge CLI config overrides (highest priority)
    if cli_overrides:
        cli_cfg = OmegaConf.from_cli(cli_overrides)
        conf = OmegaConf.merge(conf, cli_cfg)

    return conf

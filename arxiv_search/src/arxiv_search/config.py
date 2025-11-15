"""Configuration dataclasses for citation embedding training and evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
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
    conditioning_checkpoint: Optional[str] = None  # Path to conditioning model checkpoint (None = auto-find latest)


@dataclass
class RectflowTrainingConfig:
    """Rectified flow training configuration."""

    batch_size: int = 128
    num_workers: int = 6
    num_epochs: int = 100
    learning_rate: float = 2e-4

    # Time weighting
    train_time_weight: str = "linear"  # "linear" or "exponential"
    train_time_distribution: str = "lognormal"  # "uniform" or "lognormal"

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


def generate_run_name(cfg: Config, experiment_type: str = "direct") -> str:
    """
    Generate a descriptive run name based on config parameters.

    The run name includes key hyperparameters that are commonly varied,
    making it easy to identify and compare experiments.

    Args:
        cfg: Configuration object
        experiment_type: Type of experiment ("direct" or "rectflow")

    Returns:
        A string like "lr1e-3_bs256_layers1" or "rectflow_DiT_blocks2_lr2e-4"
    """
    parts = []

    if experiment_type == "direct":
        # Include key training and model parameters
        parts.append(f"lr{cfg.training.learning_rate:.0e}")
        parts.append(f"bs{cfg.training.batch_size}")
        parts.append(f"layers{cfg.model.num_hidden_layers}")
        parts.append(f"hidden{cfg.model.hidden_size}")
        parts.append(f"heads{cfg.model.num_attention_heads}")
        if cfg.model.intermediate_size != cfg.model.hidden_size * 2:
            parts.append(f"inter{cfg.model.intermediate_size}")
        if cfg.data.max_length != 256:
            parts.append(f"maxlen{cfg.data.max_length}")
    elif experiment_type == "rectflow":
        # Include rectflow-specific parameters
        parts.append(f"rectflow_{cfg.rectflow.velocity_field_type}")
        parts.append(f"blocks{cfg.rectflow.num_blocks}")
        parts.append(f"heads{cfg.rectflow.num_heads}")
        parts.append(f"lr{cfg.rectflow_training.learning_rate:.0e}")
        parts.append(f"bs{cfg.rectflow_training.batch_size}")
        parts.append(f"timew{cfg.rectflow_training.train_time_weight}")
        parts.append(f"timedist{cfg.rectflow_training.train_time_distribution}")
        if cfg.rectflow.mlp_ratio != 4.0:
            parts.append(f"mlpr{cfg.rectflow.mlp_ratio}")

    return "_".join(parts)


def find_latest_checkpoint(base_dir: str, pattern: str = "model_*.pth") -> Optional[Path]:
    """
    Find the latest checkpoint in a directory or its subdirectories.

    Searches for checkpoints matching the pattern, prioritizing checkpoints
    in subdirectories (run directories) over the base directory itself.

    Args:
        base_dir: Base directory to search (e.g., "runs" or "models")
        pattern: Glob pattern to match checkpoint files (default: "model_*.pth")

    Returns:
        Path to the latest checkpoint, or None if none found
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    # First, check subdirectories (run directories) for checkpoints
    checkpoints = []
    for run_dir in base_path.iterdir():
        if run_dir.is_dir():
            # Check in checkpoints subdirectory first
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoints.extend(checkpoints_dir.glob(pattern))
            # Also check directly in run directory (for backwards compatibility)
            checkpoints.extend(run_dir.glob(pattern))

    # If no checkpoints in subdirectories, check base directory
    if not checkpoints:
        checkpoints = list(base_path.glob(pattern))

    if not checkpoints:
        return None

    # Sort by step number (extract from filename like "model_1000.pth")
    def get_step(path: Path) -> int:
        try:
            # Extract number from "model_1000.pth"
            name = path.stem  # "model_1000"
            step_str = name.split("_")[-1]
            return int(step_str)
        except (ValueError, IndexError):
            return 0

    # Return checkpoint with highest step number
    return max(checkpoints, key=get_step)


def setup_run_directory(
    base_dir: str,
    cfg: Config,
    experiment_type: str = "direct",
    run_name: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    """
    Set up a run directory with a descriptive name and save the config.

    Creates a directory structure like:
        base_dir/
            run_name/
                config.yaml
                checkpoints/
                    (checkpoints will be saved here)
                (tensorboard logs will be written here)

    Args:
        base_dir: Base directory for runs (e.g., "runs" or "runs_rectflow")
        cfg: Configuration object
        experiment_type: Type of experiment ("direct" or "rectflow")
        run_name: Optional custom run name. If None, generates one from config.

    Returns:
        Tuple of (run_directory, tensorboard_directory, checkpoints_directory)
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Generate run name if not provided
    if run_name is None:
        run_name = generate_run_name(cfg, experiment_type)

    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints subdirectory
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Config saved to: {config_path}")

    # TensorBoard directory is the same as run directory
    return run_dir, run_dir, checkpoints_dir

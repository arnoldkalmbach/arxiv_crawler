"""arXiv Search package for citation embedding model training and evaluation."""

# Configuration
from arxiv_search.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
)

# Data loading
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    collate_embeddings_with_targets,
    ensure_dataset_exists,
    get_collate_fn,
)

# Model
from arxiv_search.model import (
    create_model,
    load_model,
)

# Training and evaluation
from arxiv_search.training import (
    train,
    train_epoch,
    evaluate,
    print_metrics,
    save_metrics,
)

__all__ = [
    # Configuration
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    # Data loading
    "CitationEmbeddingDataset",
    "collate_embeddings_with_targets",
    "ensure_dataset_exists",
    "get_collate_fn",
    # Model
    "create_model",
    "load_model",
    # Training and evaluation
    "train",
    "train_epoch",
    "evaluate",
    "print_metrics",
    "save_metrics",
]

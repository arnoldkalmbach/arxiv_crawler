"""Evaluation script for citation embedding model."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from arxiv_search.config import Config
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
    get_collate_fn,
)
from arxiv_search.model import load_model
from arxiv_search.training import evaluate, print_metrics, save_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate citation embedding model on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (cuda or cpu)",
    )
    return parser.parse_args()


def load_config() -> Config:
    """
    Load configuration from YAML file and merge with CLI overrides.

    Returns:
        Merged configuration object
    """
    # Load structured config (provides schema and defaults)
    schema = OmegaConf.structured(Config)

    # Load from default YAML file
    config_file = Path("configs/default.yaml")
    if config_file.exists():
        yaml_conf = OmegaConf.load(config_file)
        conf = OmegaConf.merge(schema, yaml_conf)
    else:
        print(f"Warning: Config file {config_file} not found, using defaults")
        conf = schema

    # Merge CLI arguments (highest priority)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    return conf


def main():
    """Main evaluation function."""
    # Parse required arguments
    args = parse_args()

    # Load configuration
    cfg = load_config()

    # Convert paths
    data_dir = Path(cfg.data.data_dir)
    model_path = Path(args.model_path)

    # Print configuration
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Device: {args.device}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print()

    # Ensure dataset exists (download if necessary)
    data_dir = ensure_dataset_exists(data_dir, cfg.data.hf_dataset_name)

    # Check if test split exists
    test_citations = data_dir / "test" / "citations.jsonl"
    if not test_citations.exists():
        raise FileNotFoundError(f"Test split not found at {test_citations}. Please ensure the test split is available.")

    # Create test dataset
    print("Loading test dataset...")
    test_ds = CitationEmbeddingDataset(
        citations_file=test_citations,
        paper_embeddings_file=data_dir / "paper_embeddings.parquet",
        citation_embeddings_dir=data_dir / "test",
        citations_batch_size=cfg.data.citations_batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        shuffle_shards=False,
    )

    # Create dataloader
    collate_fn = get_collate_fn(
        max_length=cfg.data.max_length,
        pad_to_multiple_of=cfg.data.pad_to_multiple_of,
        pad_value=cfg.data.pad_value,
        mask_dtype=torch.bool,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.evaluation.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.evaluation.num_workers > 0 else False,
        pin_memory=True if args.device == "cuda" else False,
        prefetch_factor=3 if cfg.evaluation.num_workers > 0 else None,
    )
    print("Test dataset loaded.")

    # Load model
    print(f"\nLoading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = load_model(
        str(model_path),
        device=args.device,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
    )
    print("Model loaded successfully.")

    # Run evaluation
    print("\nStarting evaluation...\n")
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=args.device,
        max_batches=cfg.evaluation.max_batches,
        show_progress=True,
    )

    # Print results
    print_metrics(metrics)

    # Save metrics to file (next to model checkpoint)
    metrics_path = model_path.parent / f"{model_path.stem}_eval_metrics.txt"
    save_metrics(metrics, metrics_path, model_path)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

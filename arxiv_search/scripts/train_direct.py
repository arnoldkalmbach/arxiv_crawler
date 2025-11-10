"""Training script for citation embedding model."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from arxiv_search.config import load_config
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
    get_collate_fn,
)
from arxiv_search.model import create_model
from arxiv_search.training import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train citation embedding model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    # Use parse_known_args to separate normal args from config overrides
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    """Main training function."""
    # Parse required arguments and collect config overrides
    args, unknown = parse_args()

    # Load configuration (merges default.yaml with CLI config overrides)
    cfg = load_config(cli_overrides=unknown)

    # Convert paths
    data_dir = Path(cfg.data.data_dir)
    models_dir = Path(args.models_dir)
    tensorboard_dir = Path(cfg.training.tensorboard_dir)

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Models directory: {models_dir}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print()

    # Ensure dataset exists (download if necessary)
    data_dir = ensure_dataset_exists(data_dir, cfg.data.hf_dataset_name)

    # Create train dataset
    print("Loading training dataset...")
    train_ds = CitationEmbeddingDataset(
        citations_file=data_dir / "train/citations.jsonl",
        paper_embeddings_file=data_dir / "paper_embeddings.parquet",
        citation_embeddings_dir=data_dir / "train",
        citations_batch_size=cfg.data.citations_batch_size,
    )
    print("Created train dataset")

    # Create dataloader
    collate_fn = get_collate_fn(
        max_length=cfg.data.max_length,
        pad_to_multiple_of=cfg.data.pad_to_multiple_of,
        pad_value=cfg.data.pad_value,
        mask_dtype=torch.bool,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.training.num_workers > 0 else False,
        pin_memory=True if args.device == "cuda" else False,
        prefetch_factor=3 if cfg.training.num_workers > 0 else None,
    )
    print("Created train loader")

    # Create model
    print("\nCreating model...")
    model = create_model(
        device=args.device,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
    )
    print("Model created")

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    # Train model
    print(f"\nStarting training for {cfg.training.num_epochs} epochs...")
    print("=" * 60)

    train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=cfg.training.num_epochs,
        log_steps=cfg.training.log_steps,
        save_steps=cfg.training.save_steps,
        save_dir=models_dir,
        tensorboard_dir=tensorboard_dir,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {models_dir}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"\nView training progress with: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()

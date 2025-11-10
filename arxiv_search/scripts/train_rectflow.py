"""Training script for citation embedding model."""

import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from arxiv_search.config import Config
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
)
from arxiv_search.iterable_coupling_dataset import IterableCouplingDataset, get_coupling_collate_fn
from arxiv_search.model import load_model
from rectified_flow import RectifiedFlow
from arxiv_search.training_rectflow import train

import torch.distributions as dist
from arxiv_search.model import VelocityField1dCrossAttention


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--conditioning-checkpoint",
        type=str,
        default="models/model_500.pth",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="rectflow_models",
        help="Directory to save model checkpoints",
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
    """Main training function."""
    # Parse required arguments
    args = parse_args()

    # Load configuration
    cfg = load_config()

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

    train_ds = IterableCouplingDataset(
        D1=CitationEmbeddingDataset(
            citations_file=data_dir / "train/citations.jsonl",
            paper_embeddings_file=data_dir / "paper_embeddings.parquet",
            citation_embeddings_dir=data_dir / "train",
            citations_batch_size=cfg.data.citations_batch_size,
        ),
        D0=dist.Normal(loc=0.0, scale=1.0),
        extract_target=lambda x: x[1],
        extract_conditioning=lambda x: x[0],
    )


    # Create dataloader
    collate_fn = get_coupling_collate_fn(
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
    conditioning_model = load_model(args.conditioning_checkpoint)
    conditioning_model.requires_grad_(False)

    velocity_model = VelocityField1dCrossAttention(1, conditioning_model, 4, 4.0).to(args.device)
    # velocity_model = VelocityField1dDiT(1, conditioning_model, 4, 4.0)

    rectified_flow = RectifiedFlow(
        data_shape=(cfg.data.max_length, conditioning_model.config.hidden_size),
        velocity_field=velocity_model,
        device=args.device,
    )

    # Create optimizer
    optimizer = optim.AdamW((param for param in velocity_model.parameters() if param.requires_grad), lr=cfg.training.learning_rate)

    # Train model
    print(f"\nStarting training for {cfg.training.num_epochs} epochs...")
    print("=" * 60)

    train(
        rectified_flow=rectified_flow,
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

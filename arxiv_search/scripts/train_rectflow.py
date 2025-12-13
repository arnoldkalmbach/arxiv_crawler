"""Training script for citation embedding model."""

import argparse
import math
from pathlib import Path

import torch
import torch.distributions as dist
import torch.optim as optim
from omegaconf import OmegaConf
from rectified_flow import RectifiedFlow
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from arxiv_search.config import find_latest_checkpoint, load_config, setup_run_directory
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
)
from arxiv_search.iterable_coupling_dataset import IterableCouplingDataset, get_coupling_collate_fn
from arxiv_search.model import VelocityField1dCrossAttention, VelocityField1dDiT, load_model
from arxiv_search.training_rectflow import train


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
        default=None,
        help="Path to conditioning model checkpoint (default: auto-finds latest in runs/)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: saves to run_dir/checkpoints/)",
    )
    # Use parse_known_args to separate normal args from config overrides
    args, unknown = parser.parse_known_args()
    return args, unknown


def _estimate_total_steps(dataloader: DataLoader, num_epochs: int, fallback_per_epoch: int = 1000) -> int:
    """
    Estimate total training steps for scheduler.

    If the dataloader has no length (e.g., IterableDataset), fall back to a sensible
    per-epoch step count to keep scheduler progression stable.
    """
    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = None

    if steps_per_epoch is None or steps_per_epoch <= 0:
        steps_per_epoch = fallback_per_epoch

    return max(1, num_epochs * steps_per_epoch)


def _build_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """
    Create LR lambda implementing linear warmup followed by cosine decay.
    """
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = min(1.0, max(0.0, (step - warmup_steps) / float(total_steps - warmup_steps)))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)

    return lr_lambda


def main():
    """Main training function."""
    # Parse required arguments and collect config overrides
    args, unknown = parse_args()

    # Load configuration (merges default.yaml with CLI config overrides)
    cfg = load_config(cli_overrides=unknown)

    # Convert paths
    data_dir = Path(cfg.data.data_dir)

    # Set up run directory with config-based naming
    run_dir, tensorboard_dir, checkpoints_dir = setup_run_directory(
        base_dir=cfg.rectflow_training.tensorboard_dir,
        cfg=cfg,
        experiment_type="rectflow",
    )

    # Use checkpoints_dir unless explicitly overridden
    models_dir = Path(args.models_dir) if args.models_dir else checkpoints_dir

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Checkpoints directory: {models_dir}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print()

    # Ensure dataset exists (download if necessary)
    data_dir = ensure_dataset_exists(data_dir, cfg.data.hf_dataset_name)

    # Do not request metadata during training to reduce CPU/IO overhead.
    # Deterministic pairing falls back to hashing the target tensor when no key is provided.
    train_ds = IterableCouplingDataset(
        D1=CitationEmbeddingDataset(
            citations_file=data_dir / "train/citations.jsonl",
            paper_embeddings_file=data_dir / "paper_embeddings.parquet",
            citation_embeddings_dir=data_dir / "train",
            citations_batch_size=cfg.data.citations_batch_size,
            return_metadata=True,
        ),
        D0=dist.Normal(loc=0.0, scale=1.0),
        extract_target=lambda x: x[1],
        extract_key=lambda x: x[2]["reference_id"],
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
        batch_size=cfg.rectflow_training.batch_size,
        num_workers=cfg.rectflow_training.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.rectflow_training.num_workers > 0 else False,
        pin_memory=True if args.device == "cuda" else False,
        prefetch_factor=3 if cfg.rectflow_training.num_workers > 0 else None,
    )
    print("Created train loader")

    # Create model
    print("\nCreating model...")
    # Determine conditioning checkpoint path
    if args.conditioning_checkpoint is not None:
        conditioning_checkpoint = args.conditioning_checkpoint
    elif cfg.rectflow.conditioning_checkpoint and Path(cfg.rectflow.conditioning_checkpoint).exists():
        conditioning_checkpoint = cfg.rectflow.conditioning_checkpoint
    else:
        # Auto-find latest checkpoint from previous runs
        print("Auto-searching for latest conditioning checkpoint...")
        latest_checkpoint = find_latest_checkpoint("runs", pattern="model_*.pth")
        if latest_checkpoint is None:
            # Fallback to old models directory for backwards compatibility
            latest_checkpoint = find_latest_checkpoint("models", pattern="model_*.pth")

        if latest_checkpoint is None:
            raise FileNotFoundError(
                "Could not find conditioning checkpoint. Please specify --conditioning-checkpoint or "
                "ensure rectflow.conditioning_checkpoint in config points to a valid file."
            )
        conditioning_checkpoint = str(latest_checkpoint)
        print(f"Using latest checkpoint: {conditioning_checkpoint}")

    conditioning_model = load_model(conditioning_checkpoint)
    conditioning_model.requires_grad_(False)

    # Create velocity field based on config
    if cfg.rectflow.velocity_field_type == "DiT":
        velocity_model = VelocityField1dDiT(
            num_blocks=cfg.rectflow.num_blocks,
            conditioning_model=conditioning_model,
            num_heads=cfg.rectflow.num_heads,
            mlp_ratio=cfg.rectflow.mlp_ratio,
        ).to(args.device)
    elif cfg.rectflow.velocity_field_type == "CrossAttention":
        velocity_model = VelocityField1dCrossAttention(
            num_blocks=cfg.rectflow.num_blocks,
            conditioning_model=conditioning_model,
            num_heads=cfg.rectflow.num_heads,
            mlp_ratio=cfg.rectflow.mlp_ratio,
        ).to(args.device)
    else:
        raise ValueError(
            f"Unknown velocity_field_type: {cfg.rectflow.velocity_field_type}. Must be 'DiT' or 'CrossAttention'"
        )

    rectified_flow = RectifiedFlow(
        data_shape=(conditioning_model.config.hidden_size,),
        train_time_weight=cfg.rectflow_training.train_time_weight,
        train_time_distribution=cfg.rectflow_training.train_time_distribution,
        velocity_field=velocity_model,
        device=args.device,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        (param for param in velocity_model.parameters() if param.requires_grad), lr=cfg.rectflow_training.learning_rate
    )
    total_steps = _estimate_total_steps(train_loader, cfg.rectflow_training.num_epochs)
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=_build_warmup_cosine_lambda(
            warmup_steps=cfg.rectflow_training.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=cfg.rectflow_training.min_lr_ratio,
        ),
    )

    # Train model
    print(f"\nStarting training for {cfg.rectflow_training.num_epochs} epochs...")
    print("=" * 60)

    train(
        rectified_flow=rectified_flow,
        dataloader=train_loader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=cfg.rectflow_training.num_epochs,
        log_steps=cfg.rectflow_training.log_steps,
        save_steps=cfg.rectflow_training.save_steps,
        save_dir=models_dir,
        tensorboard_dir=tensorboard_dir,
        lr_scheduler=lr_scheduler,
        max_grad_norm=cfg.rectflow_training.max_grad_norm,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {models_dir}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"Config saved to: {run_dir / 'config.yaml'}")
    print(f"\nView training progress with: tensorboard --logdir {tensorboard_dir.parent}")
    if models_dir != checkpoints_dir:
        print(f"Note: Checkpoints saved to custom directory: {models_dir}")


if __name__ == "__main__":
    main()

"""Training and evaluation functions for citation embedding model."""

from pathlib import Path
from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel
from tqdm import tqdm


def train_epoch(
    model: BertModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    writer: Optional[SummaryWriter] = None,
    log_steps: int = 10,
    save_steps: int = 500,
    save_dir: Optional[Path] = None,
    start_step: int = 0,
) -> int:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on
        writer: TensorBoard writer for logging (optional)
        log_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        save_dir: Directory to save checkpoints (required if save_steps > 0)
        start_step: Starting step number (for resuming training)

    Returns:
        Final step number after this epoch
    """
    model.train()
    step = start_step

    if save_steps > 0 and save_dir is not None:
        save_dir.mkdir(exist_ok=True)

    for batch in dataloader:
        optimizer.zero_grad()

        # Move batch to device
        X = batch["inputs"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"]

        # Forward pass
        preds = model(inputs_embeds=X, attention_mask=mask)

        # Compute cosine similarity loss
        cossim = torch.cosine_similarity(preds["pooler_output"], y).mean()
        loss = -cossim

        # Backward pass
        loss.backward()
        optimizer.step()

        step += 1

        # Compute sequence length statistics
        lengths_float = lengths.float()
        avg_seq_length = lengths_float.mean().item()
        p5_seq_length = torch.quantile(lengths_float, 0.05).item()
        p95_seq_length = torch.quantile(lengths_float, 0.95).item()

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("CosineSimilarity/train", cossim.item(), step)
            writer.add_scalar("SequenceLength/avg", avg_seq_length, step)
            writer.add_scalar("SequenceLength/p5", p5_seq_length, step)
            writer.add_scalar("SequenceLength/p95", p95_seq_length, step)

        # Console logging
        if step % log_steps == 0:
            print(
                f"[{step}] Cossim: {cossim.item():.4f}, Loss: {loss.item():.4f}, "
                f"Avg Tokens: {avg_seq_length:.1f} (p5: {p5_seq_length:.1f}, p95: {p95_seq_length:.1f})"
            )

        # Save checkpoint
        if save_steps > 0 and step % save_steps == 0 and save_dir is not None:
            model_path = save_dir / f"model_{step}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[{step}] Saved model to {model_path}")

    return step


def train(
    model: BertModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    log_steps: int = 10,
    save_steps: int = 500,
    save_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
) -> BertModel:
    """
    Train model for multiple epochs.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on
        num_epochs: Number of epochs to train
        log_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        save_dir: Directory to save checkpoints
        tensorboard_dir: Directory for TensorBoard logs

    Returns:
        Trained model
    """
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            log_steps=log_steps,
            save_steps=save_steps,
            save_dir=save_dir,
            start_step=step,
        )

    if writer is not None:
        writer.close()

    return model


def evaluate(
    model: BertModel,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> dict:
    """
    Evaluate model on test/validation set.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation set
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_cosine_sims = []
    num_samples = 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating", unit="batch") if show_progress else dataloader

        for batch_idx, batch in enumerate(iterator):
            # Stop if we've reached max_batches
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move batch to device
            X = batch["inputs"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["targets"].to(device)

            # Get predictions
            preds = model(inputs_embeds=X, attention_mask=mask)
            pred_embeddings = preds["pooler_output"]

            # Compute cosine similarity per sample
            cosine_sims = torch.cosine_similarity(pred_embeddings, y, dim=1)
            all_cosine_sims.append(cosine_sims.cpu())

            num_samples += X.size(0)

            # Update progress bar
            if show_progress and hasattr(iterator, "set_postfix"):
                current_mean = torch.cat(all_cosine_sims).mean().item()
                iterator.set_postfix({"mean_cossim": f"{current_mean:.4f}", "samples": num_samples})

    # Concatenate all cosine similarities
    all_cosine_sims = torch.cat(all_cosine_sims).numpy()

    # Compute metrics
    metrics = {
        "num_samples": num_samples,
        "cosine_similarity": {
            "mean": float(np.mean(all_cosine_sims)),
            "median": float(np.median(all_cosine_sims)),
            "std": float(np.std(all_cosine_sims)),
            "min": float(np.min(all_cosine_sims)),
            "max": float(np.max(all_cosine_sims)),
            "p25": float(np.percentile(all_cosine_sims, 25)),
            "p75": float(np.percentile(all_cosine_sims, 75)),
            "p95": float(np.percentile(all_cosine_sims, 95)),
        },
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nNumber of samples: {metrics['num_samples']:,}")

    print("\nCosine Similarity:")
    cossim = metrics["cosine_similarity"]
    print(f"  Mean:   {cossim['mean']:.4f}")
    print(f"  Median: {cossim['median']:.4f}")
    print(f"  Std:    {cossim['std']:.4f}")
    print(f"  Min:    {cossim['min']:.4f}")
    print(f"  Max:    {cossim['max']:.4f}")
    print(f"  P25:    {cossim['p25']:.4f}")
    print(f"  P75:    {cossim['p75']:.4f}")
    print(f"  P95:    {cossim['p95']:.4f}")
    print("=" * 60 + "\n")


def save_metrics(metrics: dict, save_path: Path, model_path: Optional[Path] = None):
    """
    Save evaluation metrics to a text file.

    Args:
        metrics: Dictionary of metrics from evaluate()
        save_path: Path to save metrics file
        model_path: Optional path to model checkpoint (for reference in file)
    """
    with open(save_path, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        if model_path is not None:
            f.write(f"Model: {model_path}\n")

        f.write(f"Number of samples: {metrics['num_samples']:,}\n\n")
        f.write("Cosine Similarity:\n")

        cossim = metrics["cosine_similarity"]
        for key, value in cossim.items():
            f.write(f"  {key}: {value:.4f}\n")

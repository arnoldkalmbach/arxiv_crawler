"""Training and evaluation functions for rectified flow model."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rectified_flow import RectifiedFlow
from rectified_flow.samplers import CurvedEulerSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    rectified_flow: RectifiedFlow,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    log_steps: int = 10,
    save_steps: int = 500,
    save_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
    start_step: int = 0,
    lr_scheduler=None,
    max_grad_norm: float = 1.0,
) -> int:
    """
    Train model for a fixed number of steps.

    Args:
        rectified_flow: RectifiedFlow model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on
        max_steps: Total number of training steps
        log_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        save_dir: Directory to save checkpoints (required if save_steps > 0)
        tensorboard_dir: Directory for TensorBoard logs
        start_step: Starting step number (for resuming training)
        lr_scheduler: Optional learning rate scheduler stepped each iteration
        max_grad_norm: Maximum gradient norm for clipping (None/<=0 disables clipping)

    Returns:
        Trained model
    """
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    if save_dir is not None:
        save_dir.mkdir(exist_ok=True)

    rectified_flow.velocity_field.train()
    step = 0
    data_iter = iter(dataloader)

    while step < max_steps:
        # Get next batch, cycling through data if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()

        X0 = batch["X0"].to(device)
        X1 = batch["X1"].to(device)
        y = batch["inputs"].to(device)
        y_attention_mask = batch["attention_mask"].to(device)

        t = rectified_flow.sample_train_time(X1.shape[0], expand_dim=False)
        time_weights = rectified_flow.train_time_weight(t)

        Xt, dot_Xt_t = rectified_flow.get_interpolation(X0, X1, t)

        v_t = rectified_flow.get_velocity(Xt, t, y=y, attention_mask=y_attention_mask)

        loss = rectified_flow.criterion(
            v_t=v_t,
            dot_x_t=dot_Xt_t,
            x_t=Xt,
            t=t,
            time_weights=time_weights,
        )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            rectified_flow.velocity_field.parameters(),
            max_grad_norm if max_grad_norm is not None and max_grad_norm > 0 else float("inf"),
        )
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        step += 1

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), step)
            if grad_norm is not None:
                writer.add_scalar("GradNorm/total", float(grad_norm), step)
            if lr_scheduler is not None and optimizer.param_groups:
                writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], step)

        # Console logging
        if step % log_steps == 0:
            print(
                f"[{step}/{max_steps}] Loss: {loss.item():.4f}, LR: {current_lr:.2e}, GradNorm: {grad_norm.item():.4f}"
            )

        # Save checkpoint
        if save_steps > 0 and step % save_steps == 0 and save_dir is not None:
            model_path = save_dir / f"model_{step}.pth"
            torch.save(rectified_flow.velocity_field.state_dict(), model_path)
            print(f"[{step}] Saved model to {model_path}")

    return step


def evaluate(
    rectified_flow: RectifiedFlow,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
    num_examples: int = 10,
    top_k: int = 5,
) -> dict:
    """
    Evaluate model on test/validation set.

    Args:
        rectified_flow: RectifiedFlow model to evaluate
        dataloader: DataLoader for evaluation set
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        show_progress: Whether to show progress bar
        num_examples: Number of random examples to save with their metadata and cosine similarities
        top_k: Number of top matches to retrieve for each example (default: 5)

    Returns:
        Dictionary with evaluation metrics and examples
    """
    rectified_flow.velocity_field.eval()
    sde_sampler = CurvedEulerSampler(rectified_flow=rectified_flow)

    all_cosine_sims = []
    all_examples = []
    num_samples = 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating", unit="batch") if show_progress else dataloader

        for batch_idx, batch in enumerate(iterator):
            # Stop if we've reached max_batches
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move batch to device
            # For coupling dataset, X1 is the target, and we use X0 as the starting point
            X0 = batch["X0"].to(device)  # Noise vectors
            X1 = batch["X1"].to(device)  # Target vectors
            y = batch["inputs"].to(device)  # Conditioning embeddings
            y_attention_mask = batch["attention_mask"].to(device)

            sde_sampler.sample_loop(num_steps=50, x_0=X0, y=y, attention_mask=y_attention_mask)

            traj = sde_sampler.trajectories

            pred_embeddings = traj[-1]

            # Compute cosine similarity per sample (compare predictions to targets X1)
            cosine_sims = torch.cosine_similarity(pred_embeddings, X1, dim=1)
            all_cosine_sims.append(cosine_sims.cpu())

            # Collect examples with metadata if available
            metadata_list = batch.get("metadata", None)
            if metadata_list is not None:
                for i, meta in enumerate(metadata_list):
                    all_examples.append(
                        {
                            "cosine_similarity": cosine_sims[i].item(),
                            "metadata": meta,
                        }
                    )

            num_samples += y.size(0)

            # Update progress bar
            if show_progress and hasattr(iterator, "set_postfix"):
                current_mean = torch.cat(all_cosine_sims).mean().item()
                iterator.set_postfix({"mean_cossim": f"{current_mean:.4f}", "samples": num_samples})

    # Concatenate all cosine similarities
    all_cosine_sims = torch.cat(all_cosine_sims).numpy()

    # Sample random examples
    sampled_examples = []
    if all_examples and num_examples > 0:
        num_to_sample = min(num_examples, len(all_examples))
        sampled_indices = random.sample(range(len(all_examples)), num_to_sample)
        sampled_examples = [all_examples[i] for i in sampled_indices]
        # Sort by cosine similarity for easier reading
        sampled_examples.sort(key=lambda x: x["cosine_similarity"], reverse=True)

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
        "examples": sampled_examples,
    }

    return metrics

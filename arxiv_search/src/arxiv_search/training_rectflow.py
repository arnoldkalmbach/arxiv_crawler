"""Training and evaluation functions for citation embedding model."""

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rectified_flow import RectifiedFlow


def train_epoch(
    rectified_flow: RectifiedFlow,
    dataloader: DataLoader,  # wraps IterableCouplingDataset
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
    rectified_flow.velocity_field.train()
    step = start_step

    if save_steps > 0 and save_dir is not None:
        save_dir.mkdir(exist_ok=True)

    for batch in dataloader:
        optimizer.zero_grad()

        X0 = batch["X0"].to(device)
        X1 = batch["X1"].to(device)
        y = batch["inputs"].to(device)
        y_attention_mask = batch["attention_mask"].to(device)

        t = rectified_flow.sample_train_time(X1.shape[0], expand_dim=False)
        time_weights = rectified_flow.train_time_weight(t)

        Xt, dot_Xt_t = rectified_flow.get_interpolation(X0, X1, t)

        # Expand Xt along the conditioning sequence length dimension
        # TODO: Not sure if we can get away with this hack, or if we need to sample Xt for each time step.
        Xt = Xt.unsqueeze(1).expand(-1, y.size(1), -1)
        v_t = rectified_flow.get_velocity(Xt, t, y=y, attention_mask=y_attention_mask)

        loss = rectified_flow.criterion(
            v_t=v_t,
            dot_x_t=dot_Xt_t,
            x_t=Xt,
            t=t,
            time_weights=time_weights,
        )

        loss.backward()
        optimizer.step()
        step += 1

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), step)

        # Console logging
        if step % log_steps == 0:
            print(f"[{step}] Loss: {loss.item():.4f}")

        # Save checkpoint
        if save_steps > 0 and step % save_steps == 0 and save_dir is not None:
            model_path = save_dir / f"model_{step}.pth"
            torch.save(rectified_flow.velocity_field.state_dict(), model_path)
            print(f"[{step}] Saved model to {model_path}")

    return step


def train(
    rectified_flow: RectifiedFlow,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    log_steps: int = 10,
    save_steps: int = 500,
    save_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
) -> RectifiedFlow:
    """
    Train model for multiple epochs.

    Args:
        rectified_flow: RectifiedFlow model to train
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
            rectified_flow=rectified_flow,
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

    return rectified_flow

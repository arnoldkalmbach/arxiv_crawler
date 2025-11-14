"""Training and evaluation functions for citation embedding model."""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertModel


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
    inference,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
    num_examples: int = 10,
    top_k: int = 5,
) -> dict:
    """
    Evaluate model on test/validation set.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation set
        device: Device to run evaluation on
        inference: Inference instance for KNN retrieval
        max_batches: Maximum number of batches to evaluate (None = all)
        show_progress: Whether to show progress bar
        num_examples: Number of random examples to save with their metadata and cosine similarities
        top_k: Number of top matches to retrieve for each example (default: 5)

    Returns:
        Dictionary with evaluation metrics and examples
    """
    model.eval()

    all_cosine_sims = []
    all_examples = []  # Store examples with metadata and cosine similarities
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
            metadata = batch.get("metadata", None)

            # Get predictions
            preds = model(inputs_embeds=X, attention_mask=mask)
            pred_embeddings = preds["pooler_output"]

            # Compute cosine similarity per sample
            cosine_sims = torch.cosine_similarity(pred_embeddings, y, dim=1)
            all_cosine_sims.append(cosine_sims.cpu())

            # Collect examples with metadata if available
            if metadata is not None:
                for i in range(len(metadata)):
                    all_examples.append(
                        {
                            "cosine_similarity": cosine_sims[i].item(),
                            "metadata": metadata[i],
                        }
                    )

            num_samples += X.size(0)

            # Update progress bar
            if show_progress and hasattr(iterator, "set_postfix"):
                current_mean = torch.cat(all_cosine_sims).mean().item()
                iterator.set_postfix({"mean_cossim": f"{current_mean:.4f}", "samples": num_samples})

    # Concatenate all cosine similarities
    all_cosine_sims = torch.cat(all_cosine_sims).numpy()

    # Sample random examples
    sampled_examples = []
    if all_examples and num_examples > 0:
        # Sample without replacement if we have enough examples
        num_to_sample = min(num_examples, len(all_examples))
        import random

        sampled_indices = random.sample(range(len(all_examples)), num_to_sample)
        sampled_examples = [all_examples[i] for i in sampled_indices]
        # Sort by cosine similarity for easier reading
        sampled_examples.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # Add KNN retrieval results for each example
        for example in sampled_examples:
            meta = example["metadata"]
            # Use the cited paper as the general context and citation context as task context
            general_context = meta.get("cited_title", "")
            task_context = meta.get("reference_context", "")

            if general_context:
                # Perform KNN search
                search_contexts = [(general_context, task_context)]
                matches_df = inference.get_matches(search_contexts, top_k=top_k)
                matches_with_meta = (
                    pl.scan_ndjson(dataloader.dataset.papers_file)
                    .join(matches_df.lazy(), on="arxiv_id", how="inner")
                    .collect()
                )

                # Filter to the first query's results and convert to list of dicts
                query_matches = matches_with_meta.filter(matches_with_meta["query_index"] == 0)
                knn_results = []
                for row in query_matches.iter_rows(named=True):
                    knn_results.append(
                        {
                            "arxiv_id": row.get("arxiv_id", ""),
                            "title": row.get("title", ""),
                            "distance": row.get("distance", 0.0),
                        }
                    )
                example["knn_matches"] = knn_results

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


def save_examples(examples: list[dict], save_path: Path, model_path: Optional[Path] = None):
    """
    Save evaluation examples with their metadata and cosine similarities to a text file.

    Args:
        examples: List of example dictionaries from evaluate()
        save_path: Path to save examples file
        model_path: Optional path to model checkpoint (for reference in file)
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("EVALUATION EXAMPLES\n")
        f.write("=" * 80 + "\n\n")

        if model_path is not None:
            f.write(f"Model: {model_path}\n\n")

        f.write(f"Number of examples: {len(examples)}\n")
        f.write("(Sorted by cosine similarity, highest to lowest)\n\n")
        f.write("=" * 80 + "\n\n")

        for i, example in enumerate(examples, 1):
            cosine_sim = example["cosine_similarity"]
            meta = example["metadata"]

            f.write(f"Example {i}:\n")
            f.write(f"  Cosine Similarity: {cosine_sim:.4f}\n\n")

            f.write(f"  Citing Paper (arxiv_id: {meta['citer_arxiv_id']}):\n")
            citer_title = meta.get("citer_title", "")
            if citer_title:
                f.write(f"    Title: {citer_title}\n")
            else:
                f.write("    Title: [Not available]\n")
            f.write("\n")

            f.write("  Citation Context:\n")
            context = meta.get("reference_context", "")
            # Wrap long lines for better readability
            if context:
                # Simple wrapping - split by sentences or just truncate if too long
                context_lines = context.replace(". ", ".\n    ")
                f.write(f"    {context_lines}\n")
            else:
                f.write("    [Not available]\n")
            f.write("\n")

            f.write(f"  Cited Paper (arxiv_id: {meta['cited_arxiv_id']}):\n")
            cited_title = meta.get("cited_title", "")
            if cited_title:
                f.write(f"    Title: {cited_title}\n")
            else:
                f.write("    Title: [Not available]\n")

            # Add KNN retrieval results if available
            knn_matches = example.get("knn_matches", [])
            if knn_matches:
                f.write("\n")
                f.write(f"  KNN Search Results (Top {len(knn_matches)} matches):\n")
                for j, match in enumerate(knn_matches, 1):
                    match_title = match.get("title", "")
                    match_arxiv_id = match.get("arxiv_id", "")
                    match_distance = match.get("distance", 0.0)

                    # Check if this is the ground truth cited paper
                    is_ground_truth = match_arxiv_id == meta.get("cited_arxiv_id", "")
                    gt_marker = " ★ [GROUND TRUTH]" if is_ground_truth else ""

                    f.write(f"    {j}. Distance: {match_distance:.4f}{gt_marker}\n")
                    f.write(f"       arxiv_id: {match_arxiv_id}\n")
                    if match_title:
                        f.write(f"       Title: {match_title}\n")
                    f.write("\n")

            f.write("-" * 80 + "\n\n")

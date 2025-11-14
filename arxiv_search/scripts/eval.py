"""Evaluation script for citation embedding model."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from arxiv_search.config import Config, load_config
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
    get_collate_fn,
)
from arxiv_search.model import load_model
from arxiv_search.training import evaluate, print_metrics, save_metrics, save_examples
from arxiv_search.inference import Inference


# TODO: Implement retrieval metrics to see how we're doing on negatives, not just cossim of positives

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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top KNN matches to retrieve for each example",
    )
    # Use parse_known_args to separate normal args from config overrides
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    """Main evaluation function."""
    # Parse required arguments and collect config overrides
    args, unknown = parse_args()

    # Load configuration (merges default.yaml with CLI config overrides)
    cfg = load_config(cli_overrides=unknown)

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
    papers_file = data_dir / "papers.jsonl"
    test_ds = CitationEmbeddingDataset(
        citations_file=test_citations,
        paper_embeddings_file=data_dir / "paper_embeddings.parquet",
        citation_embeddings_dir=data_dir / "test",
        citations_batch_size=cfg.data.citations_batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        shuffle_shards=False,
        papers_file=papers_file if papers_file.exists() else None,
        return_metadata=True,  # Enable metadata for saving examples
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

    task_model = load_model(
        str(model_path),
        device=args.device,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
    )
    print("Model loaded successfully.")

    # Load general model for inference (must match the model used to build embeddings)
    print(f"\nLoading general model: {cfg.data.basemodel_name}...")
    general_model = SentenceTransformer(cfg.data.basemodel_name, device=args.device)
    print("General model loaded successfully.")
    
    # Build KNN index
    print("\nBuilding KNN index...")
    paper_embeddings_file = data_dir / "paper_embeddings.parquet"
    if not paper_embeddings_file.exists():
        raise FileNotFoundError(f"Paper embeddings not found at {paper_embeddings_file}")
    
    inference = Inference(
        general_model=general_model,
        task_model=task_model,
        max_length=cfg.data.max_length,
        pad_to_multiple_of=cfg.data.pad_to_multiple_of,
        device=args.device,
    )
    inference.build_index(paper_embeddings_file)
    print(f"KNN index built with {len(inference.paper_embeddings)} papers.")

    # Run evaluation
    print("\nStarting evaluation...\n")
    metrics = evaluate(
        model=task_model,
        dataloader=test_loader,
        device=args.device,
        inference=inference,
        max_batches=cfg.evaluation.max_batches,
        show_progress=True,
        num_examples=20,  # Save 20 random examples
        top_k=args.top_k,
    )

    # Print results
    print_metrics(metrics)

    # Save metrics to file (next to model checkpoint)
    metrics_path = model_path.parent / f"{model_path.stem}_eval_metrics.txt"
    save_metrics(metrics, metrics_path, model_path)
    print(f"Metrics saved to {metrics_path}")
    
    # Save examples to file
    if metrics.get("examples"):
        examples_path = model_path.parent / f"{model_path.stem}_eval_examples.txt"
        save_examples(metrics["examples"], examples_path, model_path)
        print(f"Examples saved to {examples_path} ({len(metrics['examples'])} examples)")


if __name__ == "__main__":
    main()

"""Evaluation script for citation embedding model."""

import argparse
from pathlib import Path

import torch
import torch.distributions as dist
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from arxiv_search.config import find_latest_checkpoint, load_config
from arxiv_search.dataloader import (
    CitationEmbeddingDataset,
    ensure_dataset_exists,
    get_collate_fn,
)
from arxiv_search.inference import DirectVectorInference, RectFlowVectorInference
from arxiv_search.iterable_coupling_dataset import IterableCouplingDataset, get_coupling_collate_fn
from arxiv_search.model import load_model, load_rectflow_model
from arxiv_search.search import ContextualSearch
from arxiv_search.training import evaluate, print_metrics, save_examples, save_metrics
from arxiv_search.training_rectflow import evaluate as evaluate_rectflow

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
        default=None,
        help="Path to trained model checkpoint (.pth file). If not specified, auto-finds latest checkpoint of the specified type.",
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
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["direct", "rectflow"],
        required=True,
        help="Type of model: 'direct' (BERT) or 'rectflow' (RectifiedFlow)",
    )
    parser.add_argument(
        "--conditioning-checkpoint",
        type=str,
        default=None,
        help="Path to conditioning model checkpoint for rectflow models (default: auto-finds latest in runs/)",
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
    model_type = args.model_type

    # Auto-find model path if not specified
    if args.model_path is None:
        print("Auto-searching for latest checkpoint...")
        if model_type == "rectflow":
            base_dir = cfg.rectflow_training.tensorboard_dir
            latest_checkpoint = find_latest_checkpoint(base_dir, pattern="model_*.pth")
            if latest_checkpoint is None:
                # Fallback to old directory
                latest_checkpoint = find_latest_checkpoint("rectflow_models", pattern="model_*.pth")
        else:  # direct
            base_dir = cfg.training.tensorboard_dir
            latest_checkpoint = find_latest_checkpoint(base_dir, pattern="model_*.pth")
            if latest_checkpoint is None:
                # Fallback to old directory
                latest_checkpoint = find_latest_checkpoint("models", pattern="model_*.pth")

        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"Could not find {model_type} model checkpoint. Please specify --model-path or "
                f"ensure checkpoints exist in {base_dir} or the old models directory."
            )
        model_path = latest_checkpoint
        print(f"Using latest checkpoint: {model_path}")
    else:
        model_path = Path(args.model_path)

    # Print configuration
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Model type: {model_type}")
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

    # Create test dataset based on model type
    print("Loading test dataset...")
    papers_file = data_dir / "papers.jsonl"

    if model_type == "rectflow":
        # For rectflow, wrap in IterableCouplingDataset
        citation_ds = CitationEmbeddingDataset(
            citations_file=test_citations,
            paper_embeddings_file=data_dir / "paper_embeddings.parquet",
            citation_embeddings_dir=data_dir / "test",
            citations_batch_size=cfg.data.citations_batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            shuffle_shards=False,
            papers_file=papers_file if papers_file.exists() else None,
            return_metadata=True,  # Enable metadata for saving examples
        )
        test_ds = IterableCouplingDataset(
            D1=citation_ds,
            D0=dist.Normal(loc=0.0, scale=1.0),
            extract_target=lambda x: x[1],
            extract_key=lambda x: x[2]["reference_id"],
            extract_conditioning=lambda x: x[0],
            extract_metadata=lambda x: x[2],
        )
        collate_fn = get_coupling_collate_fn(
            max_length=cfg.data.max_length,
            pad_to_multiple_of=cfg.data.pad_to_multiple_of,
            pad_value=cfg.data.pad_value,
            mask_dtype=torch.bool,
        )
    else:
        # For direct models, use CitationEmbeddingDataset directly
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
        collate_fn = get_collate_fn(
            max_length=cfg.data.max_length,
            pad_to_multiple_of=cfg.data.pad_to_multiple_of,
            pad_value=cfg.data.pad_value,
            mask_dtype=torch.bool,
        )

    # Create dataloader
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

    # Load model based on type
    if model_type == "rectflow":
        # Load rectflow model
        print(f"\nLoading rectflow model from {model_path}...")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

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
            print(f"Using latest conditioning checkpoint: {conditioning_checkpoint}")

        # Load rectflow model using the centralized function
        rectified_flow = load_rectflow_model(
            velocity_field_checkpoint=str(model_path),
            rectflow_config=cfg.rectflow,
            model_config=cfg.model,
            max_length=cfg.data.max_length,
            device=args.device,
            conditioning_checkpoint=conditioning_checkpoint,
        )
        print("RectifiedFlow model loaded successfully.")

        # Load general model for inference (must match the model used to build embeddings)
        print(f"\nLoading general model: {cfg.data.basemodel_name}...")
        general_model = SentenceTransformer(cfg.data.basemodel_name, device=args.device)
        print("General model loaded successfully.")

        # Build KNN index for retrieval examples
        print("\nBuilding KNN index...")
        paper_embeddings_file = data_dir / "paper_embeddings.parquet"
        if not paper_embeddings_file.exists():
            raise FileNotFoundError(f"Paper embeddings not found at {paper_embeddings_file}")

        vector_inference = RectFlowVectorInference(rectified_flow, num_steps=50)
        search = ContextualSearch(
            general_model=general_model,
            vector_inference=vector_inference,
            max_length=cfg.data.max_length,
            pad_to_multiple_of=cfg.data.pad_to_multiple_of,
            device=args.device,
        )
        search.build_index(paper_embeddings_file)
        print(f"KNN index built with {len(search.paper_embeddings)} papers.")

        # Run evaluation
        print("\nStarting evaluation...\n")
        metrics = evaluate_rectflow(
            rectified_flow=rectified_flow,
            dataloader=test_loader,
            device=args.device,
            search=search,
            max_batches=cfg.evaluation.max_batches,
            show_progress=True,
            num_examples=20,  # Save 20 random examples
            top_k=args.top_k,
        )

    else:  # model_type == "direct"
        # Load direct model
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

        vector_inference = DirectVectorInference(task_model)
        search = ContextualSearch(
            general_model=general_model,
            vector_inference=vector_inference,
            max_length=cfg.data.max_length,
            pad_to_multiple_of=cfg.data.pad_to_multiple_of,
            device=args.device,
        )
        search.build_index(paper_embeddings_file)
        print(f"KNN index built with {len(search.paper_embeddings)} papers.")

        # Run evaluation
        print("\nStarting evaluation...\n")
        metrics = evaluate(
            model=task_model,
            dataloader=test_loader,
            device=args.device,
            inference=search,
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

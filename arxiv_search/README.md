# arxiv-search

Citation embedding model training and a web interface for exploring arXiv papers.

## Overview

This package provides:

- **Citation embedding models** — Train models to generate embeddings for paper citations
- **Evaluation framework** — Evaluate embedding quality and retrieval performance
- **Web browser UI** — Browse papers, view citations, and search the corpus

## Installation

This package is part of a uv workspace. From the repository root:

```bash
uv sync
```

This will install both `arxiv-search` and its dependency `arxiv-crawler`.

## Usage

### Training a Model

```bash
# Build the embeddings dataset from crawled papers
uv run --package arxiv-search python scripts/build_embeddings_dataset.py

# Train the citation embedding model
uv run --package arxiv-search python scripts/train.py

# Evaluate the trained model
uv run --package arxiv-search python scripts/eval.py
```

### Running the Web Browser

```bash
uv run uvicorn browser.app:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/build_embeddings_dataset.py` | Generate embedding dataset from paper data |
| `scripts/train.py` | Train the citation embedding model |
| `scripts/eval.py` | Evaluate model performance |
| `scripts/upload_to_huggingface.py` | Upload dataset/model to Hugging Face Hub |

## API

The package exports the following:

```python
from arxiv_search import (
    # Configuration
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    
    # Data loading
    CitationEmbeddingDataset,
    collate_embeddings_with_targets,
    ensure_dataset_exists,
    get_collate_fn,
    
    # Model
    create_model,
    load_model,
    
    # Training and evaluation
    train,
    train_epoch,
    evaluate,
    print_metrics,
    save_metrics,
)
```

## Configuration

Model and training configuration is stored in `configs/default.yaml`. Override settings via command-line arguments or environment variables.

## Data

- `data/papers.jsonl` — Paper metadata (copy or symlink from arxiv_crawler)
- `data/paper_embeddings.parquet` — Pre-computed paper embeddings
- `data/train/` — Training split citation embeddings
- `data/test/` — Test split citation embeddings

## Web Browser

The browser application (`browser/app.py`) provides:

- Paper browsing and search
- Citation network visualization
- Fulltext view with structured sections
- Crawler status monitoring

Built with FastAPI and Jinja2 templates.

## Development

### Linting

```bash
uv run --package arxiv-search ruff check --fix .
uv run --package arxiv-search ruff format .
```

## Dependencies

This package depends on `arxiv-crawler` (resolved from the workspace) plus:

- PyTorch & torchvision
- sentence-transformers
- diffusers
- FastAPI & uvicorn
- Hugging Face Hub


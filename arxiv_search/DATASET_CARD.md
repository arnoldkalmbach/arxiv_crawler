---
license: mit
task_categories:
- text-retrieval
- sentence-similarity
tags:
- arxiv
- citations
- embeddings
---

# arXiv Citation Embeddings Dataset

This dataset contains citation embeddings for arXiv papers, designed for training models that predict cited paper embeddings from citation context.

## Dataset Structure

### Core Files (Required for Training)
- `paper_embeddings.parquet`: Paper-level sentence embeddings (shared across splits)
- `train/citations.jsonl`: Training citation pairs with contexts
- `train/citation_embeddings_*.parquet`: Citation context embeddings (sharded)
- `test/citations.jsonl`: Test citation pairs with contexts
- `test/citation_embeddings_*.parquet`: Test citation context embeddings (sharded)
- `papers.jsonl`: Original paper metadata (title, abstract, citations)

{OPTIONAL_XML_SECTION}

## Usage

### Download Dataset (Excluding XML)
```python
from huggingface_hub import snapshot_download

# Download only essential training files (recommended)
snapshot_download(
    repo_id="{DATASET_NAME}",
    repo_type="dataset",
    allow_patterns=[
        "paper_embeddings.parquet",
        "papers.jsonl",
        "train/*.jsonl",
        "train/*.parquet",
        "test/*.jsonl",
        "test/*.parquet",
    ]
)
```

### Use with Custom PyTorch Dataset
```python
from arxiv_search.dataloader import CitationEmbeddingDataset
from pathlib import Path

# For training
train_ds = CitationEmbeddingDataset(
    citations_file=Path("train/citations.jsonl"),
    paper_embeddings_file=Path("paper_embeddings.parquet"),
    citation_embeddings_dir=Path("train"),
    citations_batch_size=10000,
    shuffle=True,
)

# For testing/evaluation
test_ds = CitationEmbeddingDataset(
    citations_file=Path("test/citations.jsonl"),
    paper_embeddings_file=Path("paper_embeddings.parquet"),
    citation_embeddings_dir=Path("test"),
    citations_batch_size=10000,
    shuffle=False,
    shuffle_shards=False,
)
```

## Data Splits

Citations are split by **citing paper** (citer_arxiv_id) to test generalization to new sources/authors. This is the standard approach for evaluating supervised learning models.

## Embedding Model

Paper and citation embeddings are generated using [allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter), a sentence transformer model trained on scientific documents.

## Citation Format

Papers use the SPECTER format: `"{title}[SEP]{abstract}"` for generating embeddings.

## Files

### paper_embeddings.parquet
- **Format**: Parquet
- **Compression**: Snappy (default)
- **Schema**:
  - `arxiv_id`: string (paper identifier)
  - `sentence_embedding`: array of float32 with shape [768]

### citations.jsonl (per split)
- **Format**: JSONL (newline-delimited JSON)
- **Schema**:
  ```json
  {
    "index": "integer (row index, starts from 0 for each split)",
    "citer_arxiv_id": "string (paper making the citation)",
    "cited_arxiv_id": "string (paper being cited)",
    "reference_contexts": "string (text context around the citation)",
    "reference_id": "string (base64-encoded hash of the reference context)"
  }
  ```

### citation_embeddings_*.parquet (per split)
- **Format**: Parquet
- **Compression**: zstd
- **Naming**: Files are named with the batch start index (e.g., `citation_embeddings_0.parquet`, `citation_embeddings_5000.parquet`)
- **Schema**:
  - `input_ids`: list of int32 (tokenized input)
  - `token_type_ids`: list of int32 (segment IDs)
  - `attention_mask`: list of int32 (attention mask for tokens)
  - `token_embeddings`: list of arrays, each array shape [768] (embeddings per token)
  - `sentence_embedding`: array of float32 with shape [768] (pooled sentence embedding)
  - `reference_id`: string (base64-encoded hash matching citations.jsonl)

## Source Code

The dataset generation pipeline and model training code is available at: [arxiv_search](https://github.com/yourusername/arxiv_crawler)

## License

MIT License


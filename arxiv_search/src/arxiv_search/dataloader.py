import polars as pl
import torch
import numpy as np
import random
from functools import partial
from typing import Any, Optional
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate
from huggingface_hub import snapshot_download


def _to_tensor(x):
    """Convert input to tensor if not already a tensor."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def collate_embeddings_with_targets(
    batch: list[tuple[torch.Tensor, Any]],
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.bool,
):
    """
    Collate function for batching embeddings with optional targets.

    Designed for use with CitationEmbeddingDataset.

    Args:
        batch: list of tuples where each tuple can be:
            - (X,) - just embeddings (inference)
            - (X, y) - embeddings with targets (training)
            - (X, y, metadata) - embeddings with targets and metadata
            where:
                X: [seq_len, embed_dim] (float tensor or array)
                y: target vector (same shape across samples; tensor/array/number/dict ok) or None
                metadata: optional dict with human-readable information
        max_length: Maximum sequence length (truncate if longer)
        pad_to_multiple_of: Pad sequences to multiple of this value
        pad_value: Value to use for padding
        mask_dtype: Data type for attention mask

    Returns:
        Dictionary containing:
            - inputs: FloatTensor [B, L, K] - padded input embeddings
            - attention_mask: Bool/Int Tensor [B, L] (True/1 = real token)
            - lengths: LongTensor [B] - pre-padding lengths
            - targets: stacked y (shape depends on your dataset; typically [B, T]) - only if targets present
            - metadata: list of metadata dicts (if present in batch)
    """
    # Determine batch structure from first item
    batch_len = len(batch[0])
    has_targets = batch_len >= 2
    has_metadata = batch_len == 3

    # Convert to tensors and apply truncation
    Xs, Ys, Metas = [], [], []
    for item in batch:
        if has_metadata:
            X, y, meta = item
            Metas.append(meta)
        elif has_targets:
            X, y = item
        else:
            X = item[0]
            y = None

        Xt = _to_tensor(X)  # [Li, K]
        if max_length is not None:
            Xt = Xt[:max_length]
        Xs.append(Xt)
        if has_targets:
            Ys.append(y)

    # Compute lengths and target pad length
    lengths = torch.tensor([x.size(0) for x in Xs], dtype=torch.long)

    # Optionally pad L up to a multiple (helps on some accelerators)
    if pad_to_multiple_of is not None:
        max_len = int(lengths.max().item())
        if max_len % pad_to_multiple_of != 0:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        # Truncate already done; pad_sequence will handle padding up to the longest in list,
        # so we extend each shorter sequence with EMPTY rows to reach max_len.
        # pad_sequence itself can't force a larger-than-maximum length, so we manually
        # right-pad each X to max_len first.
        K = Xs[0].size(1)
        padded_list = []
        for x in Xs:
            if x.size(0) < max_len:
                pad_rows = max_len - x.size(0)
                pad_block = x.new_full((pad_rows, K), pad_value)
                padded_list.append(torch.cat([x, pad_block], dim=0))
            else:
                padded_list.append(x)
        inputs = torch.stack(padded_list, dim=0)  # [B, L, K]
    else:
        # Let pad_sequence expand to the within-batch max
        inputs = pad_sequence([_to_tensor(x) for x in Xs], batch_first=True, padding_value=pad_value)

    # Build attention mask from true lengths (before any manual right-padding)
    L = inputs.size(1)
    arange = torch.arange(L).unsqueeze(0)  # [1, L]
    attention_mask = arange < lengths.unsqueeze(1)
    attention_mask = attention_mask.to(mask_dtype)

    result = {
        "inputs": inputs,  # [B, L, K], float
        "attention_mask": attention_mask,  # [B, L], bool (or chosen dtype)
        "lengths": lengths,  # [B]
    }

    # Add targets if present
    if has_targets:
        # Stack targets robustly (supports tensors, numbers, tuples, dicts, etc.)
        targets = default_collate([_to_tensor(y) for y in Ys])
        result["targets"] = targets  # e.g., [B, T] or [B] depending on your dataset

    # Add metadata if present
    if has_metadata:
        result["metadata"] = Metas

    return result


def get_collate_fn(
    max_length: int = 256,
    pad_to_multiple_of: int = 8,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.bool,
):
    """
    Get a collate function configured for CitationEmbeddingDataset.

    Args:
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        pad_value: Padding value
        mask_dtype: Data type for attention mask

    Returns:
        Partial function configured for collating embeddings
    """
    return partial(
        collate_embeddings_with_targets,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_value=pad_value,
        mask_dtype=mask_dtype,
    )


def ensure_dataset_exists(data_dir: Path, hf_dataset_name: Optional[str] = None) -> Path:
    """
    Ensure the dataset exists locally. Download from HuggingFace if it doesn't.

    Args:
        data_dir: Local directory where dataset should be
        hf_dataset_name: HuggingFace dataset name (e.g., 'username/dataset-name').
                        If None, will not attempt to download.

    Returns:
        Path to the data directory

    Raises:
        FileNotFoundError: If dataset not found locally and no HuggingFace name provided
    """
    # Check if essential files exist
    paper_embeddings = data_dir / "paper_embeddings.parquet"
    train_citations = data_dir / "train" / "citations.jsonl"

    if paper_embeddings.exists() and train_citations.exists():
        print(f"Dataset found at {data_dir}")
        return data_dir

    if hf_dataset_name is None:
        raise FileNotFoundError(
            f"Dataset not found at {data_dir} and no HuggingFace dataset name provided. "
            "Either ensure the dataset exists locally or provide --hf-dataset-name."
        )

    print(f"Dataset not found locally. Downloading from HuggingFace: {hf_dataset_name}")

    # Download dataset, excluding XML files
    downloaded_path = snapshot_download(
        repo_id=hf_dataset_name,
        repo_type="dataset",
        allow_patterns=[
            "paper_embeddings.parquet",
            "papers.jsonl",
            "train/*.jsonl",
            "train/*.parquet",
            "test/*.jsonl",
            "test/*.parquet",
        ],
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,  # Copy files instead of symlinking
    )

    print(f"Dataset downloaded to {downloaded_path}")
    return Path(downloaded_path)


class CitationEmbeddingDataset(IterableDataset):
    def __init__(
        self,
        citations_file: Path,
        paper_embeddings_file: Path,
        citation_embeddings_dir: Path,
        citations_batch_size: int,
        shuffle: bool = True,
        shuffle_shards: bool = True,
        papers_file: Optional[Path] = None,
        return_metadata: bool = False,
    ):
        # Store file paths instead of DataFrames to avoid pickling issues with multiprocessing
        self.citations_file = citations_file
        self.paper_embeddings_file = paper_embeddings_file
        self.papers_file = papers_file

        self.citations_batch_size = citations_batch_size
        self.citation_embeddings_dir = Path(citation_embeddings_dir) if citation_embeddings_dir else Path(".")

        # Shuffling options
        self.shuffle = shuffle  # Shuffle within shards
        self.shuffle_shards = shuffle_shards  # Shuffle shard order

        # Return metadata (for evaluation/inspection)
        self.return_metadata = return_metadata

    def _load_data(self):
        """Load citations and paper embeddings data in each worker."""
        citations = pl.read_ndjson(self.citations_file)
        paper_embeddings = pl.read_parquet(self.paper_embeddings_file)
        papers = None
        if self.return_metadata and self.papers_file is not None and self.papers_file.exists():
            papers = pl.read_ndjson(self.papers_file)
        return citations, paper_embeddings, papers

    def _build_paper_embeddings_dict(self, paper_embeddings: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Build a lookup dictionary mapping arxiv_id to paper embeddings."""
        paper_embeddings_dict = {}
        for row in paper_embeddings.iter_rows(named=True):
            arxiv_id = row["arxiv_id"]
            sentence_embedding = np.array(row["sentence_embedding"], dtype=np.float32)
            paper_embeddings_dict[arxiv_id] = torch.from_numpy(sentence_embedding)
        return paper_embeddings_dict

    def _build_paper_metadata_dict(self, papers: Optional[pl.DataFrame]) -> dict[str, dict]:
        """Build a lookup dictionary mapping arxiv_id to paper metadata (title, abstract)."""
        if papers is None:
            return {}

        paper_metadata_dict = {}
        for row in papers.iter_rows(named=True):
            arxiv_id = row["arxiv_id"]
            paper_metadata_dict[arxiv_id] = {
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }
        return paper_metadata_dict

    def _get_worker_shards(self, citations: pl.DataFrame):
        """Determine which shards this worker should process."""
        worker_info = get_worker_info()

        # Group citations by shard
        citations_with_shard = citations.with_columns((pl.col("index") // self.citations_batch_size).alias("shard"))

        # Get unique shards
        unique_shards = sorted(citations_with_shard["shard"].unique().to_list())

        # Split shards among workers
        if worker_info is None:
            worker_shards = unique_shards
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_shards = unique_shards[worker_id::num_workers]

        # Shuffle shard order if requested (different order each epoch)
        if self.shuffle_shards:
            random.shuffle(worker_shards)

        return citations_with_shard, worker_shards

    def _load_shard_embeddings(self, shard_id: int) -> dict[str, torch.Tensor]:
        """Load citation embeddings for a specific shard."""
        shard_file = (
            self.citation_embeddings_dir / f"citation_embeddings_{shard_id * self.citations_batch_size}.parquet"
        )

        shard_data = pl.read_parquet(shard_file)

        citation_embeddings_dict = {}
        for row in shard_data.iter_rows(named=True):
            reference_id = row["reference_id"]
            token_embeddings = np.array(row["token_embeddings"], dtype=np.float32)
            citation_embeddings_dict[reference_id] = torch.from_numpy(token_embeddings)

        return citation_embeddings_dict

    def __iter__(self):
        # Worker setup
        citations, paper_embeddings, papers = self._load_data()
        paper_embeddings_dict = self._build_paper_embeddings_dict(paper_embeddings)
        paper_metadata_dict = self._build_paper_metadata_dict(papers) if self.return_metadata else {}
        citations_with_shard, worker_shards = self._get_worker_shards(citations)

        # Process each shard assigned to this worker
        for shard_id in worker_shards:
            citation_embeddings_dict = self._load_shard_embeddings(shard_id)
            shard_citations = citations_with_shard.filter(pl.col("shard") == shard_id)

            # Get citation data as lists
            reference_ids = shard_citations["reference_id"].to_list()
            citer_arxiv_ids = shard_citations["citer_arxiv_id"].to_list()
            cited_arxiv_ids = shard_citations["cited_arxiv_id"].to_list()
            reference_contexts = (
                shard_citations["reference_contexts"].to_list() if self.return_metadata else [None] * len(reference_ids)
            )

            # Shuffle within shard if requested
            if self.shuffle:
                indices = list(range(len(reference_ids)))
                random.shuffle(indices)
                reference_ids = [reference_ids[i] for i in indices]
                citer_arxiv_ids = [citer_arxiv_ids[i] for i in indices]
                cited_arxiv_ids = [cited_arxiv_ids[i] for i in indices]
                reference_contexts = [reference_contexts[i] for i in indices]

            # Yield training samples from this shard
            for reference_id, citer_arxiv_id, cited_arxiv_id, reference_context in zip(
                reference_ids, citer_arxiv_ids, cited_arxiv_ids, reference_contexts
            ):
                # Get embeddings
                e_citation = citation_embeddings_dict.get(reference_id)
                e_citer = paper_embeddings_dict.get(citer_arxiv_id)
                e_cited = paper_embeddings_dict.get(cited_arxiv_id)

                # Skip if any embeddings are missing
                if e_citation is None or e_citer is None or e_cited is None:
                    continue

                # Create input and output tensors
                inputs_embeds = torch.vstack([e_citer, e_citation])
                output_embeds = e_cited

                if self.return_metadata:
                    # Return embeddings and metadata
                    metadata = {
                        "reference_id": reference_id,
                        "citer_arxiv_id": citer_arxiv_id,
                        "cited_arxiv_id": cited_arxiv_id,
                        "reference_context": reference_context,
                        "citer_title": paper_metadata_dict.get(citer_arxiv_id, {}).get("title", ""),
                        "cited_title": paper_metadata_dict.get(cited_arxiv_id, {}).get("title", ""),
                    }
                    yield inputs_embeds, output_embeds, metadata
                else:
                    yield inputs_embeds, output_embeds

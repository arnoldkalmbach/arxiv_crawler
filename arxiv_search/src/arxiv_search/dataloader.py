import polars as pl
import torch
import numpy as np
import random
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path


class CitationEmbeddingDataset(IterableDataset):
    def __init__(
        self,
        citations_file: Path,
        paper_embeddings_file: Path,
        citation_embeddings_dir: Path,
        citations_batch_size: int,
        shuffle: bool = True,
        shuffle_shards: bool = True,
    ):
        # Store file paths instead of DataFrames to avoid pickling issues with multiprocessing
        self.citations_file = citations_file
        self.paper_embeddings_file = paper_embeddings_file

        self.citations_batch_size = citations_batch_size
        self.citation_embeddings_dir = Path(citation_embeddings_dir) if citation_embeddings_dir else Path(".")

        # Shuffling options
        self.shuffle = shuffle  # Shuffle within shards
        self.shuffle_shards = shuffle_shards  # Shuffle shard order

    def _load_data(self):
        """Load citations and paper embeddings data in each worker."""
        citations = pl.read_ndjson(self.citations_file)
        paper_embeddings = pl.read_parquet(self.paper_embeddings_file)
        return citations, paper_embeddings

    def _build_paper_embeddings_dict(self, paper_embeddings: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Build a lookup dictionary mapping arxiv_id to paper embeddings."""
        paper_embeddings_dict = {}
        for row in paper_embeddings.iter_rows(named=True):
            arxiv_id = row["arxiv_id"]
            sentence_embedding = np.array(row["sentence_embedding"], dtype=np.float32)
            paper_embeddings_dict[arxiv_id] = torch.from_numpy(sentence_embedding)
        return paper_embeddings_dict

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
        shard_file = self.citation_embeddings_dir / f"citation_embeddings_{shard_id * self.citations_batch_size}.parquet"

        shard_data = pl.read_parquet(shard_file)

        citation_embeddings_dict = {}
        for row in shard_data.iter_rows(named=True):
            reference_id = row["reference_id"]
            token_embeddings = np.array(row["token_embeddings"], dtype=np.float32)
            citation_embeddings_dict[reference_id] = torch.from_numpy(token_embeddings)

        return citation_embeddings_dict

    def __iter__(self):
        # Worker setup
        citations, paper_embeddings = self._load_data()
        paper_embeddings_dict = self._build_paper_embeddings_dict(paper_embeddings)
        citations_with_shard, worker_shards = self._get_worker_shards(citations)

        # Process each shard assigned to this worker
        for shard_id in worker_shards:
            citation_embeddings_dict = self._load_shard_embeddings(shard_id)
            shard_citations = citations_with_shard.filter(pl.col("shard") == shard_id)

            # Get citation data as lists
            reference_ids = shard_citations["reference_id"].to_list()
            citer_arxiv_ids = shard_citations["citer_arxiv_id"].to_list()
            cited_arxiv_ids = shard_citations["cited_arxiv_id"].to_list()

            # Shuffle within shard if requested
            if self.shuffle:
                indices = list(range(len(reference_ids)))
                random.shuffle(indices)
                reference_ids = [reference_ids[i] for i in indices]
                citer_arxiv_ids = [citer_arxiv_ids[i] for i in indices]
                cited_arxiv_ids = [cited_arxiv_ids[i] for i in indices]

            # Yield training samples from this shard
            for reference_id, citer_arxiv_id, cited_arxiv_id in zip(reference_ids, citer_arxiv_ids, cited_arxiv_ids):
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

                yield inputs_embeds, output_embeds

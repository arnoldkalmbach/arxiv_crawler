"""Contextual search for papers using task-specific vector inference."""

from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from .dataloader import collate_embeddings_with_targets
from .inference import VectorInference


class ContextualSearch:
    """Semantic search for papers using task-specific vector inference.

    This class combines:
    1. A general embedding model (SentenceTransformer) for encoding text contexts
    2. A vector inference model for generating task-specific embeddings
    3. A FAISS index for fast nearest neighbor search

    The search pipeline:
    - Encode general context (e.g., paper title+abstract) with SentenceTransformer
    - Encode task context (the query) with SentenceTransformer token embeddings
    - Combine and pass through vector inference to get task vectors
    - Search FAISS index for similar paper embeddings
    """

    def __init__(
        self,
        general_model: SentenceTransformer,
        vector_inference: VectorInference,
        max_length: int = 256,
        pad_to_multiple_of: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the contextual search.

        Args:
            general_model: SentenceTransformer for encoding text contexts
            vector_inference: VectorInference instance for generating task vectors
            max_length: Maximum sequence length for padding
            pad_to_multiple_of: Pad sequences to multiple of this value
            device: Device to run inference on
        """
        self.general_model = general_model
        self.vector_inference = vector_inference
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.device = device

        # Will be set by build_index()
        self.paper_embeddings: pl.DataFrame | None = None
        self.knn_index: faiss.Index | None = None

        self.vector_inference.to(device)
        self.vector_inference.eval()

    def build_index(self, paper_embeddings_file: Path) -> None:
        """Builds IndexFlatIP and stores paper_embeddings and knn_index as instance attributes.

        Args:
            paper_embeddings_file: Path to parquet file with paper embeddings.
                Must have 'sentence_embedding' column.
        """
        paper_embeddings = pl.read_parquet(paper_embeddings_file).with_row_index("index")

        embedding_dim = paper_embeddings["sentence_embedding"].to_numpy().shape[1]
        knn_index = faiss.IndexFlatIP(embedding_dim)
        knn_index.add(paper_embeddings["sentence_embedding"].to_numpy().astype("float32"))

        # Store as instance attributes
        self.paper_embeddings = paper_embeddings
        self.knn_index = knn_index

    def get_matches(self, search_contexts: list[tuple[str, str]], top_k: int = 10) -> pl.DataFrame:
        """Returns flattened matches with query_index indicating which query each match belongs to.

        Args:
            search_contexts: List of (general_context, task_context) tuples.
                - general_context: Paper context (e.g., "{title}[SEP]{abstract}")
                - task_context: Task-specific context (e.g., citation sentence)
            top_k: Number of nearest neighbors to return per query

        Returns:
            DataFrame with columns: query_index, match_index, distance,
            plus all columns from paper_embeddings
        """
        if self.knn_index is None or self.paper_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        task_vectors = self.get_task_vectors(search_contexts)
        distances, indices = self.knn_index.search(
            task_vectors.cpu().numpy().astype("float32"), top_k
        )  # shape (N, top_k)

        # Note: np.ravel() returns a flattened view whenever possible, while np.flatten() always returns a new copy.
        n_queries = distances.shape[0]
        flat_query_indices = np.repeat(np.arange(n_queries), top_k)
        flat_match_indices = indices.ravel()
        flat_distances = distances.ravel()

        results_frame = pl.DataFrame(
            {
                "query_index": flat_query_indices,
                "match_index": flat_match_indices,
                "distance": flat_distances,
            }
        )
        # Join results with paper embeddings to get paper metadata
        return results_frame.join(self.paper_embeddings, left_on="match_index", right_on="index", how="left")

    def get_task_vectors(self, search_contexts: list[tuple[str, str]]) -> torch.Tensor:
        """Encodes each context as [general_sentence_embedding, task_token_embeddings] then passes through vector_inference.

        Args:
            search_contexts: List of (general_context, task_context) tuples

        Returns:
            Task vectors tensor of shape (batch_size, hidden_size), L2-normalized
        """
        general_contexts, task_contexts = zip(*search_contexts)

        # Get embeddings from general model
        general_embeddings = self.general_model.encode(
            general_contexts,
            output_value="sentence_embedding",
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        context_embeddings = self.general_model.encode(
            task_contexts,
            output_value=None,
            convert_to_numpy=False,
            convert_to_tensor=True,
        )

        # Combine embeddings for each sample: [general_embedding, context_token_embeddings]
        batch_inputs = []
        for i in range(len(search_contexts)):
            general_emb = general_embeddings[i].unsqueeze(0)  # [1, embed_dim]
            context_tokens = context_embeddings[i]["token_embeddings"]  # [seq_len, embed_dim]
            if context_tokens.shape[0] > self.max_length:
                print(f"Truncating context tokens from {context_tokens.shape[0]} to {self.max_length}")
            inputs_embeds = torch.vstack([general_emb, context_tokens])  # [1 + seq_len, embed_dim]
            batch_inputs.append((inputs_embeds,))  # Single-element tuple for inference (no targets)

        # Use collate function to batch and pad
        batch = collate_embeddings_with_targets(
            batch_inputs,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Move to device and get predictions
        inputs = batch["inputs"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.vector_inference(inputs_embeds=inputs, attention_mask=attention_mask)
            task_vectors = outputs["pooler_output"]

        task_vectors = task_vectors / torch.norm(task_vectors, dim=1, keepdim=True)

        return task_vectors


# Backwards compatibility alias
Inference = ContextualSearch

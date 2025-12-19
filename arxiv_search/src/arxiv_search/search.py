"""Contextual search for papers using task-specific vector inference."""

from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from .dataloader import collate_embeddings_with_targets
from .inference import DirectVectorInference, RectFlowVectorInference


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
        vector_inference: RectFlowVectorInference | DirectVectorInference,
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

        batch = self.make_vector_inference_inputs(search_contexts)
        with torch.no_grad():
            outputs = self.vector_inference(**batch)
            task_vectors = outputs["pooler_output"]
            task_vectors = task_vectors / torch.norm(task_vectors, dim=1, keepdim=True)

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

    def get_diverse_matches(
        self,
        search_contexts: list[tuple[str, str]],
        num_results: int = 10,
        num_query_variants: int = 10,
        candidates_per_variant: int | None = None,
        lambda_: float = 0.6,
        aspect_decay: float = 0.5,
        aspect_threshold: float = 0.7,
    ) -> pl.DataFrame:
        if self.knn_index is None or self.paper_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        if candidates_per_variant is None:
            candidates_per_variant = num_results

        n_queries = len(search_contexts)

        # Generate query variant vectors
        batch = self.make_vector_inference_inputs(search_contexts)
        with torch.no_grad():
            outputs = self.vector_inference(**batch, n_samples=num_query_variants, seed=42)
            query_vectors = outputs["pooler_output"]  # (n_queries, num_query_variants, hidden_size)
            query_vectors = query_vectors / torch.norm(query_vectors, dim=-1, keepdim=True)
            query_vectors = query_vectors.cpu().numpy().astype("float32")

        hidden_size = query_vectors.shape[-1]

        # Flatten for batched knn search: (n_queries * num_query_variants, hidden_size)
        query_vectors_flat = query_vectors.reshape(-1, hidden_size)

        print((query_vectors_flat @ query_vectors_flat.T).round(3))

        _, all_indices = self.knn_index.search(query_vectors_flat, candidates_per_variant)
        # all_indices: (n_queries * num_query_variants, candidates_per_variant)
        all_indices = all_indices.reshape(n_queries, num_query_variants, candidates_per_variant)

        # Get unique candidate indices per query and build mapping
        candidate_indices_per_query = []
        max_candidates = 0
        for q in range(n_queries):
            unique_indices = np.unique(all_indices[q].flatten())
            candidate_indices_per_query.append(unique_indices)
            max_candidates = max(max_candidates, len(unique_indices))

        # Batch reconstruct all unique candidates across all queries
        all_unique_indices = np.unique(np.concatenate(candidate_indices_per_query))
        print(
            f"Got {len(all_unique_indices)} unique candidates for {n_queries} queries x {num_query_variants} variants x {candidates_per_variant} candidates per variant"
        )
        all_candidate_vectors = np.stack([self.knn_index.reconstruct(int(i)) for i in all_unique_indices])
        all_candidate_vectors = all_candidate_vectors / np.linalg.norm(all_candidate_vectors, axis=1, keepdims=True)
        # Map from original index to position in all_candidate_vectors
        index_to_position = {idx: pos for pos, idx in enumerate(all_unique_indices)}

        # Compute all pairwise similarities at once
        # query_vectors_flat: (n_queries * num_query_variants, hidden_size)
        # all_candidate_vectors: (total_unique_candidates, hidden_size)
        all_similarities = (
            query_vectors_flat @ all_candidate_vectors.T
        )  # (n_queries * num_query_variants, total_unique_candidates)
        all_similarities = all_similarities.reshape(n_queries, num_query_variants, -1)

        candidate_candidate_sims = (
            all_candidate_vectors @ all_candidate_vectors.T
        )  # (total_unique_candidates, total_unique_candidates)

        # Run MMR per query
        results = []
        for q in range(n_queries):
            candidate_indices = candidate_indices_per_query[q]
            candidate_positions = np.array([index_to_position[i] for i in candidate_indices])

            # Slice similarity matrices for this query's candidates
            S = all_similarities[q][:, candidate_positions]  # (num_query_variants, num_candidates_q)
            C = candidate_candidate_sims[
                np.ix_(candidate_positions, candidate_positions)
            ]  # (num_candidates_q, num_candidates_q)

            selected_local, scores = self.multi_aspect_mmr(S, C, num_results, lambda_, aspect_decay, aspect_threshold)

            # Map back to original indices
            selected_indices = candidate_indices[selected_local]

            results.append(
                pl.DataFrame(
                    {
                        "query_index": np.full(len(selected_indices), q, dtype=np.int32),
                        "match_index": selected_indices,
                        "score": scores,
                    }
                )
            )

        results_frame = pl.concat(results)
        return results_frame.join(self.paper_embeddings, left_on="match_index", right_on="index", how="left")

    def multi_aspect_mmr(
        self,
        aspect_similarities: np.ndarray,
        candidate_similarities: np.ndarray,
        n_results: int,
        lambda_: float = 0.6,
        aspect_decay: float = 0.5,
        aspect_threshold: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select diverse results using aspect-aware Maximal Marginal Relevance.

        MMR iteratively selects candidates that balance relevance to the query aspects
        against redundancy with already-selected candidates. The aspect-aware extension
        tracks which aspects have been "covered" and downweights them in subsequent
        iterations, encouraging coverage across all query aspects.

        Loosly based on https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

        Args:
            aspect_similarities: Similarity scores between each query aspect and each
                candidate. Shape (num_aspects, num_candidates). Higher means more relevant.
            candidate_similarities: Pairwise similarity between candidates.
                Shape (num_candidates, num_candidates). Used to penalize redundancy.
            n_results: Number of candidates to select.
            lambda_: Tradeoff between relevance and diversity. 1.0 = pure relevance
                (equivalent to top-k), 0.0 = pure diversity (maximally spread out).
            aspect_decay: Multiplicative factor applied to an aspect's weight after
                selecting a candidate that scores above aspect_threshold on that aspect.
                Lower values cause faster "saturation" of aspects.
            aspect_threshold: Similarity threshold above which a candidate is considered
                to "cover" an aspect, triggering the decay.

        Returns:
            Tuple of (selected_indices, scores) where indices are positions in the
            candidate array and scores are the MMR scores at selection time.
        """
        num_aspects, num_candidates = aspect_similarities.shape

        selected_indices = []
        is_selected = np.zeros(num_candidates, dtype=bool)
        aspect_weights = np.ones(num_aspects)
        selected_scores = []

        for iteration in range(min(n_results, num_candidates)):
            # Relevance term: best weighted similarity across aspects for each candidate
            weighted_similarities = aspect_similarities * aspect_weights[:, np.newaxis]
            relevance_scores = np.max(weighted_similarities, axis=0)

            # Diversity term: max similarity to any already-selected candidate
            if selected_indices:
                redundancy_scores = np.max(candidate_similarities[:, selected_indices], axis=1)
            else:
                redundancy_scores = np.zeros(num_candidates)

            # MMR combines relevance and diversity
            mmr_scores = lambda_ * relevance_scores - (1 - lambda_) * redundancy_scores
            mmr_scores[is_selected] = -np.inf

            # Select the best candidate
            best_candidate = np.argmax(mmr_scores)
            selected_indices.append(best_candidate)
            is_selected[best_candidate] = True
            selected_scores.append(mmr_scores[best_candidate])

            # Log selection details
            print(
                f"[MMR iter {iteration + 1:2d}/{min(n_results, num_candidates):2d}] "
                f"selected idx={best_candidate:4d}, "
                f"mmr={mmr_scores[best_candidate]:6.3f}, "
                f"rel={relevance_scores[best_candidate]:6.3f}, "
                f"red={redundancy_scores[best_candidate]:6.3f}"
            )

            # Decay weight of aspects that this candidate covers well
            candidate_aspect_scores = aspect_similarities[:, best_candidate]
            covered_aspects = candidate_aspect_scores > aspect_threshold
            aspect_weights[covered_aspects] *= aspect_decay

        return np.array(selected_indices), np.array(selected_scores)

    def make_vector_inference_inputs(self, search_contexts: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        """Encodes each context as [general_sentence_embedding, task_token_embeddings] then passes through vector_inference.

        Args:
            search_contexts: List of (general_context, task_context) tuples

        Returns:
            Dictionary with "inputs_embeds" and "attention_mask" tensors.
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
        collated = collate_embeddings_with_targets(
            batch_inputs,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Move to device and set the right field names
        batch = {
            "inputs_embeds": collated["inputs"].to(self.device),
            "attention_mask": collated["attention_mask"].to(self.device),
        }

        return batch

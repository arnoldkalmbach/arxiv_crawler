import torch
import faiss
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
from .dataloader import collate_embeddings_with_targets
import polars as pl
from pathlib import Path

class Inference:
    def __init__(
        self, 
        general_model: SentenceTransformer, 
        task_model: BertModel, 
        max_length: int = 256,
        pad_to_multiple_of: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.general_model = general_model
        self.task_model = task_model
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.device = device
        
        # Will be set by build_index()
        self.paper_embeddings: pl.DataFrame | None = None
        self.knn_index: faiss.Index | None = None
        
        self.task_model.to(device)
        self.task_model.eval()

    def build_index(self, paper_embeddings_file: Path) -> None:
        """Builds IndexFlatIP and stores paper_embeddings and knn_index as instance attributes."""
        paper_embeddings = pl.read_parquet(paper_embeddings_file).with_row_index('index')
        
        embedding_dim = paper_embeddings["sentence_embedding"].to_numpy().shape[1]
        knn_index = faiss.IndexFlatIP(embedding_dim)
        knn_index.add(paper_embeddings["sentence_embedding"].to_numpy().astype('float32'))
        
        # Store as instance attributes
        self.paper_embeddings = paper_embeddings
        self.knn_index = knn_index

    def get_matches(self, search_contexts: list[tuple[str, str]], top_k: int = 10) -> pl.DataFrame:
        """Returns flattened matches with query_index indicating which query each match belongs to."""
        if self.knn_index is None or self.paper_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        task_vectors = self.get_task_vectors(search_contexts)
        distances, indices = self.knn_index.search(task_vectors.cpu().numpy().astype('float32'), top_k)  # shape (N, top_k)

        # Note: np.ravel() returns a flattened view whenever possible, while np.flatten() always returns a new copy.
        n_queries = distances.shape[0]
        flat_query_indices = np.repeat(np.arange(n_queries), top_k)
        flat_match_indices = indices.ravel()
        flat_distances = distances.ravel()

        results_frame = pl.DataFrame({
            "query_index": flat_query_indices,
            "match_index": flat_match_indices,
            "distance": flat_distances,
        })
        # Join results with paper embeddings to get paper metadata
        return results_frame.join(self.paper_embeddings, left_on="match_index", right_on="index", how="left")


    def get_task_vectors(self, search_contexts: list[tuple[str, str]]) -> torch.Tensor:
        """Encodes each context as [general_sentence_embedding, task_token_embeddings] then passes through task_model."""
        general_contexts, task_contexts = zip(*search_contexts)
        
        # Get embeddings from general model
        general_embeddings = self.general_model.encode(
            general_contexts, 
            output_value='sentence_embedding',
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
            context_tokens = context_embeddings[i]['token_embeddings']  # [seq_len, embed_dim]
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
            outputs = self.task_model(inputs_embeds=inputs, attention_mask=attention_mask)
            task_vectors = outputs["pooler_output"]

        task_vectors = task_vectors / torch.norm(task_vectors, dim=1, keepdim=True)
        
        return task_vectors



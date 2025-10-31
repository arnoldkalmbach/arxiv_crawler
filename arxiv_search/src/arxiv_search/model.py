"""Model architecture for citation embedding."""

import torch
from transformers import BertConfig, BertModel


def create_model(
    device: str = "cuda",
    hidden_size: int = 768,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 12,
    intermediate_size: int = 1536,
    max_position_embeddings: int = 2048,
) -> BertModel:
    """
    Create BERT model for citation embedding.

    Uses a small BERT architecture with:
    - Single encoder layer
    - Pooler layer for generating fixed-size embeddings
    - Custom dimensions for efficiency

    Args:
        device: Device to load model on ('cuda' or 'cpu')
        hidden_size: Dimension of hidden layers (default: 768)
        num_hidden_layers: Number of transformer layers (default: 1)
        num_attention_heads: Number of attention heads (must divide hidden_size)
        intermediate_size: Dimension of feed-forward layer (default: 1536)
        max_position_embeddings: Maximum sequence length (default: 2048)

    Returns:
        BERT model instance
    """
    cfg = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        add_pooling_layer=True,  # enables CLS pooler (dense + tanh)
        vocab_size=1,  # unused since we pass inputs_embeds
    )
    model = BertModel(cfg).to(device)
    return model


def load_model(checkpoint_path: str, device: str = "cuda", **model_kwargs) -> BertModel:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on
        **model_kwargs: Additional arguments passed to create_model()

    Returns:
        Loaded BERT model instance
    """
    model = create_model(device=device, **model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

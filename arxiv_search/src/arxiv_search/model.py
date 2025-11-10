"""Model architecture for citation embedding."""

import torch
import torch.nn as nn
from typing import Optional, Type
from timm.models.vision_transformer import Attention
import math
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


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock1d(nn.Module):
    def __init__(self, hidden_size, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm_attn = nn.LayerNorm(hidden_size)

        self.mlp = MLP(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))
        self.norm_mlp = nn.LayerNorm(hidden_size)

        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6, bias=True))

    def forward(self, x, c):
        print(x.shape, c.shape)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=1)

        print(shift_attn.shape, scale_attn.shape, gate_attn.shape, shift_mlp.shape, scale_mlp.shape, gate_mlp.shape)

        attn_out = self.attn(self.norm_attn(shift_attn[:, None] + (scale_attn[:, None] + 1) * x))

        mlp_out = self.mlp(self.norm_mlp(shift_mlp[:, None] + (scale_mlp[:, None] + 1) * x))

        x = x + gate_attn[:, None] * attn_out + gate_mlp[:, None] * mlp_out

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True)  # Regular self-attention
        self.norm_attn = nn.LayerNorm(hidden_size)

        self.mlp = MLP(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))
        self.norm_mlp = nn.LayerNorm(hidden_size)

    def forward(self, x, c):
        attn_out = self.attn(torch.cat([self.norm_attn(x), c], dim=1))
        mlp_out = self.mlp(self.norm_mlp(x))

        x = x + attn_out[:, : x.shape[1]] + mlp_out

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class VelocityField1dCrossAttention(nn.Module):
    def __init__(self, num_blocks, conditioning_model, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        hidden_size = conditioning_model.config.hidden_size

        self.time_embedder = TimestepEmbedder(hidden_size)
        self.conditioning_model = conditioning_model
        self.position_embeddings = conditioning_model.embeddings.position_embeddings
        self.position_ids = conditioning_model.embeddings.position_ids
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.act_out = conditioning_model.pooler.activation

    def forward(
        self,
        x,  # batch, seq_len, hidden_size
        t,  # batch
        y,  # batch, seq_len, hidden_size
        attention_mask,  # batch, seq_len
    ):
        _, seq_len, _ = x.shape
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)

        t_emb = self.time_embedder(t)[:, None, :]  # -> batch, 1, hidden_size
        y_emb = self.conditioning_model(inputs_embeds=y, attention_mask=attention_mask).last_hidden_state
        c = y_emb + t_emb + position_embeddings

        x = x + position_embeddings

        for block in self.blocks:
            x = block(x, c)

        return self.act_out(self.fc_out(x[:, 0]))


class VelocityField1dDiT(nn.Module):
    def __init__(self, num_blocks, conditioning_model, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        hidden_size = conditioning_model.config.hidden_size

        self.time_embedder = TimestepEmbedder(hidden_size)
        self.conditioning_model = conditioning_model
        self.position_embeddings = conditioning_model.embeddings.position_embeddings
        self.position_ids = conditioning_model.embeddings.position_ids
        self.blocks = nn.ModuleList([DiTBlock1d(hidden_size, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.act_out = conditioning_model.pooler.activation

    def forward(
        self,
        x,  # batch, seq_len, hidden_size
        t,  # batch
        y,  # batch, seq_len, hidden_size
        attention_mask,  # batch, seq_len
    ):
        _, seq_len, _ = x.shape
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)

        t_emb = self.time_embedder(t)  # -> batch, hidden_size
        y_emb = self.conditioning_model(inputs_embeds=y, attention_mask=attention_mask).pooler_output
        c = y_emb + t_emb

        x = x + position_embeddings

        for block in self.blocks:
            x = block(x, c)

        return self.act_out(self.fc_out(x[:, 0]))

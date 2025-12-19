"""Vector inference wrappers for different model architectures."""

from abc import ABC, abstractmethod

import torch
from rectified_flow import RectifiedFlow
from rectified_flow.samplers import CurvedEulerSampler
from transformers import BertModel


class VectorInference(ABC):
    """Base class for models that generate task vectors from input embeddings.

    All implementations must support:
    - `.to(device)` - move model to device
    - `.eval()` - set to evaluation mode
    - `__call__(inputs_embeds, attention_mask)` - generate task vectors

    The __call__ method returns a dict with 'pooler_output' key containing
    the generated task vectors of shape (batch_size, hidden_size).
    """

    @abstractmethod
    def to(self, device: str) -> "VectorInference":
        """Move model to device."""
        pass

    @abstractmethod
    def eval(self) -> "VectorInference":
        """Set to evaluation mode."""
        pass

    @abstractmethod
    def __call__(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Generate task vectors from input embeddings.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Dict with 'pooler_output' key containing task vectors of shape (batch_size, hidden_size)
        """
        pass


class DirectVectorInference(VectorInference):
    """Wraps BertModel for direct vector inference.

    The model takes input embeddings and produces task vectors directly
    via the pooler output.
    """

    def __init__(self, model: BertModel):
        """Initialize with a BertModel.

        Args:
            model: A BertModel instance (typically created via create_model or load_model)
        """
        self.model = model

    def to(self, device: str) -> "DirectVectorInference":
        """Move model to device."""
        self.model.to(device)
        return self

    def eval(self) -> "DirectVectorInference":
        """Set to evaluation mode."""
        self.model.eval()
        return self

    def __call__(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Generate task vectors via forward pass through BertModel.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Dict with 'pooler_output' containing task vectors
        """
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return {"pooler_output": outputs["pooler_output"]}


class RectFlowVectorInference(VectorInference):
    """Wraps RectifiedFlow for flow-based vector inference.

    The model generates task vectors by running a flow-based sampler
    from noise to the target embedding space, conditioned on input embeddings.
    """

    def __init__(self, rectified_flow: RectifiedFlow, num_steps: int = 100):
        """Initialize with a RectifiedFlow model.

        Args:
            rectified_flow: A RectifiedFlow instance with trained velocity field
            num_steps: Number of integration steps for sampling (default: 50)
        """
        self.rectified_flow = rectified_flow
        self.sampler = CurvedEulerSampler(rectified_flow=rectified_flow)
        self.hidden_size = rectified_flow.velocity_field.conditioning_model.config.hidden_size
        self.num_steps = num_steps
        self._device = next(rectified_flow.velocity_field.parameters()).device

    def to(self, device: str) -> "RectFlowVectorInference":
        """Move model to device."""
        self.rectified_flow.velocity_field.to(device)
        self._device = device
        return self

    def eval(self) -> "RectFlowVectorInference":
        """Set to evaluation mode."""
        self.rectified_flow.velocity_field.eval()
        return self

    def __call__(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, n_samples: int = 1, seed: int = 42
    ) -> dict:
        """Generate task vectors via flow-based sampling.

        Samples from noise and integrates the flow conditioned on input embeddings.

        Args:
            inputs_embeds: Conditioning embeddings of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Dict with 'pooler_output' containing task vectors (final trajectory point)
        """
        batch_size = inputs_embeds.shape[0]
        generator = torch.Generator(device=self._device).manual_seed(seed)
        X0 = torch.randn(batch_size * n_samples, self.hidden_size, device=self._device, generator=generator)

        self.sampler.sample_loop(
            num_steps=self.num_steps,
            x_0=X0,
            y=inputs_embeds,
            attention_mask=attention_mask,
        )

        pred_embeddings = self.sampler.trajectories[-1]
        return {"pooler_output": pred_embeddings.reshape(batch_size, n_samples, self.hidden_size)}

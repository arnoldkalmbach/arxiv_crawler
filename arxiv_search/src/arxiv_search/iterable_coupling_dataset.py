"""
IterableCouplingDataset for rectified flow training with IterableDataset support.

Provides deterministic noise pairing using hash-based seeding, allowing the same
sample to always get the same noise vector without storing noise data.
"""

import hashlib
from functools import partial
from typing import Callable, Hashable, Optional

import torch
import torch.distributions as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset


def _to_tensor(x):
    """Convert input to tensor if not already a tensor."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


class IterableCouplingDataset(IterableDataset):
    """
    Wraps an IterableDataset to provide (X0, X1) pairs for rectified flow training.

    Uses deterministic noise pairing: same sample always gets same noise via hash-based seeding.
    This allows deterministic pairing without storing noise vectors.
    """

    def __init__(
        self,
        D1: IterableDataset,
        D0: IterableDataset | dist.Distribution | None = None,
        reflow: bool = False,
        extract_target: Callable | None = None,
        extract_key: Callable | None = None,
        extract_conditioning: Callable | None = None,
        extract_metadata: Callable | None = None,
    ):
        """
        Args:
            D1: IterableDataset yielding target samples (or tuples where target is extracted)
            D0: Optional source samples. Can be:
                - None: defaults to standard normal (deterministic per sample via hash)
                - IterableDataset: source samples (must be same length as D1 if reflow=True)
                - dist.Distribution: distribution to sample from (deterministic per sample via hash)
            reflow: If True, D0 and D1 are paired. If False, D0 is sampled independently.
            extract_target: Optional callable to extract target from D1 samples.
                If D1 yields tuples, this extracts the target (e.g., lambda x: x[1] for (input, output) pairs)
                If None, assumes D1 yields tensors directly.
            extract_key: Optional callable to extract a hashable key from D1 samples for deterministic pairing.
                If D1 yields tuples with metadata, use e.g., lambda x: x[2]["reference_id"]
                If None, will hash the target tensor itself (slower but works).
            extract_conditioning: Optional callable to extract conditioning information from D1 samples.
                If D1 yields tuples like (inputs_embeds, output_embeds), use lambda x: x[0] to get inputs_embeds.
                If None, no conditioning information is passed along.
        """
        self.D1 = D1
        self.D0 = D0
        self.reflow = reflow
        self.extract_target = extract_target
        self.extract_key = extract_key
        self.extract_conditioning = extract_conditioning
        self.extract_metadata = extract_metadata

        if self.reflow:
            if self.D0 is None:
                raise ValueError("D0 must be provided when reflow=True")
            if not isinstance(self.D0, IterableDataset):
                raise ValueError("When reflow=True, D0 must be an IterableDataset")

        else:
            if isinstance(self.D0, IterableDataset):
                raise ValueError("When reflow=False, D0 must be a distribution or None")

    def _get_target(self, sample):
        """Extract target from D1 sample."""
        if self.extract_target is not None:
            return self.extract_target(sample)
        return sample

    def _get_key(self, sample):
        """Extract hashable key from sample for deterministic pairing."""
        if self.extract_key is not None:
            return self.extract_key(sample)

        # Fallback: hash the target tensor
        target = self._get_target(sample)
        if isinstance(target, torch.Tensor):
            # Hash tensor content (convert to bytes)
            tensor_bytes = target.detach().cpu().numpy().tobytes()
            return hashlib.md5(tensor_bytes).hexdigest()
        else:
            # Fallback: hash string representation
            return hash(str(sample))

    def _get_metadata(self, sample) -> Optional[dict]:
        """Extract metadata from sample if available."""
        if self.extract_metadata is not None:
            return self.extract_metadata(sample)

        # Auto-detect metadata when the sample is a tuple/list with a trailing dict
        if isinstance(sample, (tuple, list)) and len(sample) >= 3:
            maybe_meta = sample[-1]
            if isinstance(maybe_meta, dict):
                return maybe_meta

        return None

    def _seed_from_key(self, key: Hashable) -> int:
        """Convert key to integer seed."""
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            # Hash string to int
            return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)
        else:
            # Hash any hashable object
            return hash(key) % (2**31)

    def _sample_noise_deterministic(self, key: Hashable, shape: tuple, generator: torch.Generator) -> torch.Tensor:
        """Sample noise deterministically based on key."""
        seed = self._seed_from_key(key)
        # Set seed on the provided generator (reused across samples)
        generator.manual_seed(seed)
        # Sample from standard normal
        return torch.randn(shape, generator=generator)

    def _sample_from_dist_deterministic(
        self, key: Hashable, distribution: dist.Distribution, shape: tuple
    ) -> torch.Tensor:
        """Sample from distribution deterministically based on key."""
        seed = self._seed_from_key(key)
        torch.manual_seed(seed)
        return distribution.sample(shape)

    def __iter__(self):
        """Yield (X0, X1, conditioning) tuples for rectified flow training."""
        # Create a generator once per iteration (reused for all samples in this epoch)
        # Each worker process gets its own dataset instance, so this is thread-safe
        generator = torch.Generator()

        # Create iterator
        D1_iter = iter(self.D1)

        if self.reflow:
            # Paired mode: iterate D0 and D1 together
            D0_iter = iter(self.D0)

            for d1_sample in D1_iter:
                try:
                    d0_sample = next(D0_iter)
                except StopIteration:
                    break

                X1 = self._get_target(d1_sample)
                X0 = d0_sample if self.extract_target is None else self.extract_target(d0_sample)

                # Extract conditioning if available
                conditioning = None
                if self.extract_conditioning is not None:
                    conditioning = self.extract_conditioning(d1_sample)

                metadata = self._get_metadata(d1_sample)

                if conditioning is not None and metadata is not None:
                    yield X0, X1, conditioning, metadata
                elif conditioning is not None:
                    yield X0, X1, conditioning
                elif metadata is not None:
                    yield X0, X1, metadata
                else:
                    yield X0, X1
        else:
            # Independent mode: sample X0 deterministically based on sample key
            for d1_sample in D1_iter:
                X1 = self._get_target(d1_sample)
                key = self._get_key(d1_sample)

                if self.D0 is None:
                    # Sample from standard normal deterministically
                    X0 = self._sample_noise_deterministic(key, X1.shape, generator)
                elif isinstance(self.D0, dist.Distribution):
                    # Sample from provided distribution deterministically
                    X0 = self._sample_from_dist_deterministic(key, self.D0, X1.shape)
                # Disabled in init
                # elif isinstance(self.D0, IterableDataset):
                else:
                    raise ValueError(f"Unsupported D0 type: {type(self.D0)}")

                # Extract conditioning if available
                conditioning = None
                if self.extract_conditioning is not None:
                    conditioning = self.extract_conditioning(d1_sample)

                metadata = self._get_metadata(d1_sample)

                if conditioning is not None and metadata is not None:
                    yield X0, X1, conditioning, metadata
                elif conditioning is not None:
                    yield X0, X1, conditioning
                elif metadata is not None:
                    yield X0, X1, metadata
                else:
                    yield X0, X1


def collate_batch_elements(batch_elements):
    """
    Collate a list of batch elements (tensors, tuples, dicts, etc.).

    This is adapted from rectified-flow's coupling_collate_fn to handle
    nested structures.
    """
    if isinstance(batch_elements[0], torch.Tensor):
        return torch.stack(batch_elements)
    elif isinstance(batch_elements[0], (tuple, list)):
        # Recursively collate each element in the tuple/list
        transposed = list(zip(*batch_elements))
        return [collate_batch_elements(elements) for elements in transposed]
    elif isinstance(batch_elements[0], dict):
        # Recursively collate each value in the dict
        collated = {}
        for key in batch_elements[0]:
            collated[key] = collate_batch_elements([d[key] for d in batch_elements])
        return collated
    else:
        # If it's a scalar or unstackable type, return as a list
        return batch_elements


def collate_coupling_with_embeddings(
    batch: list[tuple],
    max_length: int | None = None,
    pad_to_multiple_of: int | None = None,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.bool,
):
    """
    Collate function for batching (X0, X1, inputs_embeds) tuples from IterableCouplingDataset.

    X0 and X1 are vectors [embed_dim] and are just stacked.
    inputs_embeds is a sequence [seq_len, embed_dim] and is padded with attention masks.

    Args:
        batch: list of tuples from IterableCouplingDataset:
            - (X0, X1) if no conditioning
            - (X0, X1, inputs_embeds) if conditioning is present
        max_length: Maximum sequence length for inputs_embeds (truncate if longer)
        pad_to_multiple_of: Pad inputs_embeds to multiple of this value
        pad_value: Value to use for padding
        mask_dtype: Data type for attention mask

    Returns:
        Dictionary containing:
            - X0: FloatTensor [B, embed_dim] - batched X0 (noise vectors)
            - X1: FloatTensor [B, embed_dim] - batched X1 (target vectors)
            - inputs: FloatTensor [B, L, K] - batched inputs_embeds (only if conditioning present)
            - attention_mask: Bool/Int Tensor [B, L] - only if inputs_embeds present
            - lengths: LongTensor [B] - only if inputs_embeds present
            - metadata: list of dicts - only if metadata present
    """
    # Extract metadata if present as the last element
    has_metadata = isinstance(batch[0][-1], dict)
    metadata_list = [elem[-1] for elem in batch] if has_metadata else None

    # Remove metadata for core processing
    core_batch = [elem[:-1] if has_metadata else elem for elem in batch]

    # Determine batch structure on the core elements
    batch_len = len(core_batch[0])
    has_conditioning = batch_len == 3

    if has_conditioning:
        X0_list, X1_list, inputs_embeds_list = zip(*core_batch)
    else:
        X0_list, X1_list = zip(*core_batch)
        inputs_embeds_list = None

    # Convert to tensors
    X0_tensors = [_to_tensor(x) for x in X0_list]
    X1_tensors = [_to_tensor(x) for x in X1_list]

    # X0 and X1 are vectors, just stack them
    result = {
        "X0": torch.stack(X0_tensors, dim=0),  # [B, embed_dim]
        "X1": torch.stack(X1_tensors, dim=0),  # [B, embed_dim]
    }

    # Handle inputs_embeds (conditioning) if present
    if has_conditioning and inputs_embeds_list is not None:
        inputs_embeds_tensors = [_to_tensor(x) for x in inputs_embeds_list]

        # Apply truncation
        if max_length is not None:
            inputs_embeds_tensors = [x[:max_length] for x in inputs_embeds_tensors]

        # Compute lengths
        lengths = torch.tensor([x.size(0) for x in inputs_embeds_tensors], dtype=torch.long)

        # Pad sequences
        if pad_to_multiple_of is not None:
            max_len = int(lengths.max().item())
            if max_len % pad_to_multiple_of != 0:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

            K = inputs_embeds_tensors[0].size(1)
            padded_list = []
            for x in inputs_embeds_tensors:
                if x.size(0) < max_len:
                    pad_rows = max_len - x.size(0)
                    pad_block = x.new_full((pad_rows, K), pad_value)
                    padded_list.append(torch.cat([x, pad_block], dim=0))
                else:
                    padded_list.append(x)
            inputs = torch.stack(padded_list, dim=0)  # [B, L, K]
        else:
            inputs = pad_sequence(inputs_embeds_tensors, batch_first=True, padding_value=pad_value)  # [B, L, K]

        # Build attention mask
        L = inputs.size(1)
        arange = torch.arange(L).unsqueeze(0)  # [1, L]
        attention_mask = arange < lengths.unsqueeze(1)
        attention_mask = attention_mask.to(mask_dtype)

        result["inputs"] = inputs
        result["attention_mask"] = attention_mask
        result["lengths"] = lengths

    # Attach metadata if provided
    if metadata_list is not None:
        result["metadata"] = metadata_list

    return result


def get_coupling_collate_fn(
    max_length: int = 256,
    pad_to_multiple_of: int = 8,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.bool,
):
    """
    Get a collate function configured for IterableCouplingDataset.

    Args:
        max_length: Maximum sequence length for inputs_embeds
        pad_to_multiple_of: Pad inputs_embeds to multiple of this value
        pad_value: Padding value
        mask_dtype: Data type for attention mask

    Returns:
        Partial function configured for collating coupling pairs
    """
    return partial(
        collate_coupling_with_embeddings,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_value=pad_value,
        mask_dtype=mask_dtype,
    )

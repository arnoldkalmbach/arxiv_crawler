from functools import partial
from typing import Any, Optional
import torch
import torch.optim as optim
from transformers import BertConfig, BertModel
from torch.utils.tensorboard import SummaryWriter

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from arxiv_search.dataloader import CitationEmbeddingDataset


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def collate_embeddings_with_targets(
    batch: list[tuple[torch.Tensor, Any]],
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.bool,
):
    """
    batch: list of (X, y)
      X: [seq_len, embed_dim] (float tensor or array)
      y: target vector (same shape across samples; tensor/array/number/dict ok)

    Returns:
      {
        "inputs": FloatTensor [B, L, K],
        "attention_mask": Bool/Int Tensor [B, L] (True/1 = real token),
        "lengths": LongTensor [B],  # pre-padding lengths
        "targets": stacked y (shape depends on your dataset; typically [B, T])
      }
    """
    # Convert to tensors and apply truncation
    Xs, Ys = [], []
    for X, y in batch:
        Xt = _to_tensor(X)  # [Li, K]
        if max_length is not None:
            Xt = Xt[:max_length]
        Xs.append(Xt)
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
        # pad_sequence itself can’t force a larger-than-maximum length, so we manually
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

    # Stack targets robustly (supports tensors, numbers, tuples, dicts, etc.)
    targets = default_collate([_to_tensor(y) for y in Ys])

    return {
        "inputs": inputs,  # [B, L, K], float
        "attention_mask": attention_mask,  # [B, L], bool (or chosen dtype)
        "lengths": lengths,  # [B]
        "targets": targets,  # e.g., [B, T] or [B] depending on your dataset
    }


collate_fn = partial(
    collate_embeddings_with_targets,
    max_length=256,
    pad_to_multiple_of=8,
    pad_value=0.0,
    mask_dtype=torch.bool,
)


train_ds = CitationEmbeddingDataset(
    citations_file="data/train/citations.jsonl",
    paper_embeddings_file="data/paper_embeddings.parquet",
    citation_embeddings_dir="data/train",
    citations_batch_size=10000,
)
print("Created train dataset")

batch_size = 256
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=6,
    collate_fn=collate_fn,
    persistent_workers=True,  # Keep workers alive to avoid reloading data
    pin_memory=True,  # Faster GPU transfers
    prefetch_factor=3,  # Prefetch more batches per worker
)
print("Created train loader")

# 1) Tiny BERT: single encoder layer, keep pooler
cfg = BertConfig(
    hidden_size=768,
    num_hidden_layers=1,  # one attention block
    num_attention_heads=12,  # must divide 768
    intermediate_size=1536,  # smaller FFN (optional)
    max_position_embeddings=2048,  # adjust as needed
    add_pooling_layer=True,  # enables CLS pooler (dense + tanh)
    vocab_size=1,  # unused since we pass inputs_embeds
)
device = "cuda"
model = BertModel(cfg).to(device)
print("Created model")


optimizer = optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 20
log_steps = 10
save_steps = 500
step = 0

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs")

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()  # Clear gradients from previous step

        X = batch["inputs"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["targets"].to(device)
        lengths = batch["lengths"]
        preds = model(inputs_embeds=X, attention_mask=mask)

        # Compute cosine similarity for the batch
        cossim = torch.cosine_similarity(preds["pooler_output"], y).mean()
        loss = -cossim

        loss.backward()
        optimizer.step()

        step += 1

        # Compute sequence length statistics (non-padding tokens)
        lengths_float = lengths.float()
        avg_seq_length = lengths_float.mean().item()
        p5_seq_length = torch.quantile(lengths_float, 0.05).item()
        p95_seq_length = torch.quantile(lengths_float, 0.95).item()

        # Log to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("CosineSimilarity/train", cossim.item(), step)
        writer.add_scalar("SequenceLength/avg", avg_seq_length, step)
        writer.add_scalar("SequenceLength/p5", p5_seq_length, step)
        writer.add_scalar("SequenceLength/p95", p95_seq_length, step)

        if step % log_steps == 0:
            print(
                f"[{step}] Cossim: {cossim.item():.4f}, Loss: {loss.item():.4f}, Avg Tokens: {avg_seq_length:.1f} (p5: {p5_seq_length:.1f}, p95: {p95_seq_length:.1f})"
            )

        if step % save_steps == 0:
            torch.save(model.state_dict(), f"models/model_{step}.pth")
            print(f"[{step}] Saved model to models/model_{step}.pth")

writer.close()


from collections.abc import Sequence

import torch
import torch.nn.functional as F

from embedding_core import encode_texts


def loss_function(
    similarity: torch.Tensor,
    target_value: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Measure how far cosine similarities are from their desired values."""
    target = torch.as_tensor(
        target_value,
        dtype=similarity.dtype,
        device=similarity.device,
    ).reshape(-1)

    if target.numel() == 1:
        target = target.expand_as(similarity)
    elif target.shape != similarity.shape:
        raise ValueError("Provide one target per text pair")

    return F.mse_loss(similarity, target)


def _as_batch(text: str | Sequence[str]) -> list[str]:
    return [text] if isinstance(text, str) else list(text)


def train(
    text_a: str | Sequence[str],
    text_b: str | Sequence[str],
    target_value: float | Sequence[float] | torch.Tensor,
    tokenizer,
    model,
    optimizer: torch.optim.Optimizer,
    task_prefix: str | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Run one fine-tuning step over one pair or a batch of text pairs.

    Use a target of 1 for texts that should be close and 0 (or -1) for texts
    that should be separated. The returned value is the batch loss.
    """
    texts_a = _as_batch(text_a)
    texts_b = _as_batch(text_b)
    if not texts_a or len(texts_a) != len(texts_b):
        raise ValueError("text_a and text_b must contain the same number of texts")

    model.train()
    optimizer.zero_grad()

    embedding_a = encode_texts(
        texts_a,
        tokenizer,
        model,
        task_prefix=task_prefix,
        device=device,
    )
    embedding_b = encode_texts(
        texts_b,
        tokenizer,
        model,
        task_prefix=task_prefix,
        device=device,
    )
    similarity = F.cosine_similarity(embedding_a, embedding_b)
    loss = loss_function(similarity, target_value)
    loss.backward()
    optimizer.step()

    return loss.detach().item()

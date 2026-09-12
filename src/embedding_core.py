from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1",
}


@dataclass(frozen=True)
class EmbeddingModelConfig:
    path: str
    prefix: str | None = None
    batch_size: int = 32
    trust_remote_code: bool = False
    device: str = "auto"


def create_model_name(name: str) -> str:
    try:
        return MODELS[name]
    except KeyError as error:
        choices = ", ".join(MODELS)
        raise ValueError(
            f"Unknown model '{name}'. Available choices: {choices}"
        ) from error


def create_model_config(config: str | dict) -> EmbeddingModelConfig:
    """Normalize the compact model name and full dictionary configurations."""
    if isinstance(config, str):
        config = {"name": config}

    if not isinstance(config, dict):
        raise TypeError("Model config must be a model name or a dictionary")

    path = config.get("path")
    if not path:
        try:
            path = create_model_name(config["name"])
        except KeyError as error:
            raise ValueError("Model config requires either 'name' or 'path'") from error

    batch_size = config.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Model batch_size must be a positive integer")

    return EmbeddingModelConfig(
        path=path,
        prefix=config.get("prefix"),
        batch_size=batch_size,
        trust_remote_code=config.get("trust_remote_code", False),
        device=config.get("device", "auto"),
    )


def add_task_prefix(text: str, task_prefix: str | None) -> str:
    if not task_prefix:
        return text

    prefix = task_prefix.strip().rstrip(":")
    if not prefix:
        return text
    return f"{prefix}: {text}"


def create_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"

    return torch.device(device_name)


def load_model(
    config: EmbeddingModelConfig,
) -> tuple[Any, Any, torch.device]:
    """Load a tokenizer and model using one shared configuration path."""
    device = create_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.path, trust_remote_code=config.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        config.path, trust_remote_code=config.trust_remote_code
    )
    model.to(device)
    return tokenizer, model, device


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings, excluding padding tokens."""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    task_prefix: str | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert texts to normalized embeddings while preserving autograd."""
    sentences = [add_task_prefix(text, task_prefix) for text in texts]
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {
        name: value.to(device) for name, value in encoded_input.items()
    }

    model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)

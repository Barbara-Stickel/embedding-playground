from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from data_types import ExpenseDataset, ExpenseRecord


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
    """Create model settings while supporting the original string config."""
    if isinstance(config, str):
        config = {"name": config}

    path = config.get("path") or create_model_name(config["name"])

    return EmbeddingModelConfig(
        path=path,
        prefix=config.get("prefix"),
        batch_size=config.get("batch_size", 32),
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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_records(
    records: list[ExpenseRecord],
    tokenizer,
    model,
    task_prefix: str | None = None,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        sentences = [add_task_prefix(record.text, task_prefix) for record in batch]

        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {
            name: value.to(device) for name, value in encoded_input.items()
        }

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        for record, embedding in zip(batch, sentence_embeddings):
            record.embedding = embedding.cpu().numpy()


def embed(data: ExpenseDataset, model_name: str | dict) -> None:
    config = create_model_config(model_name)
    device = create_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.path, trust_remote_code=config.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        config.path, trust_remote_code=config.trust_remote_code
    )
    model.to(device)
    model.eval()

    embed_records(
        data.train,
        tokenizer,
        model,
        task_prefix=config.prefix,
        batch_size=config.batch_size,
        device=device,
    )
    embed_records(
        data.test,
        tokenizer,
        model,
        task_prefix=config.prefix,
        batch_size=config.batch_size,
        device=device,
    )

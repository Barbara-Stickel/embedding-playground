import torch

from data_types import ExpenseDataset, ExpenseRecord
from embedding_core import (
    create_model_config,
    encode_texts,
    load_model,
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

        with torch.no_grad():
            sentence_embeddings = encode_texts(
                [record.text for record in batch],
                tokenizer,
                model,
                task_prefix=task_prefix,
                device=device,
            )

        for record, embedding in zip(batch, sentence_embeddings):
            record.embedding = embedding.cpu().numpy()


def embed(data: ExpenseDataset, model_name: str | dict) -> None:
    config = create_model_config(model_name)
    tokenizer, model, device = load_model(config)
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

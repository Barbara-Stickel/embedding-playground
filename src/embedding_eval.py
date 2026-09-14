from contextlib import nullcontext

import torch
from tqdm import tqdm

from data_types import TextDataset, TextRecord
from embedding_cache import EmbeddingCache, DEFAULT_DIRECTORY
from embedding_core import (
    create_model_config,
    encode_texts,
    load_model,
)


def embed_records(
    records: list[TextRecord],
    tokenizer,
    model,
    task_prefix: str | None = None,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    description: str = "Embedding records",
    cache: EmbeddingCache | None = None,
) -> None:
    with tqdm(total=len(records), desc=description, unit="record", dynamic_ncols=True) as progress:
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
            if cache is not None:
                cache.put_many({
                    cache.text_key(record.text, task_prefix): record.embedding
                    for record in batch
                })
            progress.update(len(batch))


def restore_cached_embeddings(
    records: list[TextRecord], cache: EmbeddingCache, task_prefix: str | None,
) -> list[TextRecord]:
    unique_records = {record.text: record for record in records}
    keys = {text: cache.text_key(text, task_prefix) for text in unique_records}
    cached = cache.get_many(list(keys.values()))
    for record in records:
        record.embedding = cached.get(keys[record.text])
    pending = [record for record in unique_records.values() if record.embedding is None]
    print(f"Embedding cache: {len(cached)} / {len(unique_records)} unique texts reused; {len(pending)} to compute.")
    print(f"Cache file: {cache.path}")
    return pending


def embed(
    data: TextDataset,
    model_name: str | dict,
    cache_config: dict | None = None,
) -> None:
    config = create_model_config(model_name)
    cache_config = cache_config or {}
    cache_context = (
        EmbeddingCache(config, cache_config.get("directory", DEFAULT_DIRECTORY))
        if cache_config.get("enabled", True) else nullcontext()
    )
    with cache_context as cache:
        records = data.train + data.test
        pending = restore_cached_embeddings(records, cache, config.prefix) if cache is not None else records
        if pending:
            tokenizer, model, device = load_model(config)
            model.eval()
            embed_records(
                pending, tokenizer, model,
                task_prefix=config.prefix,
                batch_size=config.batch_size,
                device=device,
                description="Embedding uncached texts" if cache is not None else "Embedding records",
                cache=cache,
            )
            if cache is not None:
                embeddings = {record.text: record.embedding for record in pending}
                for record in records:
                    if record.embedding is None:
                        record.embedding = embeddings[record.text]

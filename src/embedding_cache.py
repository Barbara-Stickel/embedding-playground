"""Persistent, text-keyed vectors, isolated by model and encoding configuration."""

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from tqdm import tqdm

from embedding_core import (
    EMBEDDING_PIPELINE_VERSION,
    EmbeddingModelConfig,
    add_task_prefix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRECTORY = PROJECT_ROOT / "data" / "embeddings"


def model_identity(config: EmbeddingModelConfig) -> dict:
    identity = {
        "format_version": 1,
        "pipeline": EMBEDDING_PIPELINE_VERSION,
        "model": config.path,
        "revision": config.revision or "main",
        "prefix": add_task_prefix("", config.prefix),
        "trust_remote_code": config.trust_remote_code,
        "cache_version": config.cache_version,
    }
    local_path = Path(config.path).expanduser()
    if local_path.is_dir():
        identity["model"] = str(local_path.resolve())
        # Local fine-tuned checkpoints can change while keeping the same path.
        # Hash their contents to avoid reusing vectors for older weights/tokenizers.
        fingerprint = hashlib.sha256()
        for path in sorted(local_path.rglob("*")):
            if path.is_file():
                relative = path.relative_to(local_path).as_posix().encode()
                fingerprint.update(len(relative).to_bytes(8, "big"))
                fingerprint.update(relative)
                file_hash = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                fingerprint.update(file_hash.digest())
        identity["checkpoint_sha256"] = fingerprint.hexdigest()
    return identity


class EmbeddingCache:
    def __init__(self, config: EmbeddingModelConfig, directory: str | Path = DEFAULT_DIRECTORY):
        directory = Path(directory).expanduser()
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        identity = json.dumps(model_identity(config), sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", config.path.rstrip("/").split("/")[-1])[:60]
        self.path = directory / f"{name}-{digest[:16]}.sqlite3"
        self.identity = identity
        self.connection = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        try:
            with self.connection:
                self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (text_hash TEXT PRIMARY KEY, vector BLOB NOT NULL)")
                stored = self.connection.execute("SELECT value FROM metadata WHERE key = 'identity'").fetchone()
                if stored is not None and stored[0] != self.identity:
                    raise ValueError(f"Embedding cache model identity does not match: {self.path}")
                self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (self.identity,))
        except BaseException:
            self.connection.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()

    @staticmethod
    def text_key(text: str, prefix: str | None) -> str:
        return hashlib.sha256(add_task_prefix(text, prefix).encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        embeddings = {}
        # Stay below SQLite's parameter limit and avoid one query per record.
        with tqdm(total=len(keys), desc="Loading cached embeddings", unit="text", dynamic_ncols=True) as progress:
            for start in range(0, len(keys), 900):
                batch = keys[start:start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"SELECT text_hash, vector FROM embeddings WHERE text_hash IN ({placeholders})",
                    batch,
                )
                for key, blob in rows:
                    embeddings[key] = np.load(io.BytesIO(blob), allow_pickle=False)
                progress.update(len(batch))
        return embeddings

    def put_many(self, embeddings: Mapping[str, np.ndarray]) -> None:
        rows = []
        for key, vector in embeddings.items():
            if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.floating) or not np.isfinite(vector).all():
                raise ValueError("Cached embeddings must be finite, one-dimensional floating-point vectors")
            output = io.BytesIO()
            np.save(output, vector, allow_pickle=False)
            rows.append((key, output.getvalue()))
        # A transaction per batch preserves completed work across interruption.
        with self.connection:
            self.connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", rows)

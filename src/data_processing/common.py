from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def huggingface_file(repo_id: str, filename: str) -> Path:
    """Download a data file once and reuse the Hugging Face cache."""
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def resolve_source(source: str | Path) -> str | Path:
    """Resolve local files against the project root; leave URLs intact."""
    if str(source).startswith(("https://", "http://")):
        return str(source)
    path = Path(source)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")


def validate_text_labels(df: pd.DataFrame) -> None:
    require_columns(df, ["text", "label"])
    for column in ("text", "label"):
        valid = df[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not valid.all():
            raise ValueError(f"Dataset '{column}' values must be non-empty strings")


def sample_rows(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return df
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Sample limits must be positive integers or null")
    return df.sample(n=limit, random_state=seed) if len(df) > limit else df


def split_rows(df: pd.DataFrame, mode: str, test_size: float = 0.2, seed: int = 42, stratify=None):
    if mode == "timeseries":
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        index = int(len(df) * (1 - test_size))
        train, test = df.iloc[:index].copy(), df.iloc[index:].copy()
    elif mode == "random":
        train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    else:
        raise ValueError(f"Unknown split mode '{mode}'. Available choices: random, timeseries")
    if train.empty or test.empty:
        raise ValueError("The dataset must have non-empty training and test splits")
    return train, test

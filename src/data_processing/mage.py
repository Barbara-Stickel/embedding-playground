import pandas as pd

from data_types import TextDataset, TextRecord

from .common import huggingface_file, require_columns, resolve_source, sample_rows, validate_text_labels

REPO_ID = "jiumei/MAGE"
DEFAULT_FILES = {
    "train": "train.csv",
    "validation": "valid.csv",
    "test": "test.csv",
}
LABELS = {0: "machine", 1: "human"}


def load_mage(config: dict) -> TextDataset:
    """Read MAGE CSVs without changing the supplied split assignments."""
    files = config.get("files", DEFAULT_FILES)
    if "train" not in files or "test" not in files:
        raise ValueError("MAGE files must include train and test")
    unknown = set(files) - set(DEFAULT_FILES)
    if unknown:
        raise ValueError(f"Unknown MAGE splits: {', '.join(sorted(unknown))}")
    splits = {}
    for split, source in files.items():
        source = resolve_source(source) if "files" in config else huggingface_file(REPO_ID, source)
        df = pd.read_csv(source)
        require_columns(df, ["text", "label", "src"])
        if not df["label"].isin(LABELS).all():
            raise ValueError("MAGE labels must be 0 (machine) or 1 (human)")
        df["label"] = df["label"].map(LABELS)
        validate_text_labels(df)
        df = sample_rows(df, config["max_records"][split], config["seed"])
        splits[split] = [
            TextRecord(text=row.text, label=row.label, metadata={"source": row.src})
            for row in df.itertuples()
        ]
    return TextDataset(**splits)

import hashlib
import math
import random

import pandas as pd

from data_types import TextDataset, TextRecord
from reporting import print_hc3_split_fallback, print_hc3_training_summary

from .common import huggingface_file, require_columns, resolve_source, sample_rows, split_rows

REPO_ID = "Hello-SimpleAI/HC3"
SUBSETS = {"all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"}


def create_records(questions: pd.DataFrame, subset: str) -> list[TextRecord]:
    records = []
    for row in questions.to_dict("records"):
        for field, label in (("human_answers", "human"), ("chatgpt_answers", "machine")):
            for answer_index, answer in enumerate(row[field]):
                # Empty answer slots carry no useful classification input.
                if not answer.strip():
                    continue
                records.append(TextRecord(
                    text=answer,
                    label=label,
                    metadata={
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "source": row.get("source", subset),
                        "answer_index": answer_index,
                    },
                ))
    return records


def balance_training_answers(records: list[TextRecord], seed: int) -> list[TextRecord]:
    """Keep equal numbers of human and machine answers within each question."""
    groups = {}
    for record in records:
        group = groups.setdefault(record.metadata["question_id"], {"human": [], "machine": []})
        group[record.label].append(record)
    rng = random.Random(seed)
    balanced = []
    for group in groups.values():
        count = min(len(group["human"]), len(group["machine"]))
        balanced.extend(rng.sample(group["human"], count))
        balanced.extend(rng.sample(group["machine"], count))
    return balanced


def split_questions(groups: pd.DataFrame, config: dict):
    """Preserve source proportions when the sample is large enough to do so."""
    test_size = config["test_size"]
    stratify = None
    if config["stratify_source"] and groups["source"].nunique() > 1:
        source_counts = groups["source"].value_counts()
        test_count = math.ceil(len(groups) * test_size)
        if source_counts.min() >= 2 and min(test_count, len(groups) - test_count) >= len(source_counts):
            stratify = groups["source"]
        else:
            print_hc3_split_fallback()
    return split_rows(groups, "random", test_size, config["seed"], stratify=stratify)


def load_hc3(config: dict) -> TextDataset:
    """Split questions first, then expose only answers as model input."""
    subset = config["subset"]
    if subset not in SUBSETS:
        raise ValueError(f"Unknown HC3 subset '{subset}'. Available choices: {', '.join(sorted(SUBSETS))}")
    source = resolve_source(config["file"]) if "file" in config else huggingface_file(REPO_ID, f"{subset}.jsonl")
    # The source's IDs are optional; group on question text so duplicate questions
    # remain together even when their IDs differ.
    df = pd.read_json(source, lines=True, dtype=False, convert_dates=False)
    require_columns(df, ["question", "human_answers", "chatgpt_answers"])
    if not df["question"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("HC3 questions must be non-empty strings")
    for field in ("human_answers", "chatgpt_answers"):
        valid = df[field].map(
            lambda answers: isinstance(answers, list) and all(isinstance(answer, str) for answer in answers)
        )
        if not valid.all():
            raise ValueError(f"HC3 '{field}' must contain lists of strings")
    df["question_id"] = df["question"].map(lambda question: question.strip())
    # Use a compact, stable group identifier in metadata.
    df["question_id"] = df["question_id"].map(lambda question: hashlib.sha256(question.encode()).hexdigest())
    if "source" not in df:
        df["source"] = subset
    df["source"] = df["source"].fillna(subset)
    groups = df[["question_id", "source"]].drop_duplicates("question_id")
    seed = config["seed"]
    groups = sample_rows(groups, config["max_questions"], seed)
    train_groups, test_groups = split_questions(groups, config)
    train = df[df["question_id"].isin(train_groups["question_id"])]
    test = df[df["question_id"].isin(test_groups["question_id"])]
    train_records = create_records(train, subset)
    if config["balance_train"]:
        balanced = balance_training_answers(train_records, seed)
        print_hc3_training_summary(len(balanced), len(train_records))
        train_records = balanced
    return TextDataset(train=train_records, test=create_records(test, subset))

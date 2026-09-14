import hashlib

import pandas as pd

from data_types import TextDataset, TextRecord

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


def load_hc3(config: dict) -> TextDataset:
    """Split questions first, then expose only answers as model input."""
    subset = config.get("subset", "all")
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
    groups = df[["question_id"]].drop_duplicates()
    seed = config.get("seed", 42)
    groups = sample_rows(groups, config.get("max_questions"), seed)
    train_groups, test_groups = split_rows(groups, "random", config.get("test_size", 0.2), seed)
    train = df[df["question_id"].isin(train_groups["question_id"])]
    test = df[df["question_id"].isin(test_groups["question_id"])]
    return TextDataset(train=create_records(train, subset), test=create_records(test, subset))

from collections import Counter
from statistics import fmean, stdev

from rich.console import Console
from rich.table import Table
from rich.text import Text

from data_types import TextDataset, TextRecord


def print_confusion_matrix(data: TextDataset, title: str = "Confusion matrix") -> None:
    """Count test predictions with actual labels as rows and predictions as columns."""
    if not data.test:
        return
    if any(record.predicted_label is None for record in data.test):
        raise ValueError("All test records must be classified before printing the confusion matrix")

    labels = sorted(
        {record.label for records in (data.train, data.validation, data.test) for record in records}
        | {record.predicted_label for record in data.test}
    )
    counts = Counter((record.label, record.predicted_label) for record in data.test)
    table = Table(
        title=title,
        caption="Rows: actual category; columns: predicted category. Green counts are correct predictions.",
    )
    table.add_column(r"Actual \ Predicted", style="cyan")
    for label in labels:
        table.add_column(Text(label), justify="right")
    for actual in labels:
        table.add_row(
            Text(actual),
            *[
                Text(str(counts[actual, predicted]), style="green" if actual == predicted else "")
                for predicted in labels
            ],
        )
    Console().print(table)


def mean_and_spread(values: list[float]) -> str:
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{fmean(values):.2%} ± {spread:.2%}"


def print_evaluation_summary(results: list[dict]) -> None:
    labels = sorted({label for result in results for label in result["recalls"]})
    table = Table(
        title="Results across seeds",
        caption="Mean ± sample standard deviation across runs. Splits can overlap; this is not a confidence interval.",
    )
    table.add_column("Seed")
    table.add_column("Accuracy", justify="right")
    table.add_column("Balanced accuracy", justify="right")
    for label in labels:
        table.add_column(Text(f"{label} recall"), justify="right")
    for result in results:
        recalls = result["recalls"]
        table.add_row(
            str(result["seed"]), f"{result['accuracy']:.2%}", f"{result['balanced_accuracy']:.2%}",
            *[f"{recalls[label]:.2%}" if label in recalls else "—" for label in labels],
        )
    table.add_row(
        "Mean ± SD",
        mean_and_spread([result["accuracy"] for result in results]),
        mean_and_spread([result["balanced_accuracy"] for result in results]),
        *[mean_and_spread([result["recalls"][label] for result in results if label in result["recalls"]])
          for label in labels],
        style="bold",
    )
    Console().print(table)


def print_incorrect_predictions(
    records: list[TextRecord], max_incorrect_predictions: int | None = None,
) -> None:
    total_incorrect = len(records)
    if max_incorrect_predictions is not None:
        records = records[:max_incorrect_predictions]
    if not records:
        return

    title = f"Incorrect predictions ({total_incorrect})"
    if len(records) < total_incorrect:
        title = f"Incorrect predictions (showing first {len(records)} of {total_incorrect})"
    table = Table(
        title=title,
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    show_dates = any(record.metadata.get("date") is not None for record in records)
    if show_dates:
        table.add_column("Date", style="dim", width=10)
    table.add_column("Text", max_width=60)
    table.add_column("Predicted", style="yellow", max_width=25)
    table.add_column("Actual", style="cyan", max_width=25)

    for index, record in enumerate(records, start=1):
        cells = [str(index)]
        if show_dates:
            date = record.metadata.get("date")
            cells.append(date.strftime("%m/%d/%Y") if date is not None else "—")
        table.add_row(
            *cells,
            record.text,
            record.predicted_label or "—",
            record.label,
        )

    Console().print(table)


def print_run_start(index: int, total: int, seed: int) -> None:
    if total > 1:
        print(f"Run {index}/{total} (seed {seed})")


def print_run_result(result: tuple[int, int, float], balanced_accuracy: float) -> None:
    correct, total, accuracy = result
    print(f"result: {accuracy:%} - {correct} / {total}")
    print(f"balanced accuracy: {balanced_accuracy:%}")


def print_embedding_cache(reused: int, total: int, pending: int, path) -> None:
    print(f"Embedding cache: {reused} / {total} unique texts reused; {pending} to compute.")
    print(f"Cache file: {path}")


def print_hc3_training_summary(selected: int, original: int) -> None:
    print(f"HC3 training: {selected // 2} human + {selected // 2} machine answers; "
          f"{original - selected} surplus answers unused.")


def print_hc3_split_fallback() -> None:
    print("HC3 sample is too small to preserve source proportions; using a random question split.")

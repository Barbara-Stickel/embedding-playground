from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.text import Text

from data_types import TextDataset, TextRecord


def print_confusion_matrix(data: TextDataset) -> None:
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
        title="Confusion matrix",
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


def print_incorrect_predictions(
    records: list[TextRecord], max_incorrect_predictions: int | None = None,
) -> None:
    if max_incorrect_predictions is not None and (
        isinstance(max_incorrect_predictions, bool)
        or not isinstance(max_incorrect_predictions, int)
        or max_incorrect_predictions < 0
    ):
        raise ValueError("max_incorrect_predictions must be a non-negative integer or null")
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


def evaluate(
    data: TextDataset, max_incorrect_predictions: int | None = None,
) -> tuple[int, int, float]:
    true_value = 0
    total_values = 0
    incorrect_predictions = []

    for record in data.test:
        if record.label == record.predicted_label:
            true_value += 1
        else:
            incorrect_predictions.append(record)

        total_values += 1

    print_incorrect_predictions(incorrect_predictions, max_incorrect_predictions)

    percentage_correct = true_value / total_values if total_values != 0 else 0

    return true_value, total_values, percentage_correct

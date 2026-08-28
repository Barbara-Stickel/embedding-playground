from rich.console import Console
from rich.table import Table

from data_types import ExpenseDataset, ExpenseRecord


def print_incorrect_predictions(records: list[ExpenseRecord]) -> None:
    if not records:
        return

    table = Table(
        title=f"Incorrect predictions ({len(records)})",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Text", max_width=60)
    table.add_column("Predicted", style="yellow", max_width=25)
    table.add_column("Actual", style="cyan", max_width=25)

    for index, record in enumerate(records, start=1):
        table.add_row(
            str(index),
            record.text,
            record.predicted_category or "—",
            record.category,
        )

    Console().print(table)


def evaluate(data: ExpenseDataset) -> tuple[int, int, float]:
    true_value = 0
    total_values = 0
    incorrect_predictions = []

    for record in data.test:
        if record.category == record.predicted_category:
            true_value += 1
        else:
            incorrect_predictions.append(record)

        total_values += 1

    print_incorrect_predictions(incorrect_predictions)

    percentage_correct = true_value / total_values if total_values != 0 else 0

    return true_value, total_values, percentage_correct

from collections import Counter

from data_types import TextDataset, TextRecord
from reporting import print_incorrect_predictions


def category_recalls(records: list[TextRecord]) -> dict[str, float]:
    totals = Counter(record.label for record in records)
    correct = Counter(record.label for record in records if record.label == record.predicted_label)
    return {label: correct[label] / totals[label] for label in sorted(totals)}


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

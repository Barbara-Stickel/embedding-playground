import unittest
from unittest.mock import patch

from data_types import TextDataset, TextRecord
from evaluation import category_recalls
from reporting import print_confusion_matrix, print_evaluation_summary


class ConfusionMatrixTests(unittest.TestCase):
    def test_majority_only_predictions_have_low_balanced_accuracy(self):
        records = [TextRecord(str(i), "human" if i < 97 else "machine", predicted_label="human") for i in range(100)]
        recalls = category_recalls(records)
        self.assertEqual(recalls, {"human": 1.0, "machine": 0.0})
        self.assertEqual(sum(recalls.values()) / len(recalls), 0.5)

    @patch("reporting.Console")
    def test_seed_summary_reports_mean_and_sample_standard_deviation(self, console):
        results = [
            {"seed": 42, "accuracy": 0.8, "balanced_accuracy": 0.5, "recalls": {"human": 1.0, "machine": 0.0}},
            {"seed": 43, "accuracy": 0.9, "balanced_accuracy": 0.75, "recalls": {"human": 0.5, "machine": 1.0}},
        ]
        print_evaluation_summary(results)
        table = console.return_value.print.call_args.args[0]
        self.assertEqual(table.row_count, 3)
        self.assertEqual(table.columns[1]._cells[-1], "85.00% ± 7.07%")
        self.assertEqual(table.columns[2]._cells[-1], "62.50% ± 17.68%")
        self.assertIn("not a confidence interval", table.caption)

    @patch("reporting.Console")
    def test_binary_counts_use_actual_rows_and_predicted_columns(self, console):
        data = TextDataset(train=[], test=[
            TextRecord("correct human", "human", predicted_label="human"),
            TextRecord("wrong human", "human", predicted_label="machine"),
            TextRecord("another wrong human", "human", predicted_label="machine"),
            TextRecord("wrong machine", "machine", predicted_label="human"),
            TextRecord("correct machine", "machine", predicted_label="machine"),
        ])
        print_confusion_matrix(data)
        table = console.return_value.print.call_args.args[0]
        self.assertEqual([str(column.header) for column in table.columns],
                         [r"Actual \ Predicted", "human", "machine"])
        cells = [[cell.plain for cell in column._cells] for column in table.columns]
        self.assertEqual(cells, [["human", "machine"], ["1", "1"], ["2", "1"]])
        self.assertEqual(table.row_count, 2)

    @patch("reporting.Console")
    def test_multiple_categories_include_zero_counts_and_prediction_only_labels(self, console):
        data = TextDataset(
            train=[TextRecord("training only", "c")],
            test=[TextRecord("correct", "a", predicted_label="a"),
                  TextRecord("incorrect", "b", predicted_label="d")],
        )
        print_confusion_matrix(data)
        table = console.return_value.print.call_args.args[0]
        self.assertEqual([str(column.header) for column in table.columns][1:], ["a", "b", "c", "d"])
        cells = [[cell.plain for cell in column._cells] for column in table.columns]
        self.assertEqual(cells[1:], [["1", "0", "0", "0"], ["0", "0", "0", "0"],
                                   ["0", "0", "0", "0"], ["0", "1", "0", "0"]])
        self.assertEqual(table.row_count, 4)


if __name__ == "__main__":
    unittest.main()

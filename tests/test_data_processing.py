import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from classifier import classify
from data_processing import load_dataset as load_configured_dataset
from data_types import TextDataset, TextRecord
from evaluation import evaluate
from main import main


def dataset_config(options):
    name = options["name"]
    if name == "unknown":
        return options
    preset = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml").read_text())["dataset"]
    if "max_records" in options:
        options = {**options, "max_records": {**preset["max_records"], **options["max_records"]}}
    return {**preset, **options}


def load_dataset(options):
    return load_configured_dataset(dataset_config(options))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def expense_file(self):
        path = self.root / "expenses.csv"
        pd.DataFrame([
            {"Date": "05/01/2026", "Description": "Last Shop", "Debit": 5,
             "Type 1": "Food", "Details": ""},
            {"Date": "02/01/2026", "Description": "Cafe", "Debit": 2,
             "Type 1": "Food", "Details": "Coffee"},
            {"Date": "01/01/2026", "Description": "First Shop", "Debit": 1,
             "Type 1": "Food", "Details": ""},
            {"Date": "03/01/2026", "Description": "Books", "Debit": 3,
             "Type 1": "Shopping", "Details": "Books"},
            {"Date": "04/01/2026", "Description": "Bus", "Debit": 4,
             "Type 1": "Transport", "Details": ""},
            {"Date": "06/01/2026", "Description": "Other", "Debit": 6,
             "Type 1": "Misc", "Details": ""},
        ]).to_csv(path, index=False)
        return path

    def hc3_file(self, rows=None):
        if rows is None:
            rows = [
                {"id": index, "question": f"Question {index}?",
                 "human_answers": [f"Human Answer {index}", f"Second Human {index}"],
                 "chatgpt_answers": [f"Machine Answer {index}", "  "], "source": "fixture"}
                for index in range(10)
            ]
            # Different source IDs for the same question must not create separate groups.
            rows.append({"id": 99, "question": " Question 0? ",
                         "human_answers": ["Extra Human"], "chatgpt_answers": ["Extra Machine"],
                         "source": "fixture"})
        path = self.root / "hc3.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        return path

    def mage_files(self):
        files = {}
        for split in ("train", "validation", "test"):
            path = self.root / f"{split}.csv"
            pd.DataFrame([
                {"text": f"{split} Human {index}", "label": 1, "src": "human_source"}
                if index % 2 else
                {"text": f"{split} Machine {index}", "label": 0, "src": "model_source"}
                for index in range(10)
            ]).to_csv(path, index=False)
            files[split] = path
        return files

    def test_expenses_preserve_formatting_labels_dates_and_chronology(self):
        dataset = load_dataset({"name": "expenses", "file": self.expense_file(),
                                "category": "detailed", "split_data": "timeseries"})
        self.assertEqual([r.metadata["date"].day for r in dataset.train], [1, 2, 3, 4])
        self.assertEqual([r.metadata["date"].day for r in dataset.test], [5])
        self.assertEqual(dataset.train[1].text, "Merchant: cafe; Price: 2")
        self.assertEqual(dataset.train[1].label, "food - coffee")
        self.assertEqual(dataset.train[0].label, "food")
        self.assertEqual(dataset.validation, [])

    def test_expense_options_still_support_primary_merchant_only_and_misc(self):
        dataset = load_dataset({"name": "expenses", "file": self.expense_file(),
                                "input": "merchant_only", "remove_misc": False})
        records = dataset.train + dataset.test
        self.assertIn("misc", {r.label for r in records})
        self.assertIn("cafe", {r.text for r in records})
        self.assertTrue(all(" - " not in r.label for r in records))

    def test_mage_preserves_supplied_splits_and_normalizes_labels(self):
        dataset = load_dataset({"name": "mage", "files": self.mage_files()})
        for split in ("train", "validation", "test"):
            records = getattr(dataset, split)
            self.assertEqual(len(records), 10)
            self.assertTrue(all(r.text.startswith(split) for r in records))
            for record in records:
                self.assertEqual(record.label, "human" if "Human" in record.text else "machine")
                self.assertEqual(record.metadata["source"], "human_source" if record.label == "human" else "model_source")

    def test_mage_sampling_is_reproducible_and_does_not_mix_splits(self):
        config = {"name": "mage", "files": self.mage_files(),
                  "max_records": {"train": 4, "test": 3}, "seed": 7}
        first, second = load_dataset(config), load_dataset(config)
        self.assertEqual([r.text for r in first.train], [r.text for r in second.train])
        self.assertEqual([len(first.train), len(first.validation), len(first.test)], [4, 10, 3])
        self.assertFalse({r.text for r in first.train} & {r.text for r in first.test})

    def test_mage_rejects_unknown_labels(self):
        files = self.mage_files()
        pd.DataFrame([{"text": "Bad", "label": 2, "src": "source"}]).to_csv(files["train"], index=False)
        with self.assertRaisesRegex(ValueError, "MAGE labels"):
            load_dataset({"name": "mage", "files": files})

    @patch("data_processing.common.hf_hub_download")
    def test_default_mage_reads_data_files_from_the_requested_repository(self, download):
        files = self.mage_files()
        download.side_effect = lambda **options: files[{
            "train.csv": "train", "valid.csv": "validation", "test.csv": "test",
        }[options["filename"]]]
        dataset = load_dataset({"name": "mage"})
        self.assertEqual(len(dataset.validation), 10)
        self.assertEqual(download.call_count, 3)
        self.assertTrue(all(call.kwargs["repo_id"] == "jiumei/MAGE" and call.kwargs["repo_type"] == "dataset" for call in download.call_args_list))

    @patch("data_processing.common.hf_hub_download")
    def test_default_hc3_downloads_the_selected_answer_corpus(self, download):
        download.return_value = self.hc3_file()
        dataset = load_dataset({"name": "hc3", "subset": "finance", "balance_train": False})
        self.assertEqual(len(dataset.train) + len(dataset.test), 32)
        download.assert_called_once_with(repo_id="Hello-SimpleAI/HC3", filename="finance.jsonl", repo_type="dataset")

    def test_hc3_embeds_answers_and_keeps_duplicate_question_groups_together(self):
        dataset = load_dataset({"name": "hc3", "file": self.hc3_file(), "balance_train": False})
        train_ids = {r.metadata["question_id"] for r in dataset.train}
        test_ids = {r.metadata["question_id"] for r in dataset.test}
        self.assertEqual([len(train_ids), len(test_ids)], [8, 2])
        self.assertFalse(train_ids & test_ids)
        records = dataset.train + dataset.test
        self.assertEqual(len(records), 32)
        self.assertTrue(all("Question" not in r.text and r.text.strip() for r in records))
        self.assertEqual({r.label for r in records}, {"human", "machine"})
        self.assertIn("Human Answer 0", {r.text for r in records})
        for split in (dataset.train, dataset.test):
            for question_id in {r.metadata["question_id"] for r in split}:
                self.assertEqual({r.label for r in split if r.metadata["question_id"] == question_id}, {"human", "machine"})
        for extra in ("Extra Human", "Extra Machine"):
            containing_split = next(split for split in (dataset.train, dataset.test) if any(r.text == extra for r in split))
            self.assertIn("Human Answer 0", {r.text for r in containing_split})

    def test_hc3_sampling_keeps_all_answers_in_selected_question_groups(self):
        config = {"name": "hc3", "file": self.hc3_file(), "max_questions": 5, "seed": 7, "balance_train": False}
        first, second = load_dataset(config), load_dataset(config)
        self.assertEqual([r.text for r in first.train], [r.text for r in second.train])
        groups = {r.metadata["question_id"] for r in first.train + first.test}
        self.assertEqual(len(groups), 5)
        for question_id in groups:
            count = sum(r.metadata["question_id"] == question_id for r in first.train + first.test)
            self.assertIn(count, (3, 5))

    def test_hc3_domain_file_does_not_require_source_or_id_columns(self):
        rows = [{"question": f"Q {i}", "human_answers": [f"H {i}"],
                 "chatgpt_answers": [f"M {i}"]} for i in range(5)]
        dataset = load_dataset({"name": "hc3", "subset": "finance", "file": self.hc3_file(rows)})
        self.assertTrue(all(r.metadata["source"] == "finance" for r in dataset.train + dataset.test))

    def test_hc3_rejects_malformed_answer_lists(self):
        rows = [{"question": "Q", "human_answers": "A string", "chatgpt_answers": []}]
        with self.assertRaisesRegex(ValueError, "lists of strings"):
            load_dataset({"name": "hc3", "file": self.hc3_file(rows)})

    def test_hc3_rejects_a_single_question_instead_of_splitting_its_answers(self):
        rows = [{"question": "Q", "human_answers": ["H1", "H2"], "chatgpt_answers": ["M1", "M2"]}]
        with self.assertRaises(ValueError):
            load_dataset({"name": "hc3", "file": self.hc3_file(rows)})

    def test_invalid_dataset_and_split_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            load_dataset({"name": "unknown"})
        with self.assertRaisesRegex(ValueError, "Unknown split mode"):
            load_dataset({"name": "expenses", "file": self.expense_file(), "split_data": "typo"})

    def test_missing_source_columns_are_reported(self):
        path = self.root / "bad.csv"
        pd.DataFrame({"text": ["only text"]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Missing dataset columns"):
            load_dataset({"name": "mage", "files": {"train": path, "test": path}})

    def test_invalid_sample_limits_are_rejected(self):
        for limit in (0, -1, 0.5, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integers"):
                load_dataset({"name": "hc3", "file": self.hc3_file(), "max_questions": limit})

    @patch("main.embed")
    def test_cli_runs_hc3_with_answer_text_and_generic_classification(self, embed):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "dataset": dataset_config({"name": "hc3", "file": str(self.hc3_file())}),
            "evaluation": {"seeds": [42], "max_incorrect_predictions": 10},
            "embedding_cache": {"enabled": False, "directory": "data/embeddings"},
            "model": "mini",
            "classifier": {"name": "nearest_neighbors", "options": {"n_neighbors": 1}},
        }))

        def fake_embed(dataset, model_name, cache_config=None):
            for record in dataset.train + dataset.test:
                self.assertNotIn("Question", record.text)
                record.embedding = np.array([1., 0.]) if record.label == "human" else np.array([0., 1.])

        embed.side_effect = fake_embed
        with patch("sys.argv", ["embedding-playground", "--config", str(config_path)]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main()
        self.assertIn("100.000000%", stdout.getvalue())
        embed.assert_called_once()

    @patch("main.evaluate", return_value=(1, 1, 1.0))
    @patch("main.classify")
    @patch("main.embed")
    @patch("main.load_dataset", return_value=TextDataset(train=[], test=[]))
    def test_cli_uses_explicit_expense_configuration(self, load, embed, classify_mock, evaluate_mock):
        config_path = self.root / "expenses.yaml"
        config_path.write_text(yaml.safe_dump({
            "dataset": dataset_config({"name": "expenses", "input": "merchant_only",
                                       "category": "detailed", "remove_misc": False,
                                       "split_data": "timeseries"}),
            "evaluation": {"seeds": [42], "max_incorrect_predictions": None},
            "embedding_cache": {"enabled": False, "directory": "data/embeddings"},
            "model": "mini", "classifier": {"name": "nearest_neighbors"},
        }))
        with patch("sys.argv", ["embedding-playground", "--config", str(config_path)]), patch("sys.stdout", new_callable=io.StringIO):
            main()
        expected = yaml.safe_load(config_path.read_text())["dataset"]
        load.assert_called_once_with(expected)

    @patch("reporting.Console")
    def test_classification_and_evaluation_work_without_expense_dates(self, console):
        dataset = TextDataset(
            train=[TextRecord("human text", "human", embedding=np.array([1., 0.])),
                   TextRecord("machine text", "machine", embedding=np.array([0., 1.]))],
            test=[TextRecord("new human", "human", embedding=np.array([1., 0.])),
                  TextRecord("wrong human", "human", embedding=np.array([0., 1.]))],
        )
        classify(dataset, {"name": "nearest_neighbors", "options": {"n_neighbors": 1}})
        self.assertEqual(evaluate(dataset), (1, 2, 0.5))
        table = console.return_value.print.call_args.args[0]
        self.assertNotIn("Date", [column.header for column in table.columns])

    @patch("reporting.Console")
    def test_limiting_printed_mistakes_does_not_change_evaluation_totals(self, console):
        dataset = TextDataset(train=[], test=[
            TextRecord("correct", "human", predicted_label="human"),
            *[TextRecord(f"mistake {i}", "human", predicted_label="machine") for i in range(4)],
        ])
        self.assertEqual(evaluate(dataset, max_incorrect_predictions=2), (1, 5, 0.2))
        table = console.return_value.print.call_args.args[0]
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.title, "Incorrect predictions (showing first 2 of 4)")

    @patch("reporting.Console")
    def test_hiding_mistakes_still_evaluates_every_record(self, console):
        dataset = TextDataset(train=[], test=[TextRecord("mistake", "human", predicted_label="machine")])
        self.assertEqual(evaluate(dataset, max_incorrect_predictions=0), (0, 1, 0.0))
        console.assert_not_called()

    @patch("reporting.Console")
    def test_evaluation_keeps_dates_when_available(self, console):
        dataset = TextDataset(train=[], test=[TextRecord("expense", "food", metadata={"date": datetime.date(2026, 1, 1)})])
        self.assertEqual(evaluate(dataset), (0, 1, 0.0))
        table = console.return_value.print.call_args.args[0]
        self.assertIn("Date", [column.header for column in table.columns])


if __name__ == "__main__":
    unittest.main()

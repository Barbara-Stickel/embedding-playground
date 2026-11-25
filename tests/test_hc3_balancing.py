from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from data_processing import load_dataset
from data_processing.hc3 import balance_training_answers
from main import main


class HC3BalancingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rows = []
        for index in range(20):
            rows.append({
                "question": f"Question {index}", "source": "A" if index < 10 else "B",
                "human_answers": [f"Human {index}/{j}" for j in range(3 if index % 2 == 0 else 1)],
                "chatgpt_answers": [f"Machine {index}/{j}" for j in range(1 if index % 2 == 0 else 3)] + [" "],
            })
        self.rows = rows
        self.path = self.root / "hc3.jsonl"
        self.write_rows(rows)
        preset = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/hc3.yaml").read_text())
        self.config = {**preset["dataset"], "file": str(self.path), "seed": 42}

    def write_rows(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows))

    def test_training_is_balanced_per_question_and_source_with_test_answers_preserved(self):
        original = load_dataset({**self.config, "balance_train": False})
        balanced = load_dataset(self.config)
        self.assertEqual(Counter(r.label for r in balanced.train), {"human": 16, "machine": 16})
        for attribute in ("question_id", "source"):
            groups = defaultdict(Counter)
            for record in balanced.train:
                groups[record.metadata[attribute]][record.label] += 1
            self.assertTrue(all(counts["human"] == counts["machine"] for counts in groups.values()))
        self.assertEqual([r.text for r in balanced.test], [r.text for r in original.test])
        train_ids = {r.metadata["question_id"] for r in balanced.train}
        test_ids = {r.metadata["question_id"] for r in balanced.test}
        self.assertFalse(train_ids & test_ids)
        kept = {r.text for r in balanced.train}
        discarded = {r.text for r in original.train} - kept
        self.assertEqual(len(discarded), 32)
        self.assertFalse(discarded & {r.text for r in balanced.test})

    def test_source_proportions_are_preserved_in_both_splits(self):
        dataset = load_dataset(self.config)
        for records, expected in ((dataset.train, {"A": 8, "B": 8}), (dataset.test, {"A": 2, "B": 2})):
            questions = {(r.metadata["question_id"], r.metadata["source"]) for r in records}
            self.assertEqual(Counter(source for question_id, source in questions), expected)

    def test_answer_sampling_is_reproducible_and_changes_with_seed(self):
        original = load_dataset({**self.config, "balance_train": False})
        first = balance_training_answers(original.train, 42)
        again = balance_training_answers(original.train, 42)
        self.assertEqual([r.text for r in first], [r.text for r in again])
        choices = {tuple(sorted(r.text for r in balance_training_answers(original.train, seed))) for seed in range(5)}
        self.assertGreater(len(choices), 1)

    def test_duplicate_question_rows_are_balanced_as_one_group(self):
        rows = [{"question": f"Q {i}", "human_answers": [f"H {i}"],
                 "chatgpt_answers": [f"M {i}"]} for i in range(10)]
        rows[0]["chatgpt_answers"] = ["M 0", "M 0 extra 1", "M 0 extra 2"]
        rows.append({"question": " Q 0 ", "human_answers": ["H extra 1", "H extra 2", "H extra 3"],
                     "chatgpt_answers": ["M extra"]})
        self.write_rows(rows)
        dataset = load_dataset(self.config)
        duplicate_group = [r for r in dataset.train if r.metadata["question"].strip() == "Q 0"]
        self.assertEqual(Counter(r.label for r in duplicate_group), {"human": 4, "machine": 4})
        self.assertFalse(any(r.metadata["question"].strip() == "Q 0" for r in dataset.test))

    def test_unpaired_training_answers_are_unused_and_do_not_move_to_testing(self):
        # With one source and seed 42, question 0 is in training.
        rows = [{"question": f"Q {i}", "human_answers": [f"H {i}"],
                 "chatgpt_answers": [f"M {i}"] if i else []} for i in range(10)]
        self.write_rows(rows)
        dataset = load_dataset(self.config)
        self.assertNotIn("H 0", {r.text for r in dataset.train + dataset.test})
        self.assertEqual(Counter(r.label for r in dataset.train), {"human": 7, "machine": 7})

    @patch("main.embed")
    def test_multiple_seeds_run_the_pipeline_and_report_spread_with_final_run_matrix(self, embed):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump({
            "dataset": self.config, "evaluation": {"seeds": [42, 43], "max_incorrect_predictions": 0},
            "embedding_cache": {"enabled": False, "directory": "data/embeddings"},
            "model": "mini", "classifier": {"name": "nearest_neighbors", "options": {"n_neighbors": 1}},
        }))

        def fake_embed(dataset, **options):
            counts = Counter(r.label for r in dataset.train)
            self.assertEqual(counts["human"], counts["machine"])
            for record in dataset.train + dataset.test:
                record.embedding = np.array([1., 0.]) if record.label == "human" else np.array([0., 1.])

        embed.side_effect = fake_embed
        with patch("sys.argv", ["embedding-playground", "--config", str(path)]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main()
        output = " ".join(stdout.getvalue().split())
        self.assertEqual(embed.call_count, 2)
        self.assertIn("Run 1/2 (seed 42)", output)
        self.assertIn("Run 2/2 (seed 43)", output)
        self.assertIn("Results across seeds", output)
        self.assertIn("Mean ± SD", output)
        self.assertIn("Confusion matrix (final run, seed 43)", output)
        self.assertGreater(output.index("Confusion matrix"), output.index("Results across seeds"))


if __name__ == "__main__":
    unittest.main()

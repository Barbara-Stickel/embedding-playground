import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from data_types import TextDataset, TextRecord
from embedding_cache import EmbeddingCache
from embedding_core import add_task_prefix, create_model_config
from embedding_eval import embed


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = {"directory": str(self.root / "cache")}
        loader = patch("embedding_eval.load_model", return_value=(None, Mock(), torch.device("cpu")))
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        encoder = patch("embedding_eval.encode_texts", side_effect=self.encode)
        self.encoder = encoder.start()
        self.addCleanup(encoder.stop)

    @staticmethod
    def encode(texts, *args, **options):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(add_task_prefix(text, options.get("task_prefix")).encode()).digest()
            vectors.append([value / 255 for value in digest[:3]])
        return torch.tensor(vectors, dtype=torch.float32)

    @staticmethod
    def dataset(train, test):
        return TextDataset(
            train=[TextRecord(text, "human") for text in train],
            test=[TextRecord(text, "machine") for text in test],
        )

    def test_vectors_follow_text_across_splits_labels_and_record_order_without_loading_model(self):
        first = self.dataset(["alpha", "beta"], ["gamma"])
        embed(first, {"name": "mini", "batch_size": 2}, self.settings)
        expected = {r.text: r.embedding.copy() for r in first.train + first.test}
        self.loader.reset_mock()
        self.encoder.reset_mock()
        second = self.dataset(["gamma"], ["beta", "alpha"])
        # The explicit model ID is equivalent to its alias, and batching/device
        # choices do not change the reusable input/model identity.
        embed(second, {"path": "sentence-transformers/all-MiniLM-L6-v2",
                       "batch_size": 16, "device": "cpu"}, self.settings)
        self.loader.assert_not_called()
        self.encoder.assert_not_called()
        for record in second.train + second.test:
            np.testing.assert_array_equal(record.embedding, expected[record.text])

    def test_duplicate_texts_are_encoded_once_and_only_new_texts_are_added(self):
        first = self.dataset(["shared", "alpha"], ["shared", "gamma"])
        embed(first, "mini", self.settings)
        self.assertEqual(self.encoder.call_args.args[0], ["shared", "alpha", "gamma"])
        np.testing.assert_array_equal(first.train[0].embedding, first.test[0].embedding)
        self.encoder.reset_mock()
        second = self.dataset(["gamma", "alpha"], ["new text", "shared"])
        embed(second, "mini", self.settings)
        self.assertEqual(self.encoder.call_args.args[0], ["new text"])
        self.assertTrue(all(r.embedding is not None for r in second.train + second.test))

    def test_models_prefixes_revisions_and_versions_use_separate_caches(self):
        configurations = [
            {"name": "mini"},
            {"name": "nomic"},
            {"name": "mini", "prefix": "classification"},
            {"name": "mini", "revision": "pinned-commit"},
            {"name": "mini", "cache_version": "fine-tuned-v2"},
        ]
        for config in configurations:
            with self.subTest(config=config):
                self.loader.reset_mock()
                self.encoder.reset_mock()
                embed(self.dataset(["alpha"], ["beta"]), config, self.settings)
                self.loader.assert_called_once()
                self.encoder.assert_called_once()
        self.assertEqual(len(list((self.root / "cache").glob("*.sqlite3"))), len(configurations))

    def test_equivalent_prefix_spelling_reuses_vectors(self):
        embed(self.dataset(["alpha"], ["beta"]), {"name": "mini", "prefix": "classification"}, self.settings)
        self.loader.reset_mock()
        embed(self.dataset(["alpha"], ["beta"]), {"name": "mini", "prefix": " classification: "}, self.settings)
        self.loader.assert_not_called()

    def test_modified_local_checkpoint_does_not_reuse_older_embeddings(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        weights = checkpoint / "model.safetensors"
        weights.write_bytes(b"old weights")
        config = {"path": str(checkpoint)}
        embed(self.dataset(["alpha"], ["beta"]), config, self.settings)
        self.loader.reset_mock()
        weights.write_bytes(b"new weights")
        embed(self.dataset(["alpha"], ["beta"]), config, self.settings)
        self.loader.assert_called_once()
        self.assertEqual(len(list((self.root / "cache").glob("*.sqlite3"))), 2)

    def test_interrupted_run_keeps_completed_batches_for_the_next_run(self):
        data = self.dataset(["alpha", "beta"], ["gamma"])
        config = {"name": "mini", "batch_size": 2}
        self.encoder.side_effect = [self.encode(["alpha", "beta"]), RuntimeError("interrupted")]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            embed(data, config, self.settings)
        self.encoder.reset_mock()
        self.encoder.side_effect = self.encode
        resumed = self.dataset(["gamma"], ["beta", "alpha"])
        embed(resumed, config, self.settings)
        self.assertEqual(self.encoder.call_args.args[0], ["gamma"])
        self.assertTrue(all(r.embedding is not None for r in resumed.train + resumed.test))

    def test_disabling_cache_recomputes_without_writing_files(self):
        settings = {**self.settings, "enabled": False}
        for _ in range(2):
            embed(self.dataset(["alpha"], ["beta"]), "mini", settings)
        self.assertEqual(self.loader.call_count, 2)
        self.assertFalse((self.root / "cache").exists())

    def test_persistent_vectors_preserve_values_shape_and_dtype(self):
        vectors = {"single": np.array([1.25, 2.5], dtype=np.float32),
                   "double": np.array([3.25, 4.5], dtype=np.float64)}
        config = create_model_config("mini")
        with EmbeddingCache(config, self.settings["directory"]) as cache:
            cache.put_many(vectors)
        with EmbeddingCache(config, self.settings["directory"]) as cache:
            restored = cache.get_many(["double", "missing", "single"])
        self.assertEqual(set(restored), set(vectors))
        for key, vector in vectors.items():
            np.testing.assert_array_equal(restored[key], vector)
            self.assertEqual(restored[key].dtype, vector.dtype)


if __name__ == "__main__":
    unittest.main()

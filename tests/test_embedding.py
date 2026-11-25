import unittest
from unittest.mock import patch

import torch

from data_types import TextDataset, TextRecord
from embedding_core import (
    add_task_prefix,
    create_device,
    create_model_config,
)
from embedding_eval import (
    embed,
    embed_records,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sentences, **options):
        self.calls.append((sentences, options))
        return {
            "input_ids": torch.ones((len(sentences), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(sentences), 2), dtype=torch.long),
        }


class FakeModel:
    def __init__(self):
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, input_ids, attention_mask):
        embeddings = torch.stack((input_ids.float(), input_ids.float() * 2), dim=2)
        return (embeddings,)


def record(text):
    return TextRecord(text, "label")


class EmbeddingTests(unittest.TestCase):
    def test_task_prefix_is_formatted_once(self):
        self.assertEqual(
            add_task_prefix("coffee shop", "classification"),
            "classification: coffee shop",
        )
        self.assertEqual(
            add_task_prefix("coffee shop", "classification:"),
            "classification: coffee shop",
        )
        self.assertEqual(add_task_prefix("coffee shop", None), "coffee shop")
        self.assertEqual(add_task_prefix("coffee shop", "  "), "coffee shop")

    def test_original_compact_model_config_still_works(self):
        config = create_model_config("mini")

        self.assertEqual(config.path, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsNone(config.prefix)

    def test_model_config_reads_options(self):
        config = create_model_config({
            "path": "organization/model",
            "batch_size": 16,
            "prefix": "classification",
            "trust_remote_code": True,
        })

        self.assertEqual(config.path, "organization/model")
        self.assertEqual(config.prefix, "classification")
        self.assertEqual(config.batch_size, 16)
        self.assertTrue(config.trust_remote_code)

    @patch("embedding_core.torch.backends.mps.is_available", return_value=True)
    def test_auto_device_uses_mps_when_available(self, _is_available):
        self.assertEqual(create_device("auto"), torch.device("mps"))

    def test_embed_records_adds_prefixes_and_batches(self):
        records = [record("one"), record("two"), record("three")]
        tokenizer = FakeTokenizer()

        embed_records(
            records,
            tokenizer,
            FakeModel(),
            task_prefix="classification",
            batch_size=2,
        )

        self.assertEqual(len(tokenizer.calls), 2)
        self.assertEqual(
            tokenizer.calls[0][0],
            ["classification: one", "classification: two"],
        )
        self.assertEqual(tokenizer.calls[1][0], ["classification: three"])
        self.assertTrue(all(record.embedding is not None for record in records))

    @patch("embedding_core.AutoModel.from_pretrained")
    @patch("embedding_core.AutoTokenizer.from_pretrained")
    def test_embed_uses_configured_prefix(self, tokenizer_loader, model_loader):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        tokenizer_loader.return_value = tokenizer
        model_loader.return_value = model
        data = TextDataset(train=[record("known")], test=[record("new")])

        embed(data, {
            "name": "nomic",
            "prefix": "classification",
            "trust_remote_code": True,
            "device": "cpu",
        }, cache_config={"enabled": False})

        tokenizer_loader.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        model_loader.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        self.assertTrue(model.eval_called)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(tokenizer.calls[0][0], ["classification: known", "classification: new"])


if __name__ == "__main__":
    unittest.main()

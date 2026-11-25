import unittest

import torch

from embedding_training import loss_function, train


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, sentences, **options):
        self.calls.append((sentences, options))
        token_ids = [1 if "coffee" in sentence else 2 for sentence in sentences]
        return {
            "input_ids": torch.tensor(token_ids).reshape(-1, 1),
            "attention_mask": torch.ones((len(sentences), 1), dtype=torch.long),
        }


class TrainableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(3, 2)

    def forward(self, input_ids, attention_mask):
        return (self.embeddings(input_ids),)


class EmbeddingTrainingTests(unittest.TestCase):
    def test_loss_uses_the_similarity_device_and_dtype(self):
        similarity = torch.tensor([0.25, 0.75], dtype=torch.float64)
        loss = loss_function(similarity, [0.0, 1.0])

        self.assertEqual(loss.dtype, torch.float64)
        self.assertAlmostEqual(loss.item(), 0.0625)

    def test_train_updates_the_model_and_returns_loss(self):
        torch.manual_seed(1)
        tokenizer = FakeTokenizer()
        model = TrainableModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_weights = model.embeddings.weight.detach().clone()

        loss = train(
            ["coffee shop", "coffee beans"],
            ["book store", "book store"],
            [0.0, 1.0],
            tokenizer,
            model,
            optimizer,
            task_prefix="classification",
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(model.training)
        self.assertFalse(torch.equal(original_weights, model.embeddings.weight))
        self.assertEqual(
            tokenizer.calls[0][0],
            ["classification: coffee shop", "classification: coffee beans"],
        )

    def test_train_rejects_mismatched_batches(self):
        model = TrainableModel()

        with self.assertRaisesRegex(ValueError, "same number"):
            train(
                ["one", "two"],
                ["one"],
                [1.0, 1.0],
                FakeTokenizer(),
                model,
                torch.optim.SGD(model.parameters(), lr=0.1),
            )


if __name__ == "__main__":
    unittest.main()

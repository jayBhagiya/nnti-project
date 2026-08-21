import math
import unittest
from types import SimpleNamespace

import torch

from scripts.lm_eval import causal_labels, evaluate_causal_lm, select_indices


class FakeTokenizer:
    bos_token_id = 10
    eos_token_id = None
    pad_token_id = 0
    token_ids = {"a": [2, 3], "é": [4], "long": [2, 3, 4, 5, 6]}

    def __call__(self, sentences, **_kwargs):
        return {"input_ids": [self.token_ids[sentence] for sentence in sentences]}

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.bos_token_id, *token_ids]

    def pad(self, encoded, **_kwargs):
        sequences = encoded["input_ids"]
        width = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            padding = width - len(sequence)
            input_ids.append(sequence + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(sequence) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class FakeModel:
    def __init__(self):
        self.received_attention_mask = False

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, labels):
        del input_ids
        self.received_attention_mask = attention_mask is not None
        targets = labels[:, 1:]
        return SimpleNamespace(loss=targets[targets != -100].float().mean())


class LanguageModelEvaluationTest(unittest.TestCase):
    def test_metrics_are_token_weighted_and_batch_invariant(self):
        sentences = ["a", "é"]
        batched_model = FakeModel()
        batched = evaluate_causal_lm(
            batched_model,
            FakeTokenizer(),
            sentences,
            batch_size=2,
            device="cpu",
        )
        unbatched = evaluate_causal_lm(
            FakeModel(),
            FakeTokenizer(),
            sentences,
            batch_size=1,
            device="cpu",
        )

        self.assertTrue(batched_model.received_attention_mask)
        self.assertEqual(batched["tokens"], 3)
        self.assertEqual(batched["predicted_tokens"], 3)
        self.assertEqual(batched["utf8_bytes"], 3)
        self.assertAlmostEqual(batched["nll"], 3.0)
        self.assertAlmostEqual(batched["bits_per_byte"], 3 / math.log(2))
        self.assertAlmostEqual(batched["nll"], unbatched["nll"])
        indices = select_indices(10, 4, 7)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), 4)
        self.assertEqual(indices, select_indices(10, 4, 7))
        labels = causal_labels(
            torch.tensor([[0, 10, 2]]),
            torch.tensor([[0, 1, 1]]),
        )
        self.assertEqual(labels.tolist(), [[-100, -100, 2]])

        windowed = evaluate_causal_lm(
            FakeModel(),
            FakeTokenizer(),
            ["long"],
            batch_size=1,
            device="cpu",
            max_length=3,
        )
        self.assertEqual(windowed["context_windows"], 5)
        self.assertEqual(windowed["predicted_tokens"], 5)
        self.assertAlmostEqual(windowed["nll"], 4.0)


if __name__ == "__main__":
    unittest.main()

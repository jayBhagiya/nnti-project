import unittest
from unittest.mock import patch

from scripts.task3_data_preparation import prepare_adaptation_data


class Task3DataTest(unittest.TestCase):
    def test_adaptation_splits_are_disjoint_with_test_priority(self):
        dataset = {
            "train": {"qu": ["train", "shared-train-val", "shared-all"]},
            "validation": {"qu": ["validation", "shared-train-val", "shared-all"]},
            "test": {"qu": ["test", "shared-all"]},
        }
        with patch(
            "scripts.task3_data_preparation.datasets.load_dataset",
            return_value=dataset,
        ):
            prepared = prepare_adaptation_data(
                "test/dataset",
                data_files=None,
                seed=42,
                train_samples=None,
            )

        self.assertEqual(prepared["test"], ["test", "shared-all"])
        self.assertEqual(
            set(prepared["validation"]), {"validation", "shared-train-val"}
        )
        self.assertEqual(set(prepared["train"]), {"train"})
        self.assertFalse(set(prepared["train"]) & set(prepared["validation"]))
        self.assertFalse(set(prepared["train"]) & set(prepared["test"]))
        self.assertFalse(set(prepared["validation"]) & set(prepared["test"]))


if __name__ == "__main__":
    unittest.main()

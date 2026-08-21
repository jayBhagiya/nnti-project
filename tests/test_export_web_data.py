import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.export_web_data import (
    TASK3_CAMPAIGN_CONFIGS,
    _validate_task3_campaign,
    export_task2,
    export_task3,
)


class WebDataExportTest(unittest.TestCase):
    def test_campaign_gate_groups_configurations_by_seed(self):
        complete = [(42, method, rank) for method, rank in TASK3_CAMPAIGN_CONFIGS]
        _validate_task3_campaign(complete)
        mixed = complete[:-1] + [(43, *complete[-1][1:])]
        with self.assertRaisesRegex(ValueError, "by seed"):
            _validate_task3_campaign(mixed)

    @staticmethod
    def _task2_file(path, method, offset):
        languages = ["eng_Latn", "quy_Latn"]
        with h5py.File(path, "w") as output:
            output.attrs.update(
                {
                    "schema_version": 2,
                    "complete": True,
                    "model_name": "test/xglm",
                    "model_revision": "revision",
                    "adaptation_method": method,
                    "lora_rank": 0,
                    "dataset_name": "test/flores",
                    "dataset_revision": "dataset-revision",
                    "split": "dev",
                    "num_layers": 3,
                    "hidden_size": 5,
                    "language_mapping": json.dumps(dict(enumerate(languages))),
                }
            )
            for language in languages:
                output.attrs[f"dataset_fingerprint_{language}"] = "fingerprint"
            output.create_dataset("sample_indices", data=np.arange(4))
            output.create_dataset(
                "language_codes",
                data=np.asarray(languages, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            for name in (
                "sentence_embeddings",
                "token_embeddings",
                "sequences",
                "tokens",
                "sample_ids",
                "token_ids",
                "token_sentence_ids",
                "token_sentence_indices",
                "sequence_lengths",
            ):
                output.create_group(name)

            generator = np.random.default_rng(10 + offset)
            for language_index, language in enumerate(languages):
                key = str(language_index)
                sentence_values = generator.normal(
                    loc=language_index + offset,
                    size=(4, 3, 5),
                )
                token_values = generator.normal(
                    loc=language_index + offset,
                    size=(8, 3, 5),
                )
                for name, values in (
                    ("sentence_embeddings", sentence_values),
                    ("token_embeddings", token_values),
                ):
                    dataset = output[name].create_dataset(key, data=values)
                    dataset.attrs["language"] = language
                output["sample_ids"].create_dataset(
                    key, data=np.arange(100, 104)
                )
                output["sequences"].create_dataset(
                    key,
                    data=np.asarray(
                        [f"sentence-{value}" for value in range(4)], dtype=object
                    ),
                    dtype=h5py.string_dtype("utf-8"),
                )
                output["tokens"].create_dataset(
                    key,
                    data=np.asarray(
                        [f"token-{value}" for value in range(8)], dtype=object
                    ),
                    dtype=h5py.string_dtype("utf-8"),
                )
                output["token_ids"].create_dataset(key, data=np.arange(8))
                output["token_sentence_ids"].create_dataset(
                    key, data=np.repeat(np.arange(100, 104), 2)
                )
                output["token_sentence_indices"].create_dataset(
                    key, data=np.repeat(np.arange(4), 2)
                )
                output["sequence_lengths"].create_dataset(
                    key, data=np.full(4, 2)
                )

    @staticmethod
    def _rows(path):
        with path.open(encoding="utf-8", newline="") as input_file:
            return list(csv.DictReader(input_file))

    def test_task2_and_task3_exports_are_complete_and_text_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.h5"
            full = root / "full.h5"
            self._task2_file(base, "base", 0)
            self._task2_file(full, "full", 1)
            projections = root / "task2_projections.csv"
            stats = root / "task2_layer_stats.csv"

            export_task2(
                [base, full],
                projections,
                stats,
                sentences_per_language=2,
                tokens_per_language=2,
                layers=[0, 2],
                tsne_layers=[2],
                seed=42,
            )
            projection_rows = self._rows(projections)
            self.assertEqual(len(projection_rows), 32)
            self.assertEqual(len(self._rows(stats)), 8)
            self.assertNotIn("token", projection_rows[0])
            self.assertNotIn("sentence", projection_rows[0])
            self.assertTrue(all(len(row["point_id"]) == 16 for row in projection_rows))
            self.assertTrue(
                all(row["tsne_x"] for row in projection_rows if row["layer"] == "2")
            )
            self.assertTrue(
                all(not row["tsne_x"] for row in projection_rows if row["layer"] == "0")
            )
            repeated_projections = root / "task2_projections_repeated.csv"
            repeated_stats = root / "task2_layer_stats_repeated.csv"
            export_task2(
                [base, full],
                repeated_projections,
                repeated_stats,
                sentences_per_language=2,
                tokens_per_language=2,
                layers=[0, 2],
                tsne_layers=[2],
                seed=42,
            )
            self.assertEqual(
                projections.read_bytes(), repeated_projections.read_bytes()
            )
            self.assertEqual(stats.read_bytes(), repeated_stats.read_bytes())
            with h5py.File(full, "r+") as output:
                output["sample_ids"]["0"][0] = 999
                output["token_sentence_ids"]["0"][:2] = 999
            with self.assertRaisesRegex(ValueError, "not aligned"):
                export_task2(
                    [base, full],
                    root / "misaligned-projections.csv",
                    root / "misaligned-stats.csv",
                    1,
                    1,
                    [0],
                    [],
                    42,
                )
            with self.assertRaises(FileExistsError):
                export_task2(
                    [base], projections, stats, 1, 1, [0], [], 42
                )

            run_dir = root / "lora-r4-s42"
            run_dir.mkdir()
            summary = {
                "method": "lora",
                "rank": 4,
                "seed": 42,
                "trainable_parameters": 10,
                "baseline_development": {
                    "adaptation_validation": {"nll": 3.0, "perplexity": 20.0}
                },
                "baseline_final": {
                    "adaptation_test": {"nll": 3.1, "perplexity": 22.0}
                },
                "final": {
                    "adaptation_test": {"nll": 2.0, "perplexity": 7.4}
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "git_commit": "abc123",
                        "model_commit": "model123",
                        "training_manifest_sha256": "manifest123",
                    }
                )
            )
            (run_dir / "history.jsonl").write_text(
                json.dumps({"stage": "train", "epoch": 1, "nll": 2.5}) + "\n"
            )
            summary_csv = root / "task3_summary.csv"
            history_csv = root / "task3_history.csv"
            export_task3(
                [run_dir / "summary.json"], summary_csv, history_csv
            )

            summary_rows = self._rows(summary_csv)
            self.assertEqual(len(summary_rows), 3)
            self.assertEqual({row["run_id"] for row in summary_rows}, {run_dir.name})
            self.assertEqual({row["rank"] for row in summary_rows}, {"4"})
            self.assertEqual({row["git_commit"] for row in summary_rows}, {"abc123"})
            history_rows = self._rows(history_csv)
            self.assertEqual(history_rows[0]["method"], "lora")
            self.assertEqual(history_rows[0]["seed"], "42")
            with self.assertRaisesRegex(ValueError, "Incomplete Task 3 campaign"):
                export_task3(
                    [run_dir / "summary.json"],
                    root / "gated-summary.csv",
                    root / "gated-history.csv",
                    require_complete_campaign=True,
                )


if __name__ == "__main__":
    unittest.main()

"""Dataset preparation shared by Task 3 training and evaluation."""

from __future__ import annotations

import random

import datasets


FLORES_DATASET = "facebook/flores"


def _clean_unique(texts: list[str]) -> list[str]:
    return list(dict.fromkeys(text.strip() for text in texts if text.strip()))


def _seeded_subset(texts: list[str], limit: int | None, seed: int) -> list[str]:
    texts = _clean_unique(texts)
    random.Random(seed).shuffle(texts)
    return texts if limit is None else texts[:limit]


def _disjoint_splits(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    """Remove exact leakage while preserving test, then validation, examples."""

    test = _clean_unique(splits["test"])
    test_set = set(test)
    validation = [
        text
        for text in _clean_unique(splits["validation"])
        if text not in test_set
    ]
    held_out = test_set | set(validation)
    train = [
        text
        for text in _clean_unique(splits["train"])
        if text not in held_out
    ]
    return {"train": train, "validation": validation, "test": test}


def prepare_adaptation_data(
    dataset_path: str,
    data_files: dict[str, str] | None,
    seed: int,
    train_samples: int | None,
    revision: str | None = None,
) -> dict[str, list[str]]:
    """Load deterministic, de-duplicated Quechua train/validation/test text."""

    dataset = datasets.load_dataset(
        dataset_path,
        data_files=data_files,
        revision=revision,
    )
    required = {"train", "validation", "test"}
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"Adaptation dataset is missing splits: {sorted(missing)}")

    prepared = _disjoint_splits(
        {split: dataset[split]["qu"] for split in sorted(required)}
    )
    prepared["train"] = _seeded_subset(
        prepared["train"],
        limit=train_samples,
        seed=seed,
    )
    return prepared


def prepare_flores_eval(
    languages: list[str],
    split: str = "dev",
    sample_size: int | None = None,
    seed: int = 0,
    revision: str | None = None,
) -> dict[str, list[str]]:
    """Load the same deterministic FLORES subset for each language."""

    result: dict[str, list[str]] = {}
    shared_ids: list[int] | None = None
    expected_size: int | None = None
    for language in languages:
        dataset = datasets.load_dataset(
            FLORES_DATASET,
            language,
            revision=revision,
            token=True,
            trust_remote_code=True,
        )[split]
        if shared_ids is None:
            expected_size = len(dataset)
            if sample_size is None or sample_size >= len(dataset):
                shared_ids = list(range(len(dataset)))
            else:
                shared_ids = sorted(
                    random.Random(seed).sample(range(len(dataset)), sample_size)
                )
        elif len(dataset) != expected_size:
            raise ValueError(f"FLORES split lengths differ for {language}")
        result[language] = [dataset[index]["sentence"] for index in shared_ids]
    return result


def prepare_data_fine_tune_eval(languages: list[str]) -> dict[str, list[str]]:
    """Compatibility wrapper for the original coursework entry point."""

    return prepare_flores_eval(languages)


def prepare_data_qu(
    dataset_path: str,
    data_files: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Compatibility wrapper returning the original train/validation pair."""

    prepared = prepare_adaptation_data(
        dataset_path,
        data_files=data_files,
        seed=0,
        train_samples=None,
    )
    return prepared["train"], prepared["validation"]

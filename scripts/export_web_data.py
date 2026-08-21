"""Export compact, text-free Task 2 and Task 3 data for the website."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable


TASK2_SCHEMA_VERSION = 2
TASK2_GROUPS = (
    "sentence_embeddings",
    "token_embeddings",
    "sequences",
    "tokens",
    "sample_ids",
    "token_ids",
    "token_sentence_ids",
    "token_sentence_indices",
    "sequence_lengths",
)
TASK3_STAGES = ("baseline_development", "baseline_final", "final")
TASK3_CAMPAIGN_CONFIGS = {
    ("full", ""),
    ("bitfit", ""),
    ("ia3", ""),
    ("lora", 1),
    ("lora", 2),
    ("lora", 4),
    ("lora", 8),
}
PUBLIC_CONFIG_FIELDS = (
    "git_commit",
    "model_name",
    "model_commit",
    "model_revision",
    "dataset",
    "dataset_revision",
    "flores_revision",
    "data_seed",
    "max_sequence_length",
    "training_manifest_sha256",
    "development_manifest_sha256",
    "final_manifest_sha256",
    "device",
    "torch_version",
    "cuda_version",
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _float(value: float | None) -> str:
    return "" if value is None else format(float(value), ".10g")


def _seed(seed: int, *parts: object) -> int:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _point_id(kind: str, language: str, *identity: object) -> str:
    value = ":".join((kind, language, *(str(item) for item in identity)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _sample_indices(total: int, limit: int, seed: int, *parts: object):
    import numpy as np

    if total < 1:
        name = "/".join(map(str, parts))
        raise ValueError(f"Cannot sample from an empty {name} dataset")
    count = min(total, limit)
    if count == total:
        return np.arange(total, dtype=np.int64)
    generator = np.random.default_rng(_seed(seed, *parts))
    return np.sort(generator.choice(total, size=count, replace=False))


def _check_outputs(
    paths: Iterable[Path],
    overwrite: bool,
    protected: Iterable[Path] = (),
) -> None:
    outputs = list(paths)
    normalized = [path.resolve() for path in outputs]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Output paths must be different")
    protected_paths = {path.resolve() for path in protected}
    if protected_paths.intersection(normalized):
        raise ValueError("An output path cannot replace an input artifact")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}; pass --overwrite")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_name = output.name
            writer = csv.DictWriter(
                output, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _task2_model_metadata(hdf5_file) -> dict[str, Any]:
    method = _text(hdf5_file.attrs.get("adaptation_method", "base"))
    rank = int(hdf5_file.attrs.get("lora_rank", 0))
    model_id = f"lora-r{rank}" if method == "lora" else method
    return {
        "model_id": model_id,
        "model_name": _text(hdf5_file.attrs["model_name"]),
        "model_revision": _text(hdf5_file.attrs.get("model_revision", "")),
        "method": method,
        "rank": rank if method == "lora" else "",
        "dataset": _text(hdf5_file.attrs.get("dataset_name", "")),
        "dataset_revision": _text(hdf5_file.attrs.get("dataset_revision", "")),
        "split": _text(hdf5_file.attrs.get("split", "")),
    }


def _validate_task2_file(path: Path, hdf5_file) -> tuple[dict[str, Any], list[str]]:
    import numpy as np

    if int(hdf5_file.attrs.get("schema_version", -1)) != TASK2_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected Task 2 schema version {TASK2_SCHEMA_VERSION}"
        )
    if not bool(hdf5_file.attrs.get("complete", False)):
        raise ValueError(f"{path}: extraction is not marked complete")
    for name in ("model_name", "num_layers", "hidden_size"):
        if name not in hdf5_file.attrs:
            raise ValueError(f"{path}: missing {name!r} metadata")
    if "language_codes" not in hdf5_file:
        raise ValueError(f"{path}: missing 'language_codes' dataset")
    if "sample_indices" not in hdf5_file:
        raise ValueError(f"{path}: missing 'sample_indices' dataset")
    for name in TASK2_GROUPS:
        if name not in hdf5_file:
            raise ValueError(f"{path}: missing {name!r} group")

    languages = [_text(value) for value in hdf5_file["language_codes"][:]]
    if not languages or len(languages) != len(set(languages)):
        raise ValueError(f"{path}: language codes must be non-empty and unique")
    try:
        mapping = json.loads(_text(hdf5_file.attrs["language_mapping"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid language_mapping metadata") from error
    if mapping != {str(index): language for index, language in enumerate(languages)}:
        raise ValueError(f"{path}: language_mapping does not match language_codes")

    num_layers = int(hdf5_file.attrs["num_layers"])
    hidden_size = int(hdf5_file.attrs["hidden_size"])
    sampled_count = len(hdf5_file["sample_indices"])
    if num_layers < 1 or hidden_size < 1:
        raise ValueError(f"{path}: embedding dimensions must be positive")
    for index, language in enumerate(languages):
        if f"dataset_fingerprint_{language}" not in hdf5_file.attrs:
            raise ValueError(f"{path}: missing dataset fingerprint for {language}")
        key = str(index)
        missing = [name for name in TASK2_GROUPS if key not in hdf5_file[name]]
        if missing:
            raise ValueError(f"{path}: {language} is missing groups {missing}")
        sentences = hdf5_file["sentence_embeddings"][key]
        tokens = hdf5_file["token_embeddings"][key]
        for name, dataset in (("sentence", sentences), ("token", tokens)):
            if dataset.ndim != 3 or dataset.shape[1:] != (num_layers, hidden_size):
                raise ValueError(
                    f"{path}: {language} {name} embeddings have shape "
                    f"{dataset.shape}, expected (*, {num_layers}, {hidden_size})"
                )
            if not np.issubdtype(dataset.dtype, np.number):
                raise ValueError(
                    f"{path}: {language} {name} embeddings are not numeric"
                )
            if _text(dataset.attrs.get("language", "")) != language:
                raise ValueError(f"{path}: {language} {name} metadata does not match")
        sentence_count = sentences.shape[0]
        token_count = tokens.shape[0]
        if sentence_count != sampled_count:
            raise ValueError(f"{path}: {language} sentence/sample-index counts differ")
        if sentence_count != len(hdf5_file["sample_ids"][key]):
            raise ValueError(f"{path}: {language} sentence/sample counts differ")
        if sentence_count != len(hdf5_file["sequences"][key]):
            raise ValueError(f"{path}: {language} sentence/text counts differ")
        if sentence_count != len(hdf5_file["sequence_lengths"][key]):
            raise ValueError(f"{path}: {language} sentence/length counts differ")
        for name in (
            "tokens",
            "token_ids",
            "token_sentence_ids",
            "token_sentence_indices",
        ):
            if token_count != len(hdf5_file[name][key]):
                raise ValueError(f"{path}: {language} token/{name} counts differ")
        lengths = hdf5_file["sequence_lengths"][key][:]
        if np.any(lengths < 1) or int(lengths.sum()) != token_count:
            raise ValueError(f"{path}: {language} sequence lengths are inconsistent")
        expected_indices = np.repeat(np.arange(sentence_count), lengths)
        if not np.array_equal(
            hdf5_file["token_sentence_indices"][key][:], expected_indices
        ):
            raise ValueError(
                f"{path}: {language} token/sentence indices are inconsistent"
            )
        expected_ids = np.repeat(hdf5_file["sample_ids"][key][:], lengths)
        if not np.array_equal(
            hdf5_file["token_sentence_ids"][key][:], expected_ids
        ):
            raise ValueError(f"{path}: {language} token/sentence IDs are inconsistent")

    return _task2_model_metadata(hdf5_file), languages


def _validate_task2_alignment(sources: list[dict[str, Any]]) -> None:
    import numpy as np

    reference = sources[0]
    reference_file = reference["file"]
    provenance = (
        "model_name",
        "model_revision",
        "dataset_name",
        "dataset_revision",
        "split",
        "seed",
        "num_samples",
    )
    aligned_groups = (
        "sequences",
        "tokens",
        "sample_ids",
        "token_ids",
        "token_sentence_ids",
        "token_sentence_indices",
        "sequence_lengths",
    )
    for source in sources[1:]:
        hdf5_file = source["file"]
        mismatched = [
            name
            for name in provenance
            if hdf5_file.attrs.get(name) != reference_file.attrs.get(name)
        ]
        if mismatched:
            raise ValueError(
                f"{source['path']}: Task 2 provenance differs for {mismatched}"
            )
        if not np.array_equal(
            hdf5_file["sample_indices"][:], reference_file["sample_indices"][:]
        ):
            raise ValueError(f"{source['path']}: sampled FLORES rows differ")
        for index, language in enumerate(reference["languages"]):
            fingerprint = f"dataset_fingerprint_{language}"
            if hdf5_file.attrs.get(fingerprint) != reference_file.attrs.get(
                fingerprint
            ):
                raise ValueError(
                    f"{source['path']}: dataset fingerprint differs for {language}"
                )
            key = str(index)
            for group in aligned_groups:
                if not np.array_equal(
                    hdf5_file[group][key][:], reference_file[group][key][:]
                ):
                    raise ValueError(
                        f"{source['path']}: {language} {group} are not aligned"
                    )


def _task2_identity(hdf5_file, kind: str, key: str, indices) -> list[str]:
    import numpy as np

    language = _text(hdf5_file[f"{kind}_embeddings"][key].attrs["language"])
    if kind == "sentence":
        sample_ids = hdf5_file["sample_ids"][key][indices]
        return [_point_id(kind, language, sample_id) for sample_id in sample_ids]

    sentence_indices = hdf5_file["token_sentence_indices"][key][indices]
    sentence_ids = hdf5_file["token_sentence_ids"][key][indices]
    lengths = hdf5_file["sequence_lengths"][key][:]
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    positions = indices - offsets[sentence_indices]
    return [
        _point_id(kind, language, sentence_id, position)
        for sentence_id, position in zip(sentence_ids, positions)
    ]


def _safe_silhouette(values, labels) -> float | None:
    from sklearn.metrics import silhouette_score

    label_count = len(set(labels))
    if label_count < 2 or label_count >= len(labels):
        return None
    score = float(silhouette_score(values, labels, metric="cosine"))
    return score if math.isfinite(score) else None


def export_task2(
    inputs: list[Path],
    projections_output: Path,
    layer_stats_output: Path,
    sentences_per_language: int,
    tokens_per_language: int,
    layers: list[int] | None,
    tsne_layers: list[int],
    seed: int,
    overwrite: bool = False,
) -> None:
    import h5py
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if sentences_per_language < 1 or tokens_per_language < 1:
        raise ValueError("Sample counts must be positive")
    if not inputs:
        raise ValueError("At least one Task 2 HDF5 file is required")
    _check_outputs(
        (projections_output, layer_stats_output),
        overwrite,
        protected=inputs,
    )

    with ExitStack() as stack:
        sources = []
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            hdf5_file = stack.enter_context(h5py.File(path, "r"))
            metadata, languages = _validate_task2_file(path, hdf5_file)
            sources.append(
                {
                    "file": hdf5_file,
                    "path": path,
                    "metadata": metadata,
                    "languages": languages,
                }
            )

        model_ids = [source["metadata"]["model_id"] for source in sources]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"Task 2 inputs have duplicate model IDs: {model_ids}")
        dimensions = {
            (
                int(source["file"].attrs["num_layers"]),
                int(source["file"].attrs["hidden_size"]),
            )
            for source in sources
        }
        language_sets = {tuple(source["languages"]) for source in sources}
        if len(dimensions) != 1 or len(language_sets) != 1:
            raise ValueError("Task 2 inputs must use the same dimensions and languages")
        _validate_task2_alignment(sources)
        num_layers, _ = dimensions.pop()
        selected_layers = (
            list(range(num_layers)) if layers is None else sorted(set(layers))
        )
        if not selected_layers or any(
            layer < 0 or layer >= num_layers for layer in selected_layers
        ):
            raise ValueError(f"Layers must be between 0 and {num_layers - 1}")
        requested_tsne = set(tsne_layers)
        if not requested_tsne.issubset(selected_layers):
            raise ValueError("Every t-SNE layer must also be selected by --layers")

        selections = {}
        for source in sources:
            for index, language in enumerate(source["languages"]):
                key = str(index)
                for kind, limit in (
                    ("sentence", sentences_per_language),
                    ("token", tokens_per_language),
                ):
                    total = source["file"][f"{kind}_embeddings"][key].shape[0]
                    selections[(source["metadata"]["model_id"], kind, key)] = (
                        _sample_indices(total, limit, seed, kind, language)
                    )

        projection_rows: list[dict[str, Any]] = []
        stats_rows: list[dict[str, Any]] = []
        for kind in ("sentence", "token"):
            for layer in selected_layers:
                blocks = []
                records = []
                for source in sources:
                    metadata = source["metadata"]
                    for index, language in enumerate(source["languages"]):
                        key = str(index)
                        indices = selections[(metadata["model_id"], kind, key)]
                        values = np.asarray(
                            source["file"][f"{kind}_embeddings"][key][
                                indices, layer, :
                            ],
                            dtype=np.float64,
                        )
                        if not np.isfinite(values).all():
                            raise ValueError(
                                f"{source['path']}: non-finite {kind} embeddings "
                                f"for {language}, layer {layer}"
                            )
                        point_ids = _task2_identity(source["file"], kind, key, indices)
                        blocks.append(values)
                        records.extend(
                            {
                                **metadata,
                                "representation": kind,
                                "layer": layer,
                                "language": language,
                                "point_id": point_id,
                            }
                            for point_id in point_ids
                        )

                combined = np.concatenate(blocks)
                if min(combined.shape) < 2:
                    raise ValueError(
                        f"Need at least two points and dimensions for {kind} PCA"
                    )
                pca = PCA(
                    n_components=2,
                    svd_solver="randomized",
                    random_state=_seed(seed, "pca", kind, layer) % (2**32 - 1),
                )
                pca_values = pca.fit_transform(combined)
                tsne_values = None
                if layer in requested_tsne:
                    perplexity = min(30.0, max(1.0, (len(combined) - 1) / 3))
                    tsne_values = TSNE(
                        n_components=2,
                        init="pca",
                        learning_rate="auto",
                        perplexity=perplexity,
                        random_state=_seed(seed, "tsne", kind, layer) % (2**32 - 1),
                    ).fit_transform(combined)

                for row_index, row in enumerate(records):
                    row.update(
                        {
                            "pca_x": _float(pca_values[row_index, 0]),
                            "pca_y": _float(pca_values[row_index, 1]),
                            "tsne_x": _float(
                                None
                                if tsne_values is None
                                else tsne_values[row_index, 0]
                            ),
                            "tsne_y": _float(
                                None
                                if tsne_values is None
                                else tsne_values[row_index, 1]
                            ),
                        }
                    )
                    projection_rows.append(row)

                explained = float(pca.explained_variance_ratio_.sum())
                start = 0
                for source in sources:
                    count = sum(
                        len(
                            selections[
                                (source["metadata"]["model_id"], kind, str(index))
                            ]
                        )
                        for index in range(len(source["languages"]))
                    )
                    stop = start + count
                    labels = [row["language"] for row in records[start:stop]]
                    metadata = source["metadata"]
                    stats_rows.append(
                        {
                            **metadata,
                            "representation": kind,
                            "layer": layer,
                            "points": count,
                            "languages": len(set(labels)),
                            "joint_pca_explained_variance": _float(explained),
                            "silhouette_cosine": _float(
                                _safe_silhouette(combined[start:stop], labels)
                            ),
                        }
                    )
                    start = stop

    projection_fields = [
        "model_id",
        "model_name",
        "model_revision",
        "method",
        "rank",
        "dataset",
        "dataset_revision",
        "split",
        "representation",
        "layer",
        "language",
        "point_id",
        "pca_x",
        "pca_y",
        "tsne_x",
        "tsne_y",
    ]
    stats_fields = projection_fields[:8] + [
        "representation",
        "layer",
        "points",
        "languages",
        "joint_pca_explained_variance",
        "silhouette_cosine",
    ]
    _write_csv(projections_output, projection_fields, projection_rows)
    _write_csv(layer_stats_output, stats_fields, stats_rows)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _task3_identity(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    config_path = summary_path.with_name("config.json")
    config = _read_json_object(config_path) if config_path.is_file() else {}
    for name in ("method", "rank", "seed"):
        if name in summary and name in config and summary[name] != config[name]:
            raise ValueError(f"{summary_path}: summary/config {name} mismatch")

    def value(name: str, default: Any = None) -> Any:
        return summary.get(name, config.get(name, default))

    method = value("method")
    seed = value("seed")
    if (
        method not in {"full", "bitfit", "ia3", "lora"}
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError(f"{summary_path}: missing or invalid method/seed")
    rank = value("rank") if method == "lora" else ""
    if method == "lora" and (not isinstance(rank, int) or rank < 1):
        raise ValueError(f"{summary_path}: LoRA rank must be positive")
    run_id = value("wandb_run_id") or value("run_id") or summary_path.parent.name
    run_name = value("wandb_run_name") or run_id
    return {
        "run_id": _text(run_id),
        "run_name": _text(run_name),
        "method": method,
        "rank": rank,
        "seed": seed,
        **{
            name: config[name]
            for name in PUBLIC_CONFIG_FIELDS
            if config.get(name) is not None
        },
    }


def _scalar_items(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}{key}": item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def _fieldnames(rows: list[dict[str, Any]], leading: list[str]) -> list[str]:
    return leading + sorted(set().union(*(row.keys() for row in rows)) - set(leading))


def _validate_task3_campaign(
    configurations: list[tuple[int, str, int | str]],
) -> None:
    by_seed: dict[int, list[tuple[str, int | str]]] = {}
    for seed, method, rank in configurations:
        by_seed.setdefault(seed, []).append((method, rank))

    errors = {}
    for seed, runs in by_seed.items():
        present = set(runs)
        if len(runs) != len(TASK3_CAMPAIGN_CONFIGS) or present != TASK3_CAMPAIGN_CONFIGS:
            errors[seed] = {
                "missing": sorted(TASK3_CAMPAIGN_CONFIGS - present, key=str),
                "unexpected": sorted(present - TASK3_CAMPAIGN_CONFIGS, key=str),
                "runs": len(runs),
            }
    if errors:
        raise ValueError(f"Incomplete Task 3 campaign by seed: {errors}")


def export_task3(
    summaries: list[Path],
    summary_output: Path,
    history_output: Path,
    overwrite: bool = False,
    require_complete_campaign: bool = False,
) -> None:
    if not summaries:
        raise ValueError("At least one Task 3 summary.json is required")
    protected = [
        path
        for summary in summaries
        for path in (
            summary,
            summary.with_name("config.json"),
            summary.with_name("history.jsonl"),
        )
    ]
    _check_outputs(
        (summary_output, history_output),
        overwrite,
        protected=protected,
    )

    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    configurations: list[tuple[int, str, int | str]] = []
    for summary_path in summaries:
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = _read_json_object(summary_path)
        identity = _task3_identity(summary_path, summary)
        if identity["run_id"] in run_ids:
            raise ValueError(f"Duplicate Task 3 run ID: {identity['run_id']}")
        run_ids.add(identity["run_id"])
        configurations.append(
            (identity["seed"], identity["method"], identity["rank"])
        )

        for required_stage in ("baseline_final", "final"):
            if (
                not isinstance(summary.get(required_stage), dict)
                or not summary[required_stage]
            ):
                raise ValueError(f"{summary_path}: missing {required_stage} metrics")
        run_values = _scalar_items(
            {
                key: value
                for key, value in summary.items()
                if key not in {*TASK3_STAGES, "method", "rank", "seed"}
            },
            prefix="run_",
        )
        for stage in TASK3_STAGES:
            datasets = summary.get(stage, {})
            if not isinstance(datasets, dict):
                raise ValueError(f"{summary_path}: {stage} must be an object")
            for dataset in sorted(datasets):
                metrics = datasets[dataset]
                if not isinstance(metrics, dict):
                    raise ValueError(
                        f"{summary_path}: {stage}/{dataset} must be an object"
                    )
                summary_rows.append(
                    {
                        **identity,
                        "stage": stage,
                        "dataset": dataset,
                        **run_values,
                        **_scalar_items(metrics),
                    }
                )

        history_path = summary_path.with_name("history.jsonl")
        if not history_path.is_file():
            raise FileNotFoundError(history_path)
        with history_path.open(encoding="utf-8") as history_file:
            for line_number, line in enumerate(history_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{history_path}:{line_number}: invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{history_path}:{line_number}: expected an object"
                    )
                history_rows.append({**_scalar_items(row), **identity})

    if not history_rows:
        raise ValueError("Task 3 histories contain no rows")
    if require_complete_campaign:
        _validate_task3_campaign(configurations)
    identity_fields = ["run_id", "run_name", "method", "rank", "seed"]
    _write_csv(
        summary_output,
        _fieldnames(summary_rows, identity_fields + ["stage", "dataset"]),
        summary_rows,
    )
    _write_csv(
        history_output,
        _fieldnames(history_rows, identity_fields),
        history_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    task2 = commands.add_parser("task2", help="Export Task 2 HDF5 projections")
    task2.add_argument("inputs", nargs="+", type=Path)
    task2.add_argument("--projections-output", type=Path, required=True)
    task2.add_argument("--layer-stats-output", type=Path, required=True)
    task2.add_argument(
        "--sentences-per-language",
        type=int,
        default=100,
        help="maximum sampled sentence points per language",
    )
    task2.add_argument(
        "--tokens-per-language",
        type=int,
        default=100,
        help="maximum sampled token points per language",
    )
    task2.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="zero-based layers to export (default: all)",
    )
    task2.add_argument(
        "--tsne-layers",
        type=int,
        nargs="*",
        default=[],
        help="selected layers that also receive t-SNE coordinates",
    )
    task2.add_argument("--seed", type=int, default=42)
    task2.add_argument("--overwrite", action="store_true")

    task3 = commands.add_parser("task3", help="Export Task 3 run metrics")
    task3.add_argument("summaries", nargs="+", type=Path)
    task3.add_argument("--summary-output", type=Path, required=True)
    task3.add_argument("--history-output", type=Path, required=True)
    task3.add_argument("--require-complete-campaign", action="store_true")
    task3.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "task2":
        export_task2(
            args.inputs,
            args.projections_output,
            args.layer_stats_output,
            args.sentences_per_language,
            args.tokens_per_language,
            args.layers,
            args.tsne_layers,
            args.seed,
            args.overwrite,
        )
    else:
        export_task3(
            args.summaries,
            args.summary_output,
            args.history_output,
            args.overwrite,
            args.require_complete_campaign,
        )


if __name__ == "__main__":
    main()

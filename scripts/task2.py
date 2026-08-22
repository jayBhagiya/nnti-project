"""Extract layer-wise XGLM token and sentence representations to HDF5."""

import argparse
import json
import random
from pathlib import Path


MODEL_NAME = "facebook/xglm-564M"
MODEL_REVISION = "f3059f01b98ccc877c673149e0178c0e957660f9"
DATASET_NAME = "facebook/flores"
DATASET_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"
LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "ita_Latn",
    "deu_Latn",
    "arb_Arab",
    "tel_Telu",
    "tam_Taml",
    "quy_Latn",
]


def sample_indices(dataset_size, num_samples, seed):
    """Return a reproducible, sorted subset shared by every language."""
    if dataset_size < 1:
        raise ValueError("The selected dataset split is empty.")
    if not 1 <= num_samples <= dataset_size:
        raise ValueError(
            f"num_samples must be between 1 and {dataset_size}, got {num_samples}."
        )
    return sorted(random.Random(seed).sample(range(dataset_size), num_samples))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("embeddings.h5"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--method", choices=("full", "bitfit", "ia3", "lora"), default=None
    )
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.rank is not None and args.rank < 1:
        parser.error("--rank must be positive")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.checkpoint is None and any(
        value is not None for value in (args.method, args.rank, args.alpha)
    ):
        parser.error("--method, --rank, and --alpha require --checkpoint")
    return args


def load_dataset(args, language, datasets):
    kwargs = {
        "split": args.split,
        "token": True,
        "trust_remote_code": True,
    }
    if args.dataset_revision:
        kwargs["revision"] = args.dataset_revision
    return datasets.load_dataset(args.dataset_name, language, **kwargs)


def _checkpoint_value(name, explicit, checkpoint, default):
    stored = checkpoint.get(name)
    if explicit is not None and stored is not None and explicit != stored:
        raise ValueError(
            f"--{name.replace('_', '-')}={explicit!r} conflicts with checkpoint "
            f"metadata {stored!r}."
        )
    return stored if stored is not None else (explicit if explicit is not None else default)


def load_model(args, torch, transformers):
    """Load a base model or reconstruct and load a Task 3 checkpoint."""
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        required = {"method", "model_name", "state_dict"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError(
                "Checkpoint must contain method, model_name, and state_dict."
            )

    metadata = checkpoint or {}
    model_name = _checkpoint_value(
        "model_name", args.model_name, metadata, MODEL_NAME
    )
    model_revision = _checkpoint_value(
        "model_revision", args.model_revision, metadata, MODEL_REVISION
    )
    method = _checkpoint_value("method", args.method, metadata, "base")
    rank = _checkpoint_value("rank", args.rank, metadata, 4)
    alpha = _checkpoint_value("alpha", args.alpha, metadata, 1.0)

    model_kwargs = {}
    if model_revision:
        model_kwargs["revision"] = model_revision
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, **model_kwargs
    )
    resolved_model_revision = (
        getattr(model.config, "_commit_hash", None) or model_revision
    )

    if checkpoint:
        from task3_custom_peft import adapt_model, load_trainable_state_dict

        model = adapt_model(model, method, rank=rank, alpha=alpha)
        state_dict = checkpoint["state_dict"]
        if method == "full":
            model.load_state_dict(state_dict, strict=True)
        else:
            load_trainable_state_dict(model, state_dict)

    return model, {
        "model_name": model_name,
        "model_revision": resolved_model_revision,
        "method": method,
        "rank": rank if method == "lora" else 0,
        "alpha": alpha if method == "lora" else 0.0,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
    }


def append_rows(dataset, values):
    """Append rows to an extendable HDF5 dataset."""
    count = len(values)
    if count == 0:
        return
    start = dataset.shape[0]
    dataset.resize(start + count, axis=0)
    dataset[start:] = values


def model_dimensions(model):
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "d_model", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = getattr(config, "num_layers", None)
    if hidden_size is None or num_hidden_layers is None:
        raise ValueError("Could not derive hidden dimensions from the model config.")
    return int(num_hidden_layers) + 1, int(hidden_size)


def create_language_datasets(hdf5_file, language_index, language, dimensions):
    import h5py

    num_layers, hidden_size = dimensions
    key = str(language_index)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    embedding_options = {
        "shape": (0, num_layers, hidden_size),
        "maxshape": (None, num_layers, hidden_size),
        "chunks": (8, num_layers, hidden_size),
        "dtype": "float32",
    }
    result = {
        "token_embeddings": hdf5_file["token_embeddings"].create_dataset(
            key, **embedding_options
        ),
        "sentence_embeddings": hdf5_file["sentence_embeddings"].create_dataset(
            key, **embedding_options
        ),
        "tokens": hdf5_file["tokens"].create_dataset(
            key, shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype
        ),
        "token_ids": hdf5_file["token_ids"].create_dataset(
            key, shape=(0,), maxshape=(None,), chunks=True, dtype="int64"
        ),
        "token_sentence_indices": hdf5_file[
            "token_sentence_indices"
        ].create_dataset(
            key, shape=(0,), maxshape=(None,), chunks=True, dtype="int64"
        ),
        "token_sentence_ids": hdf5_file["token_sentence_ids"].create_dataset(
            key, shape=(0,), maxshape=(None,), chunks=True, dtype="int64"
        ),
    }
    for dataset in result.values():
        dataset.attrs["language"] = language
    return result


def extract_language(
    args,
    dataset,
    language,
    language_index,
    expected_sample_ids,
    model,
    tokenizer,
    device,
    dimensions,
    hdf5_file,
    np,
    torch,
):
    sampled = dataset.select(hdf5_file["sample_indices"][:].tolist())
    sample_ids = [int(value) for value in sampled["id"]]
    if sample_ids != expected_sample_ids:
        raise ValueError(
            f"{language} is not aligned with {LANGUAGES[0]} for the sampled rows."
        )
    sequences = list(sampled["sentence"])
    key = str(language_index)
    hdf5_file["sample_ids"].create_dataset(key, data=sample_ids, dtype="int64")
    hdf5_file["sequences"].create_dataset(
        key,
        data=np.asarray(sequences, dtype=object),
        dtype=hdf5_file["language_codes"].dtype,
    )
    sequence_lengths = hdf5_file["sequence_lengths"].create_dataset(
        key, shape=(len(sequences),), dtype="int64"
    )
    datasets_out = create_language_datasets(
        hdf5_file, language_index, language, dimensions
    )

    loader = torch.utils.data.DataLoader(
        sequences, batch_size=args.batch_size, shuffle=False
    )
    num_layers, hidden_size = dimensions
    sentence_offset = 0
    with torch.inference_mode():
        for sentences in loader:
            sentences = list(sentences)
            encoded = tokenizer(
                sentences, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("The model did not return hidden states.")
            stacked = torch.stack(hidden_states, dim=2)
            if stacked.shape[2:] != (num_layers, hidden_size):
                raise RuntimeError(
                    "Hidden-state shape does not match the model configuration: "
                    f"{tuple(stacked.shape[2:])} != {(num_layers, hidden_size)}."
                )

            mask = attention_mask.bool()
            lengths = mask.sum(dim=1)
            if torch.any(lengths == 0):
                raise RuntimeError("Tokenizer produced an empty sequence.")
            weights = mask[:, :, None, None].to(stacked.dtype)
            sentence_embeddings = stacked.mul(weights).sum(dim=1)
            sentence_embeddings = sentence_embeddings / lengths[:, None, None]
            token_embeddings = stacked[mask]

            batch_size = len(sentences)
            batch_lengths = lengths.cpu().numpy()
            stop = sentence_offset + batch_size
            sentence_indices = np.repeat(
                np.arange(sentence_offset, stop, dtype=np.int64), batch_lengths
            )
            sentence_ids = np.repeat(
                np.asarray(sample_ids[sentence_offset:stop], dtype=np.int64),
                batch_lengths,
            )
            flat_token_ids = input_ids[mask].cpu().numpy().astype(np.int64)
            token_strings = tokenizer.convert_ids_to_tokens(flat_token_ids.tolist())

            append_rows(
                datasets_out["token_embeddings"],
                token_embeddings.float().cpu().numpy(),
            )
            append_rows(
                datasets_out["sentence_embeddings"],
                sentence_embeddings.float().cpu().numpy(),
            )
            append_rows(datasets_out["tokens"], np.asarray(token_strings, dtype=object))
            append_rows(datasets_out["token_ids"], flat_token_ids)
            append_rows(
                datasets_out["token_sentence_indices"], sentence_indices
            )
            append_rows(datasets_out["token_sentence_ids"], sentence_ids)
            sequence_lengths[sentence_offset:stop] = batch_lengths
            sentence_offset = stop
            del (
                output,
                hidden_states,
                stacked,
                sentence_embeddings,
                token_embeddings,
                input_ids,
                attention_mask,
                mask,
                weights,
                lengths,
            )

    if sentence_offset != len(sequences):
        raise RuntimeError(f"Only extracted {sentence_offset}/{len(sequences)} sequences.")
    hdf5_file["sample_ids"][key].attrs["language"] = language
    hdf5_file["sequences"][key].attrs["language"] = language
    sequence_lengths.attrs["language"] = language
    datasets_out["token_embeddings"].attrs["num_tokens"] = datasets_out[
        "token_embeddings"
    ].shape[0]
    hdf5_file.flush()


def run(args):
    import datasets
    import h5py
    import numpy as np
    import torch
    import transformers

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it."
        )

    reference_dataset = load_dataset(args, LANGUAGES[0], datasets)
    indices = sample_indices(len(reference_dataset), args.num_samples, args.seed)
    reference_ids = [int(reference_dataset[index]["id"]) for index in indices]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, model_metadata = load_model(args, torch, transformers)
    tokenizer_kwargs = {}
    if model_metadata["model_revision"]:
        tokenizer_kwargs["revision"] = model_metadata["model_revision"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_metadata["model_name"], **tokenizer_kwargs
    )
    tokenizer.padding_side = "left"
    model.eval().to(device)
    dimensions = model_dimensions(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as hdf5_file:
        hdf5_file.attrs.update(
            {
                "schema_version": 2,
                "complete": False,
                "contains_raw_dataset_text": True,
                "publication_ready": False,
                "model_name": model_metadata["model_name"],
                "model_revision": getattr(model.config, "_commit_hash", None)
                or model_metadata["model_revision"]
                or "",
                "adaptation_method": model_metadata["method"],
                "lora_rank": model_metadata["rank"],
                "lora_alpha": model_metadata["alpha"],
                "checkpoint": model_metadata["checkpoint"],
                "dataset_name": args.dataset_name,
                "dataset_revision": args.dataset_revision or "",
                "split": args.split,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "num_layers": dimensions[0],
                "hidden_size": dimensions[1],
                "language_mapping": json.dumps(dict(enumerate(LANGUAGES))),
            }
        )
        hdf5_file.create_dataset("sample_indices", data=indices, dtype="int64")
        hdf5_file.create_dataset(
            "language_codes",
            data=np.asarray(LANGUAGES, dtype=object),
            dtype=string_dtype,
        )
        for group_name in (
            "token_embeddings",
            "sentence_embeddings",
            "tokens",
            "token_ids",
            "token_sentence_indices",
            "token_sentence_ids",
            "sequences",
            "sample_ids",
            "sequence_lengths",
        ):
            hdf5_file.create_group(group_name)

        for language_index, language in enumerate(LANGUAGES):
            print(f"Extracting {language} ({language_index + 1}/{len(LANGUAGES)})")
            language_dataset = (
                reference_dataset
                if language_index == 0
                else load_dataset(args, language, datasets)
            )
            if len(language_dataset) != len(reference_dataset):
                raise ValueError(
                    f"{language} has {len(language_dataset)} rows; expected "
                    f"{len(reference_dataset)} for an aligned FLORES split."
                )
            hdf5_file.attrs[f"dataset_fingerprint_{language}"] = getattr(
                language_dataset, "_fingerprint", ""
            )
            extract_language(
                args,
                language_dataset,
                language,
                language_index,
                reference_ids,
                model,
                tokenizer,
                device,
                dimensions,
                hdf5_file,
                np,
                torch,
            )
        hdf5_file.attrs["complete"] = True

    print(f"Embeddings saved to {args.output}")


def self_test():
    first = sample_indices(50, 20, 42)
    assert first == sample_indices(50, 20, 42)
    assert first != sample_indices(50, 20, 43)
    assert first == sorted(first) and len(first) == len(set(first)) == 20
    try:
        sample_indices(10, 11, 42)
    except ValueError:
        pass
    else:
        raise AssertionError("Oversampling should fail.")
    print("Task 2 sampling self-test passed.")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        run(arguments)

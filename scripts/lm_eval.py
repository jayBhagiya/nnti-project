"""Evaluate causal language models on parallel FLORES sentences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch


DATASET_NAME = "facebook/flores"
DATASET_REVISION = "2db78afdeaccaedc3b33a95442a4e55766887e17"
TASK1_MODELS = ("facebook/xglm-564M", "gpt2")
MODEL_REVISIONS = {
    "facebook/xglm-564M": "f3059f01b98ccc877c673149e0178c0e957660f9",
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
}
TASK1_LANGUAGES = (
    "eng_Latn",
    "spa_Latn",
    "ita_Latn",
    "deu_Latn",
    "arb_Arab",
    "tel_Telu",
    "tam_Taml",
    "quy_Latn",
    "guj_Gujr",
)
NATIVE_LANGUAGE = "guj_Gujr"


def causal_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask padding and each sequence's context-free first token."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    real_tokens = attention_mask.bool()
    first_real_token = attention_mask.cumsum(dim=1).eq(1) & real_tokens
    labels = input_ids.clone()
    labels[~real_tokens | first_real_token] = -100
    return labels


def evaluate_causal_lm(
    model: Any,
    tokenizer: Any,
    sentences: Sequence[str],
    *,
    batch_size: int,
    device: str | torch.device,
    max_length: int | None = None,
) -> dict[str, int | float]:
    """Return token-weighted metrics, using sliding windows when needed."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not sentences:
        raise ValueError("sentences must not be empty")
    if max_length is None:
        max_length = getattr(
            getattr(model, "config", None),
            "max_position_embeddings",
            None,
        )
    if max_length is not None and max_length < 2:
        raise ValueError("max_length must leave room for context and a target")

    nll_sum = 0.0
    token_count = 0
    predicted_token_count = 0
    context_windows = 0
    byte_count = sum(len(sentence.encode("utf-8")) for sentence in sentences)
    default_specials = tokenizer.build_inputs_with_special_tokens([])
    start_token_id = default_specials[0] if default_specials else None
    if start_token_id is None:
        start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = tokenizer.eos_token_id
    if start_token_id is None:
        raise ValueError("tokenizer must define a BOS or EOS token")

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            batch = list(sentences[start : start + batch_size])
            token_ids = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            windows: list[list[int]] = []
            prefix_lengths: list[int] = []
            for ids in token_ids:
                sequence = [start_token_id, *ids]
                if max_length is None or len(sequence) <= max_length:
                    windows.append(sequence)
                    prefix_lengths.append(1)
                else:
                    stride = max(1, max_length // 2)
                    for target_start in range(1, len(sequence), stride):
                        target_end = min(target_start + stride, len(sequence))
                        window_start = max(0, target_end - max_length)
                        windows.append(sequence[window_start:target_end])
                        prefix_lengths.append(target_start - window_start)
            encoded = tokenizer.pad(
                {"input_ids": windows},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            if "attention_mask" not in encoded:
                raise ValueError("tokenizer must return an attention_mask")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = causal_labels(input_ids, attention_mask)
            for row, prefix_length in enumerate(prefix_lengths):
                labels[row, :prefix_length] = -100
            scored_tokens = labels[:, 1:].ne(-100).sum().item()
            context_windows += len(windows)
            token_count += sum(len(ids) for ids in token_ids)
            if scored_tokens == 0:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            nll_sum += outputs.loss.detach().float().item() * scored_tokens
            predicted_token_count += scored_tokens

    if predicted_token_count == 0 or byte_count == 0:
        raise ValueError("evaluation data must contain scored tokens and UTF-8 bytes")

    nll = nll_sum / predicted_token_count
    return {
        "sentences": len(sentences),
        "context_windows": context_windows,
        "tokens": token_count,
        "predicted_tokens": predicted_token_count,
        "utf8_bytes": byte_count,
        "nll": nll,
        "perplexity": math.exp(nll),
        "bits_per_byte": nll_sum / (byte_count * math.log(2)),
        "tokens_per_byte": token_count / byte_count,
    }


def select_indices(size: int, max_samples: int | None, seed: int) -> list[int]:
    """Select one deterministic, order-preserving subset for every language."""
    if size < 1:
        raise ValueError("dataset split must not be empty")
    if max_samples is None or max_samples >= size:
        return list(range(size))
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    return sorted(random.Random(seed).sample(range(size), max_samples))


def write_results(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write web-ready CSV or JSON based on the output filename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output filename must end in .csv or .json")
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Native-language comparison: --languages {NATIVE_LANGUAGE}",
    )
    parser.add_argument("--models", nargs="+", default=TASK1_MODELS)
    parser.add_argument("--languages", nargs="+", default=TASK1_LANGUAGES)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("task1_metrics.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("output filename must end in .csv or .json")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    language_data: dict[str, list[str]] = {}
    dataset_fingerprints: dict[str, str] = {}
    indices: list[int] | None = None
    split_size: int | None = None
    for language in args.languages:
        dataset = load_dataset(
            args.dataset,
            language,
            split=args.split,
            revision=args.dataset_revision,
            trust_remote_code=True,
        )
        if split_size is None:
            split_size = len(dataset)
            indices = select_indices(split_size, args.max_samples, args.seed)
        elif len(dataset) != split_size:
            raise ValueError("FLORES language splits must have equal lengths")
        selected = dataset.select(indices)
        language_data[language] = selected["sentence"]
        dataset_fingerprints[language] = getattr(selected, "_fingerprint", "")

    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        requested_revision = MODEL_REVISIONS.get(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=requested_revision,
        ).to(device)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=resolved_revision,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(f"{model_name} has neither a pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        for language, sentences in language_data.items():
            metrics = evaluate_causal_lm(
                model,
                tokenizer,
                sentences,
                batch_size=args.batch_size,
                device=device,
            )
            row = {
                "dataset": args.dataset,
                "split": args.split,
                "model": model_name,
                "model_revision": resolved_revision or "",
                "language": language,
                "dataset_revision": args.dataset_revision or "",
                "dataset_fingerprint": dataset_fingerprints[language],
                "seed": args.seed,
                **metrics,
            }
            rows.append(row)
            write_results(rows, args.output)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

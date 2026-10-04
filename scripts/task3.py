"""Train one XGLM adaptation configuration and record reproducible metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler

try:
    from .lm_eval import causal_labels, evaluate_causal_lm
    from .task3_custom_peft import (
        build_adapted_model,
        configure_gradient_checkpointing,
        load_trainable_state_dict,
        trainable_state_dict,
    )
    from .task3_data_preparation import (
        prepare_adaptation_data,
        prepare_flores_eval,
    )
    from .task3_utils import parameter_counts
except ImportError:
    from lm_eval import causal_labels, evaluate_causal_lm
    from task3_custom_peft import (
        build_adapted_model,
        configure_gradient_checkpointing,
        load_trainable_state_dict,
        trainable_state_dict,
    )
    from task3_data_preparation import (
        prepare_adaptation_data,
        prepare_flores_eval,
    )
    from task3_utils import parameter_counts


MODEL_NAME = "facebook/xglm-564M"
MODEL_REVISION = "f3059f01b98ccc877c673149e0178c0e957660f9"
ADAPTATION_DATASET = "somosnlp-hackathon-2022/spanish-to-quechua"
ADAPTATION_DATASET_REVISION = "aa48b3c7f4d0c1450f8f2df27ceb8a882b022600"
FLORES_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"
ADAPTATION_DATA_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
FLORES_LANGUAGES = (
    "eng_Latn",
    "spa_Latn",
    "ita_Latn",
    "deu_Latn",
    "arb_Arab",
    "tel_Telu",
    "tam_Taml",
    "quy_Latn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("full", "bitfit", "lora", "ia3"), required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset", default=ADAPTATION_DATASET)
    parser.add_argument("--dataset-revision", default=ADAPTATION_DATASET_REVISION)
    parser.add_argument("--flores-revision", default=FLORES_REVISION)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--adaptation-eval-samples", type=int, default=200)
    parser.add_argument("--flores-eval-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", default="nnti-xglm-quechua")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", "pilot-v1"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "offline"),
    )
    args = parser.parse_args()

    positive = {
        "train_samples": args.train_samples,
        "adaptation_eval_samples": args.adaptation_eval_samples,
        "flores_eval_samples": args.flores_eval_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if args.method == "lora" and args.rank < 1:
        raise ValueError("LoRA requires --rank >= 1")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_texts(texts: Sequence[str], sample_size: int | None, seed: int) -> list[str]:
    if sample_size is None or sample_size >= len(texts):
        return list(texts)
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    indices = sorted(random.Random(seed).sample(range(len(texts)), sample_size))
    return [texts[index] for index in indices]


def text_manifest_hash(named_texts: dict[str, Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named_texts):
        digest.update(name.encode("utf-8"))
        for text in named_texts[name]:
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def git_commit() -> str:
    git_dir = Path(__file__).resolve().parents[1] / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head.removeprefix("ref: ")
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {ref}"):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def tokenize_training_data(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
) -> list[dict[str, list[int]]]:
    return [
        tokenizer(text, truncation=True, max_length=max_length)
        for text in texts
    ]


def make_train_loader(
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    seed: int,
    num_workers: int,
    max_length: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        tokenize_training_data(tokenizer, texts, max_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        generator=generator,
        num_workers=num_workers,
    )


def evaluate_named_sets(
    model,
    tokenizer,
    named_texts: dict[str, Sequence[str]],
    batch_size: int,
    device: torch.device,
    stage: str,
    epoch: int,
    step: int,
    history_path: Path,
    run,
) -> dict[str, dict[str, int | float]]:
    results: dict[str, dict[str, int | float]] = {}
    for name, texts in named_texts.items():
        metrics = evaluate_causal_lm(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            device=device,
        )
        results[name] = metrics
        row = {
            "stage": stage,
            "epoch": epoch,
            "step": step,
            "dataset": name,
            **metrics,
        }
        append_jsonl(history_path, row)
        run.log(
            {
                "global_step": step,
                **{
                    f"{stage}/{name}/{key}": value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return results


def save_checkpoint(
    path: Path,
    model,
    args: argparse.Namespace,
    epoch: int,
    validation_nll: float,
) -> None:
    state_dict = (
        model.state_dict()
        if args.method == "full"
        else trainable_state_dict(model)
    )
    payload = {
        "method": args.method,
        "rank": args.rank,
        "alpha": args.alpha,
        "model_name": args.model_name,
        "model_revision": getattr(model.config, "_commit_hash", None)
        or args.model_revision,
        "epoch": epoch,
        "validation_nll": validation_nll,
        "state_dict": state_dict,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, method: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload["method"] != method:
        raise ValueError(
            f"Checkpoint method {payload['method']} does not match {method}"
        )
    if method == "full":
        model.load_state_dict(payload["state_dict"], strict=True)
    else:
        load_trainable_state_dict(model, payload["state_dict"])
    return payload


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    fp16: bool,
    gradient_accumulation: int,
    max_grad_norm: float,
    epoch: int,
    global_step: int,
    history_path: Path,
    run,
) -> tuple[int, int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    nll_sum = 0.0
    predicted_tokens = 0
    step_nll_sum = 0.0
    step_tokens = 0

    for batch_index, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = causal_labels(input_ids, attention_mask)
        scored_tokens = labels[:, 1:].ne(-100).sum().item()
        group_start = (batch_index // gradient_accumulation) * gradient_accumulation
        group_size = min(gradient_accumulation, len(loader) - group_start)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            scaled_loss = loss / group_size

        scaler.scale(scaled_loss).backward()
        nll_sum += loss.detach().float().item() * scored_tokens
        predicted_tokens += scored_tokens
        step_nll_sum += loss.detach().float().item() * scored_tokens
        step_tokens += scored_tokens

        should_step = (
            (batch_index + 1) % gradient_accumulation == 0
            or batch_index + 1 == len(loader)
        )
        if not should_step:
            continue

        scaler.unscale_(optimizer)
        clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() < previous_scale:
            step_nll_sum = 0.0
            step_tokens = 0
            continue

        scheduler.step()
        global_step += 1
        run.log(
            {
                "global_step": global_step,
                "train/step_nll": step_nll_sum / step_tokens,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/predicted_tokens": predicted_tokens,
                "epoch": epoch,
            },
        )
        step_nll_sum = 0.0
        step_tokens = 0

    epoch_nll = nll_sum / predicted_tokens
    row = {
        "stage": "train",
        "epoch": epoch,
        "step": global_step,
        "nll": epoch_nll,
        "perplexity": math.exp(epoch_nll),
        "predicted_tokens": predicted_tokens,
    }
    append_jsonl(history_path, row)
    run.log(
        {
            "global_step": global_step,
            **{
                f"train/{key}": value
                for key, value in row.items()
                if key not in {"stage", "step"}
            },
        },
    )
    print(json.dumps(row), flush=True)
    return global_step, predicted_tokens, epoch_nll


def run_task3(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    # fp16 autocast and loss scaling only help on a GPU.
    use_fp16 = args.fp16 and use_cuda
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    args.output_dir.mkdir(parents=True)
    history_path = args.output_dir / "history.jsonl"
    checkpoint_path = args.output_dir / "best.pt"

    adaptation = prepare_adaptation_data(
        args.dataset,
        data_files=ADAPTATION_DATA_FILES,
        seed=args.data_seed,
        train_samples=args.train_samples,
        revision=args.dataset_revision,
    )
    flores_dev = prepare_flores_eval(
        list(FLORES_LANGUAGES),
        split="dev",
        sample_size=args.flores_eval_samples,
        seed=args.data_seed,
        revision=args.flores_revision,
    )
    flores_devtest = prepare_flores_eval(
        list(FLORES_LANGUAGES),
        split="devtest",
        sample_size=args.test_samples,
        seed=args.data_seed,
        revision=args.flores_revision,
    )
    development_sets = {
        "adaptation_validation": select_texts(
            adaptation["validation"],
            args.adaptation_eval_samples,
            args.data_seed,
        ),
        **{f"flores_dev/{language}": texts for language, texts in flores_dev.items()},
    }
    final_sets = {
        "adaptation_test": select_texts(
            adaptation["test"],
            args.test_samples,
            args.data_seed,
        ),
        **{
            f"flores_devtest/{language}": texts
            for language, texts in flores_devtest.items()
        },
    }

    model = build_adapted_model(
        args.model_name,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        revision=args.model_revision,
    ).to(device)
    resolved_model_revision = (
        getattr(model.config, "_commit_hash", None) or args.model_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=resolved_model_revision,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if args.gradient_checkpointing:
        configure_gradient_checkpointing(model)

    counts = parameter_counts(model)
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **counts,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_sequence_length": model.config.max_position_embeddings,
        "git_commit": git_commit(),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "device": torch.cuda.get_device_name(device) if use_cuda else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "adaptation_split_sizes": {
            split: len(texts) for split, texts in adaptation.items()
        },
        "training_manifest_sha256": text_manifest_hash(
            {"adaptation_train": adaptation["train"]}
        ),
        "development_manifest_sha256": text_manifest_hash(development_sets),
        "final_manifest_sha256": text_manifest_hash(final_sets),
    }
    write_json(args.output_dir / "config.json", config)

    run_name = args.wandb_run_name or (
        f"{args.method}-r{args.rank}-s{args.seed}"
        if args.method == "lora"
        else f"{args.method}-s{args.seed}"
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    started_at = time.perf_counter()
    baseline_development = evaluate_named_sets(
        model,
        tokenizer,
        development_sets,
        args.batch_size,
        device,
        stage="baseline_development",
        epoch=0,
        step=0,
        history_path=history_path,
        run=run,
    )
    baseline_final = evaluate_named_sets(
        model,
        tokenizer,
        final_sets,
        args.batch_size,
        device,
        stage="baseline_final",
        epoch=0,
        step=0,
        history_path=history_path,
        run=run,
    )
    best_nll = baseline_development["adaptation_validation"]["nll"]
    best_epoch = 0
    save_checkpoint(checkpoint_path, model, args, best_epoch, best_nll)

    train_loader = make_train_loader(
        tokenizer,
        adaptation["train"],
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_length=model.config.max_position_embeddings,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(f"{args.method} did not expose any trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    total_train_tokens = 0
    training_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        training_started_at = time.perf_counter()
        global_step, epoch_tokens, _ = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_fp16,
            args.gradient_accumulation,
            args.max_grad_norm,
            epoch,
            global_step,
            history_path,
            run,
        )
        training_seconds += time.perf_counter() - training_started_at
        total_train_tokens += epoch_tokens
        development = evaluate_named_sets(
            model,
            tokenizer,
            development_sets,
            args.batch_size,
            device,
            stage="development",
            epoch=epoch,
            step=global_step,
            history_path=history_path,
            run=run,
        )
        validation_nll = development["adaptation_validation"]["nll"]
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                args,
                best_epoch,
                best_nll,
            )

    checkpoint = load_checkpoint(checkpoint_path, model, args.method)
    final_metrics = evaluate_named_sets(
        model,
        tokenizer,
        final_sets,
        args.batch_size,
        device,
        stage="final",
        epoch=best_epoch,
        step=global_step,
        history_path=history_path,
        run=run,
    )
    wall_seconds = time.perf_counter() - started_at
    summary = {
        "method": args.method,
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        **counts,
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "checkpoint_validation_nll": checkpoint["validation_nll"],
        "wall_seconds": wall_seconds,
        "training_seconds": training_seconds,
        "training_predicted_tokens": total_train_tokens,
        "training_tokens_per_second": total_train_tokens / training_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else 0,
        "baseline_development": baseline_development,
        "baseline_final": baseline_final,
        "final": final_metrics,
    }
    write_json(args.output_dir / "summary.json", summary)
    run.summary.update(
        {
            key: value
            for key, value in summary.items()
            if isinstance(value, (int, float, str))
        }
    )
    run.finish()


if __name__ == "__main__":
    run_task3(parse_args())

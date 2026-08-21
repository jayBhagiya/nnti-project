"""Small, self-contained implementations of the Task 3 adaptation methods."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj"}


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_a, std=0.02)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scale


class IA3OutputLinear(nn.Module):
    """Scale a frozen projection's output, used for attention keys/values."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.ia3_scale = nn.Parameter(torch.ones(base.out_features))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) * self.ia3_scale


class IA3InputLinear(nn.Module):
    """Scale the feed-forward inner activation before its output projection."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.ia3_scale = nn.Parameter(torch.ones(base.in_features))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs * self.ia3_scale)


def _replace_linears(
    module: nn.Module,
    target_names: set[str],
    factory: Callable[[nn.Linear], nn.Module],
) -> None:
    for name, child in list(module.named_children()):
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name, factory(child))
        else:
            _replace_linears(child, target_names, factory)


def adapt_model(
    model: nn.Module,
    method: str,
    rank: int = 4,
    alpha: float = 1.0,
) -> nn.Module:
    """Apply one adaptation method in place and return ``model``."""

    method = method.lower()
    if method not in {"full", "bitfit", "lora", "ia3"}:
        raise ValueError(f"Unknown adaptation method: {method}")

    for parameter in model.parameters():
        parameter.requires_grad_(method == "full")

    if method == "bitfit":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.endswith(".bias"))
    elif method == "lora":
        _replace_linears(
            model,
            LORA_TARGETS,
            lambda layer: LoRALinear(layer, rank=rank, alpha=alpha),
        )
    elif method == "ia3":
        _replace_linears(model, {"k_proj", "v_proj"}, IA3OutputLinear)
        _replace_linears(model, {"fc2"}, IA3InputLinear)

    return model


def build_adapted_model(
    model_name: str,
    method: str,
    rank: int = 4,
    alpha: float = 1.0,
    revision: str | None = None,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    return adapt_model(model, method=method, rank=rank, alpha=alpha)


def configure_gradient_checkpointing(model: nn.Module) -> None:
    """Keep frozen-parameter adapters connected through reentrant checkpoints."""

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU snapshot of parameters that the optimizer can update."""

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def load_trainable_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state_dict) != expected:
        missing = sorted(expected - set(state_dict))
        unexpected = sorted(set(state_dict) - expected)
        raise ValueError(
            f"Adapter checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state_dict, strict=False)


class _AdaptedModel(nn.Module):
    method = "full"

    def __init__(
        self,
        model_name: str,
        rank: int = 4,
        alpha: float = 1.0,
        revision: str | None = None,
    ):
        super().__init__()
        self.model = build_adapted_model(
            model_name,
            method=self.method,
            rank=rank,
            alpha=alpha,
            revision=revision,
        )
        self.model_config = self.model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class BitFitAdaptedModel(_AdaptedModel):
    method = "bitfit"


class LoRaAdaptedModel(_AdaptedModel):
    method = "lora"


class IA3AdaptedModel(_AdaptedModel):
    method = "ia3"

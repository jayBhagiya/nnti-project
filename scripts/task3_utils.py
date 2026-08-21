"""Small reporting helpers for Task 3."""


def parameter_counts(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


class ParamsUtils:
    """Compatibility reporter retained for the original notebook/script API."""

    def __init__(self, model, peft_model, name):
        del model
        self.model = peft_model
        self.name = name

    def print_stats(self):
        counts = parameter_counts(self.model)
        percentage = (
            100 * counts["trainable_parameters"] / counts["total_parameters"]
        )
        print(f"Parameter counts for {self.name}")
        print(f"Total parameters: {counts['total_parameters']:,}")
        print(f"Trainable parameters: {counts['trainable_parameters']:,}")
        print(f"Trainable share: {percentage:.4f}%")

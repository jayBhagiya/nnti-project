import copy
import unittest

import torch
from torch import nn
from transformers import XGLMConfig, XGLMForCausalLM

from scripts.task3_custom_peft import (
    IA3InputLinear,
    IA3OutputLinear,
    LoRALinear,
    adapt_model,
    configure_gradient_checkpointing,
    load_trainable_state_dict,
    trainable_state_dict,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 4)
        self.k_proj = nn.Linear(3, 4)
        self.v_proj = nn.Linear(3, 4)
        self.out_proj = nn.Linear(4, 3)
        self.fc2 = nn.Linear(5, 3)


class PeftTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randn(2, 3)

    def test_lora_starts_at_base_and_only_adapters_train(self):
        base = ToyModel()
        expected = base.q_proj(self.inputs)
        model = adapt_model(copy.deepcopy(base), "lora", rank=2, alpha=2)

        self.assertIsInstance(model.q_proj, LoRALinear)
        torch.testing.assert_close(model.q_proj(self.inputs), expected)
        model.q_proj(self.inputs).sum().backward()
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertIsNone(model.q_proj.base.weight.grad)

    def test_ia3_starts_at_base_and_round_trips(self):
        base = ToyModel()
        expected_k = base.k_proj(self.inputs)
        ff_inputs = torch.randn(2, 5)
        expected_ff = base.fc2(ff_inputs)
        model = adapt_model(copy.deepcopy(base), "ia3")

        self.assertIsInstance(model.k_proj, IA3OutputLinear)
        self.assertIsInstance(model.fc2, IA3InputLinear)
        torch.testing.assert_close(model.k_proj(self.inputs), expected_k)
        torch.testing.assert_close(model.fc2(ff_inputs), expected_ff)

        state = trainable_state_dict(model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data.fill_(2)
        changed = trainable_state_dict(model)
        self.assertFalse(all(torch.equal(state[key], changed[key]) for key in state))
        load_trainable_state_dict(model, state)
        for key, value in trainable_state_dict(model).items():
            torch.testing.assert_close(value, state[key])

    def test_bitfit_only_trains_biases(self):
        model = adapt_model(ToyModel(), "bitfit")
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(name.endswith(".bias") for name in trainable))

    def test_frozen_adapters_backpropagate_with_gradient_checkpointing(self):
        config = XGLMConfig(
            vocab_size=32,
            max_position_embeddings=16,
            d_model=16,
            ffn_dim=32,
            num_layers=2,
            attention_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
        )
        input_ids = torch.tensor([[2, 4, 5, 6]])
        for method in ("bitfit", "lora", "ia3"):
            model = adapt_model(XGLMForCausalLM(config), method, rank=2)
            configure_gradient_checkpointing(model)
            model.train()
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients, method)
            self.assertTrue(all(gradient is not None for gradient in gradients), method)


if __name__ == "__main__":
    unittest.main()

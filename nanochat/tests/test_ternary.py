import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig
from nanochat.kernels import TernaryLinear, ternary_linear_kernel, ternary_quantize


def test_ternary_quantize_expected_values():
    weight = torch.tensor([[0.1, -0.9, 0.4], [0.0, 0.2, -0.8]], dtype=torch.float32)
    stats = ternary_quantize(weight, threshold=0.5)
    expected = torch.tensor([[0.0, -0.65, 0.65], [0.0, 0.5, -0.5]], dtype=weight.dtype)
    expected_mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=weight.dtype)
    assert torch.allclose(stats.weight, expected, atol=1e-6)
    assert torch.allclose(stats.mask, expected_mask, atol=1e-6)


def test_ternary_linear_matches_manual_projection():
    weight = torch.tensor([[0.1, -0.9, 0.4], [0.0, 0.2, -0.8]], dtype=torch.float32)
    stats = ternary_quantize(weight, threshold=0.5)
    layer = TernaryLinear(in_features=3, out_features=2, threshold=0.5)
    with torch.no_grad():
        layer.weight.copy_(weight)
    inputs = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)
    output = layer(inputs)
    manual = inputs.matmul(stats.weight.t())
    assert torch.allclose(output, manual, atol=1e-6)


def test_ternary_kernel_backward_masks_zero_entries():
    torch.manual_seed(0)
    inputs = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    output = ternary_linear_kernel(inputs, weight, threshold=0.6)
    loss = output.pow(2).mean()
    loss.backward()
    stats = ternary_quantize(weight.detach(), threshold=0.6)
    assert torch.all(torch.isfinite(inputs.grad))
    assert torch.all(torch.isfinite(weight.grad))
    assert torch.allclose(weight.grad * (1 - stats.mask), torch.zeros_like(weight.grad), atol=1e-6)


def test_gpt_ternary_flag_switches_linear_layers():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        ternary_weights=True,
        ternary_threshold=0.55,
    )
    model = GPT(config)
    block = model.transformer.h[0]
    assert isinstance(block.attn.c_q, TernaryLinear)
    assert isinstance(block.attn.c_proj, TernaryLinear)
    assert isinstance(block.mlp.c_fc, TernaryLinear)
    assert isinstance(block.mlp.c_proj, TernaryLinear)
    assert block.attn.c_q.threshold == pytest.approx(0.55, abs=1e-6)
    assert isinstance(model.lm_head, TernaryLinear)

    dense_config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        ternary_weights=False,
    )
    dense_model = GPT(dense_config)
    dense_block = dense_model.transformer.h[0]
    assert isinstance(dense_block.attn.c_q, nn.Linear)
    assert isinstance(dense_block.mlp.c_fc, nn.Linear)
    assert isinstance(dense_model.lm_head, nn.Linear)

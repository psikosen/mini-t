import torch

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

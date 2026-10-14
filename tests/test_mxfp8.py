import prime_kernels
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def kernel():
    return prime_kernels.load("mxfp8")


def reference(x):
    blocked = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocked.abs().amax(-1)
    _, binary_exponent = torch.frexp(amax.double())
    unbiased = binary_exponent - 9
    boundary = torch.ldexp(torch.full_like(amax, 448, dtype=torch.float64), unbiased)
    unbiased += (amax > boundary).to(torch.int32)
    exponent = torch.where(amax == 0, 0, unbiased + 127).clamp(0, 254)
    scale = exponent.to(torch.uint8)
    inv_scale = torch.ldexp(torch.ones_like(amax, dtype=torch.float64), 127 - exponent).float()
    inv_scale = torch.where(scale == 0, 0, inv_scale)
    q = (blocked * inv_scale.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    decoded = q.float().reshape_as(blocked) * torch.ldexp(torch.ones_like(amax), exponent - 127).unsqueeze(-1)
    return q, scale, decoded.reshape_as(x).bfloat16()


@pytest.mark.parametrize("shape", [(17, 96), (512, 4096), (3, 129, 256)])
@pytest.mark.parametrize("magnitude", [1.0, 0.02, 1e-20])
def test_nearest_cast_and_dequantize(kernel, shape, magnitude):
    torch.manual_seed(37)
    x = torch.randn((*shape[:-2], shape[-1], shape[-2]), device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
    x = x * magnitude
    x[..., 0, :] = 0
    x[..., 1, :] = -0.0
    q, s = kernel.quantize(x)
    rq, rs, decoded = reference(x)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s, rs)
    assert torch.equal(kernel.dequantize(q, s), decoded)


def test_all_finite_bf16_magnitudes_match_vllm(kernel):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    bits = torch.arange(0x7F80, device="cuda", dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16).unsqueeze(-1).repeat(1, 32)
    x[:, 1::2] *= -1
    q, s = kernel.quantize(x)
    vq, vs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    assert torch.equal(s, vs)
    assert torch.equal(q.view(torch.uint8), vq.view(torch.uint8))


def test_empty_input(kernel):
    x = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    q, s = kernel.quantize(x)
    assert q.shape == x.shape
    assert s.shape == (0, 2)
    assert kernel.dequantize(q, s).shape == x.shape

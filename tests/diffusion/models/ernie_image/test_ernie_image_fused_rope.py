# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import (
    _apply_qk_rotary_emb,
    _apply_rotary_emb,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def reset_fused_rope_state():
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    fused_rope._FAILED_KEYS.clear()
    fused_rope._VERIFIED_KEYS.clear()
    yield
    fused_rope._FAILED_KEYS.clear()
    fused_rope._VERIFIED_KEYS.clear()


def _inputs(shape: tuple[int, int, int, int]):
    batch, sequence, _, head_dim = shape
    torch.manual_seed(17)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    freqs_cos = torch.randn(
        batch,
        sequence,
        head_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    freqs_sin = torch.randn_like(freqs_cos)
    return query, key, freqs_cos, freqs_sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 128),
        (2, 257, 7, 128),
        (2, 4608, 32, 128),
    ],
)
def test_fused_qk_rope_is_bit_exact(shape):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs(shape)
    with torch.inference_mode():
        expected_query = _apply_rotary_emb(query, freqs_cos, freqs_sin)
        expected_key = _apply_rotary_emb(key, freqs_cos, freqs_sin)
        actual_query, actual_key = _apply_qk_rotary_emb(query, key, freqs_cos, freqs_sin)

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)
    assert len(fused_rope._VERIFIED_KEYS) == 1
    assert not fused_rope._FAILED_KEYS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_verified_key_does_not_repeat_eager_check():
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((2, 257, 8, 128))
    with torch.inference_mode():
        first = fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin, _apply_rotary_emb)
        assert first is not None

        def unexpected_eager_call(*args):
            raise AssertionError("verified geometry must not rerun eager verification")

        second = fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin, unexpected_eager_call)
    assert second is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_mismatch_disables_runtime_key(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((1, 17, 3, 128))
    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        lambda *args: (torch.zeros_like(query), torch.zeros_like(key)),
    )
    with torch.inference_mode():
        first = fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin, _apply_rotary_emb)
    assert first is not None
    assert torch.equal(first[0], _apply_rotary_emb(query, freqs_cos, freqs_sin))
    assert torch.equal(first[1], _apply_rotary_emb(key, freqs_cos, freqs_sin))
    assert len(fused_rope._FAILED_KEYS) == 1

    def unexpected_launch(*args):
        raise AssertionError("failed geometry must not launch again")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        unexpected_launch,
    )
    with torch.inference_mode():
        second = fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin, _apply_rotary_emb)
    assert second is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_compile_path_does_not_launch_fused_kernel(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((1, 17, 3, 128))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def unexpected_launch(*args):
        raise AssertionError("compile path must remain native")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        unexpected_launch,
    )
    with torch.inference_mode():
        assert fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin, _apply_rotary_emb) is None

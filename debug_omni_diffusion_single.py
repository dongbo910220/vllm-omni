#!/usr/bin/env python3
"""
Debug 脚本（离线推理）：Z-Image (Single GPU, no SP)

用途：
- 用来区分“模型本身/环境问题” vs “SP(hybrid) 引入的问题”
- 走 vLLM-Omni 的真实链路：Engine -> Executor -> Worker -> Pipeline

运行：
  source .venv/bin/activate
  python debug_omni_diffusion_single.py

输出：
  ./outputs/debug_zimage_single.png
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Force SDPA to use math kernel (avoid mem_efficient/cutlass kernels on V100).
# NOTE: This runs in worker processes too under multiprocessing spawn.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

    model_path = "/root/autodl-tmp/hf-downloads/Tongyi-MAI/Z-Image-Turbo"
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Model path not found: {model_path}")

    # Prefer bf16 for numerical stability on Z-Image Turbo (fp16 can produce NaNs).
    dtype = torch.bfloat16

    parallel = DiffusionParallelConfig(
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        cfg_parallel_size=1,
    )

    od = OmniDiffusion(
        model=model_path,
        parallel_config=parallel,
        dtype=dtype,
        attention_backend="sdpa",
        enforce_eager=True,
    )

    prompt = "a cute cat wearing a hat, high quality, detailed"
    params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=5.0,
        seed=0,
        num_outputs_per_prompt=1,
    )

    out = od.generate(prompt, params)
    img = out[0].images[0]

    out_path = Path("outputs/debug_zimage_single.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[DebugOmni] Saved -> {out_path.resolve()}", flush=True)

    od.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug 脚本（离线推理）：Z-Image + Hybrid SP (Ulysses + Ring)

为什么推荐用这个脚本 Debug：
- 不依赖 OpenAI HTTP server（少一层 FastAPI/uvicorn 干扰）
- 仍然走 vLLM-Omni 的“真实执行链路”：Engine -> Executor -> Worker -> Pipeline
- 默认开 4 卡，并启用 hybrid SP（ulysses_degree=2, ring_degree=2）

运行：
  python debug_omni_diffusion_hybrid.py

输出图片：
  ./outputs/debug_zimage_hybrid_sp.png
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 让 ring attention 走 PyTorch SDPA（便于 Debug；无 FA 依赖）
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

    model_path = "/root/autodl-tmp/hf-downloads/Tongyi-MAI/Z-Image-Turbo"
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Model path not found: {model_path}")

    # Z-Image Turbo 在 fp16 上容易出现数值问题（NaN -> 出图全黑）。
    # 即使在 V100 上，使用 bf16 也更稳定；Ring Attention 内部会在必要时
    # 将 bf16 attention 输入临时 cast 到 fp16（见 ring_kernels.py），避免
    # Volta 上 bf16 efficient SDPA kernel 缺失导致的报错。
    dtype = torch.bfloat16

    # 4 卡 hybrid：ulysses_degree * ring_degree = 4
    parallel = DiffusionParallelConfig(
        ulysses_degree=2,
        ring_degree=2,
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

    out_path = Path("outputs/debug_zimage_hybrid_sp.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[DebugOmni] Saved -> {out_path.resolve()}", flush=True)

    od.close()


if __name__ == "__main__":
    main()

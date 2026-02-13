#!/usr/bin/env python3
"""
Debug 启动脚本（vLLM-Omni / Diffusion OpenAI Server）

目标：
- 方便在 PyCharm 里 Debug（避免 uvloop / CUDA Graph 干扰断点）
- 启动 OpenAI 兼容的 HTTP Server，并支持 /v1/images/generations
- 默认使用 “Hybrid SP: Ulysses + Ring”（ulysses_degree > 1 且 ring_degree > 1）

用法：
1) 直接运行（启动 server）：
   python debug_omni_diffusion_server.py

2) 启动后发请求（另开终端）：
   curl http://127.0.0.1:8091/v1/models
   curl http://127.0.0.1:8091/v1/images/generations \\
     -H 'Content-Type: application/json' \\
     -d '{"prompt":"a cute cat wearing a hat","n":1,"size":"512x512","num_inference_steps":8,"guidance_scale":1.0,"seed":0}'
"""

from __future__ import annotations

import asyncio
import os
import sys
from argparse import Namespace


def _default_model_path() -> str:
    # Prefer local weights to avoid downloads.
    local = "/root/autodl-tmp/hf-downloads/Tongyi-MAI/Z-Image-Turbo"
    return local


def _build_serve_args() -> Namespace:
    # Disable uvloop (must be set before any vLLM imports that may install uvloop).
    os.environ.setdefault("VLLM_USE_UVLOOP", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Z-Image Turbo 在 fp16 上容易出现数值问题（NaN -> 出图全黑）。
    # 即使在 V100 上，使用 bf16 也更稳定；Ring Attention 内部会在必要时
    # 将 bf16 attention 输入临时 cast 到 fp16（见 ring_kernels.py）。
    dtype = "bfloat16"

    # Build argv using upstream vLLM OpenAI server parser (to get all defaults).
    sys.argv = [
        "debug_omni_diffusion_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8091",
        "--model",
        _default_model_path(),
        "--dtype",
        dtype,
        "--enforce-eager",  # ✅ 禁用 torch.compile / CUDA Graph，便于断点调试
    ]

    # Use the same parser as `vllm serve`, then inject omni/diffusion-only knobs.
    from vllm.entrypoints.openai.api_server import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    cli_env_setup()

    parser = FlexibleArgumentParser(description="vLLM-Omni Diffusion OpenAI Server (debug)")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Force Omni mode and diffusion SP settings (these fields are consumed by AsyncOmni).
    setattr(args, "omni", True)

    # Hybrid SP: ulysses_degree * ring_degree = sequence_parallel_size
    setattr(args, "ulysses_degree", 2)
    setattr(args, "ring_degree", 2)

    # Keep it simple for debugging.
    setattr(args, "tensor_parallel_size", 1)
    setattr(args, "cfg_parallel_size", 1)

    # Optional: explicit stage configs if you have one; default is None.
    setattr(args, "stage_configs_path", None)

    # Reduce noise while debugging.
    setattr(args, "disable_uvicorn_access_log", True)
    return args


def main() -> None:
    from vllm_omni.entrypoints.openai.api_server import omni_run_server

    args = _build_serve_args()
    print(
        "[DebugOmni] Starting server with:\n"
        f"  model={getattr(args, 'model', None)}\n"
        f"  host={getattr(args, 'host', None)} port={getattr(args, 'port', None)}\n"
        f"  ulysses_degree={getattr(args, 'ulysses_degree', None)} ring_degree={getattr(args, 'ring_degree', None)}\n"
        f"  dtype={getattr(args, 'dtype', None)} enforce_eager={getattr(args, 'enforce_eager', None)}\n"
        f"  VLLM_USE_UVLOOP={os.environ.get('VLLM_USE_UVLOOP')}",
        flush=True,
    )

    asyncio.run(omni_run_server(args))


if __name__ == "__main__":
    main()

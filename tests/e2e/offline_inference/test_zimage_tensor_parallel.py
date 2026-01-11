# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.utils.platform_utils import is_npu, is_rocm

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _create_random_zimage_model(model_dir: Path) -> str:
    """Create a tiny, random-weight Z-Image model for CI.

    This avoids downloading / loading the full Tongyi-MAI/Z-Image-Turbo weights
    under the tight diffusion-parallelism CI timeout.
    """
    model_index_path = model_dir / "model_index.json"
    if model_index_path.exists():
        return str(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "scheduler").mkdir(parents=True, exist_ok=True)
    (model_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (model_dir / "vae").mkdir(parents=True, exist_ok=True)

    model_index = {
        "_class_name": "ZImagePipeline",
        "_diffusers_version": "0.36.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.z_image.z_image_transformer import (
        ZImageTransformer2DModel,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.save_pretrained(model_dir / "scheduler")

    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32, 64),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
    )
    vae.save_pretrained(model_dir / "vae", safe_serialization=True)

    transformer_cfg = {
        "_class_name": "ZImageTransformer2DModel",
        "_diffusers_version": "0.36.0",
        "all_f_patch_size": [1],
        "all_patch_size": [2],
        "axes_dims": [16, 16, 32],  # sum == head_dim (768 / 12 = 64)
        "axes_lens": [512, 128, 128],
        "cap_feat_dim": 32,
        "dim": 768,
        "in_channels": 4,
        "n_heads": 12,
        "n_kv_heads": 12,
        "n_layers": 8,
        "n_refiner_layers": 1,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
    }
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps(transformer_cfg, indent=2) + "\n",
        encoding="utf-8",
    )

    # Z-Image uses vLLM TP-aware linear layers which require model-parallel
    # groups to be initialized, even for TP=1. Initialize a 1-rank group to
    # create and save a deterministic random checkpoint.
    import tempfile

    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    try:
        if not model_parallel_is_initialized():
            with tempfile.TemporaryDirectory() as tmpdir:
                init_path = Path(tmpdir) / "dist_init"
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=f"file://{init_path}",
                    backend="gloo",
                )
                ensure_model_parallel_initialized(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    backend="gloo",
                )

        torch.manual_seed(0)
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(transformer_cfg["all_patch_size"]),
            all_f_patch_size=tuple(transformer_cfg["all_f_patch_size"]),
            in_channels=transformer_cfg["in_channels"],
            dim=transformer_cfg["dim"],
            n_layers=transformer_cfg["n_layers"],
            n_refiner_layers=transformer_cfg["n_refiner_layers"],
            n_heads=transformer_cfg["n_heads"],
            n_kv_heads=transformer_cfg["n_kv_heads"],
            norm_eps=transformer_cfg["norm_eps"],
            qk_norm=transformer_cfg["qk_norm"],
            cap_feat_dim=transformer_cfg["cap_feat_dim"],
            rope_theta=transformer_cfg["rope_theta"],
            t_scale=transformer_cfg["t_scale"],
            axes_dims=transformer_cfg["axes_dims"],
            axes_lens=transformer_cfg["axes_lens"],
        )

        # Save in "unstacked" format so `ZImageTransformer2DModel.load_weights`
        # can map (to_q, to_k, to_v) -> to_qkv and (w1, w3) -> w13.
        dim = int(transformer_cfg["dim"])
        n_heads = int(transformer_cfg["n_heads"])
        n_kv_heads = int(transformer_cfg["n_kv_heads"])
        head_dim = dim // n_heads
        q_out = n_heads * head_dim
        kv_out = n_kv_heads * head_dim
        hidden_dim = int(dim / 3 * 8)

        unstacked: dict[str, torch.Tensor] = {}
        for name, tensor in transformer.state_dict().items():
            if name.endswith("attention.to_qkv.weight"):
                q, k, v = torch.split(tensor, [q_out, kv_out, kv_out], dim=0)
                unstacked[name.replace("to_qkv.weight", "to_q.weight")] = q
                unstacked[name.replace("to_qkv.weight", "to_k.weight")] = k
                unstacked[name.replace("to_qkv.weight", "to_v.weight")] = v
                continue
            if name.endswith("feed_forward.w13.weight"):
                w1, w3 = torch.split(tensor, [hidden_dim, hidden_dim], dim=0)
                unstacked[name.replace("w13.weight", "w1.weight")] = w1
                unstacked[name.replace("w13.weight", "w3.weight")] = w3
                continue
            unstacked[name] = tensor

        state_dict = {k: v.detach().cpu() for k, v in unstacked.items()}
        save_file(state_dict, str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    finally:
        cleanup_dist_env_and_memory()
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(key, None)

    return str(model_dir)


def _get_zimage_model(tmp_path: Path) -> str:
    override = os.environ.get("VLLM_TEST_ZIMAGE_MODEL")
    if override:
        return override

    # Default to a local random model to keep CI self-contained and fast.
    return _create_random_zimage_model(tmp_path / "zimage_random_model")


def _pil_to_float_rgb_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    ta = _pil_to_float_rgb_tensor(a)
    tb = _pil_to_float_rgb_tensor(b)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


def _get_images(output):
    if hasattr(output, "images") and output.images:
        return output.images

    if getattr(output, "request_output", None) is None:
        return None

    if isinstance(output.request_output, list) and len(output.request_output) == 0:
        return None

    item = output.request_output[0]
    while hasattr(item, "output") and not hasattr(item, "images"):
        item = item.output
        if item is None:
            return None

    if isinstance(item, dict):
        return item.get("images")
    return getattr(item, "images", None)


def _extract_single_image(outputs) -> Image.Image:
    first_output = outputs[0]
    images = _get_images(first_output)
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _write_tp_stage_config_yaml(path: Path, *, model: str, devices: str, tp_size: int) -> None:
    path.write_text(
        "\n".join(
            [
                "stage_args:",
                "  - stage_id: 0",
                "    stage_type: diffusion",
                "    runtime:",
                "      process: true",
                f'      devices: "{devices}"',
                "      max_batch_size: 1",
                "    engine_args:",
                f'      model: "{model}"',
                "      parallel_config:",
                "        pipeline_parallel_size: 1",
                "        data_parallel_size: 1",
                f"        tensor_parallel_size: {tp_size}",
                "        sequence_parallel_size: 1",
                "        ulysses_degree: 1",
                "        ring_degree: 1",
                "        cfg_parallel_size: 1",
                "      cache_backend: none",
                "      cache_config: null",
                "      model_stage: diffusion",
                "    final_output: true",
                "    final_output_type: image",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _needs_dummy_prompt_embeds(model: str) -> bool:
    if not os.path.isdir(model):
        return False
    return not (os.path.isdir(os.path.join(model, "tokenizer")) and os.path.isdir(os.path.join(model, "text_encoder")))


def _get_cuda_vram_used_total_gib(devices: list[int]) -> dict[int, tuple[float, float]]:
    from vllm.platforms import current_platform

    from tests.utils import get_physical_device_indices

    if not current_platform.is_cuda():
        return {}

    from vllm.third_party.pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    devices = get_physical_device_indices(devices)
    nvmlInit()
    try:
        out: dict[int, tuple[float, float]] = {}
        for device in devices:
            handle = nvmlDeviceGetHandleByIndex(device)
            mem = nvmlDeviceGetMemoryInfo(handle)
            out[device] = (mem.used / 2**30, mem.total / 2**30)
        return out
    finally:
        nvmlShutdown()


def _run_zimage_generate(
    *,
    tmp_path: Path,
    model: str,
    devices: str,
    tp_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
) -> tuple[Image.Image, float, dict[int, tuple[float, float]]]:
    stage_cfg_path = tmp_path / f"zimage_tp{tp_size}.yaml"
    _write_tp_stage_config_yaml(
        stage_cfg_path,
        model=model,
        devices=devices,
        tp_size=tp_size,
    )

    m = Omni(model=model, stage_configs_path=str(stage_cfg_path))
    try:
        # NOTE: Omni closes itself when a generate() call is exhausted.
        # For perf, measure time to produce N outputs (excluding teardown).
        num_requests = 1

        generate_kwargs = dict(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            seed=seed,
            num_outputs_per_prompt=1,
            py_generator=True,
        )
        if _needs_dummy_prompt_embeds(model):
            cap_feat_dim = 32
            torch.manual_seed(0)
            base = torch.randn(1, cap_feat_dim, dtype=torch.bfloat16)
            generate_kwargs["prompt_embeds"] = [base.clone() for _ in range(num_requests)]

        dev_list = [int(d) for d in devices.split(",") if d]
        vram_before = _get_cuda_vram_used_total_gib(dev_list)

        gen = m.generate([PROMPT] * num_requests, **generate_kwargs)

        t0 = time.perf_counter()
        last_output = None
        for _ in range(num_requests):
            last_output = next(gen)
        t1 = time.perf_counter()

        assert last_output is not None
        avg_time_s = (t1 - t0) / num_requests
        img = _extract_single_image([last_output])
        vram_after = _get_cuda_vram_used_total_gib(dev_list)

        # Ensure the generator is fully consumed so it can clean up.
        for _ in gen:
            pass

        return img, avg_time_s, (vram_after or vram_before)
    finally:
        m.close()
        cleanup_dist_env_and_memory()
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(key, None)
        time.sleep(2)


@pytest.mark.integration
def test_zimage_tensor_parallel_tp2(tmp_path: Path):
    if is_npu() or is_rocm():
        pytest.skip("Z-Image TP e2e test is only supported on CUDA for now.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Z-Image TP=2 requires >= 2 CUDA devices.")

    height = 256
    width = 256
    num_inference_steps = 2
    seed = 42

    model = _get_zimage_model(tmp_path)

    tp1_img, tp1_time_s, tp1_vram = _run_zimage_generate(
        tmp_path=tmp_path,
        model=model,
        devices="0",
        tp_size=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    tp2_img, tp2_time_s, tp2_vram = _run_zimage_generate(
        tmp_path=tmp_path,
        model=model,
        devices="0,1",
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    tp1_path = tmp_path / "zimage_tp1.png"
    tp2_path = tmp_path / "zimage_tp2.png"
    tp1_img.save(tp1_path)
    tp2_img.save(tp2_path)

    assert tp1_img.width == width and tp1_img.height == height
    assert tp2_img.width == width and tp2_img.height == height

    mean_abs_diff, max_abs_diff = _diff_metrics(tp1_img, tp2_img)
    mean_threshold = 3e-2
    max_threshold = 3.5e-1
    print(
        "Z-Image TP image diff stats (TP=1 vs TP=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"tp1_img={tp1_path}, tp2_img={tp2_path}"
    )
    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e})"
    )

    print(f"Z-Image TP perf (lower is better): tp1_time_s={tp1_time_s:.6f}, tp2_time_s={tp2_time_s:.6f}")
    print(f"Z-Image TP vram (GiB): tp1={tp1_vram}, tp2={tp2_vram}")

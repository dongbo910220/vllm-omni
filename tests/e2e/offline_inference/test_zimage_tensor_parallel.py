# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import is_npu, is_rocm

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _is_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("BUILDKITE"))


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

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
    )
    vae.save_pretrained(model_dir / "vae", safe_serialization=True)

    # Tiny Z-Image transformer that supports TP=2.
    transformer_cfg = {
        "_class_name": "ZImageTransformer2DModel",
        "_diffusers_version": "0.36.0",
        "all_f_patch_size": [1],
        "all_patch_size": [2],
        "axes_dims": [2, 2, 2],  # sum == head_dim (48 / 8 = 6)
        "axes_lens": [512, 128, 128],
        "cap_feat_dim": 32,
        "dim": 48,
        "in_channels": 4,
        "n_heads": 8,
        "n_kv_heads": 8,
        "n_layers": 2,
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
    state_dict = {k: v.detach().cpu() for k, v in transformer.state_dict().items()}
    save_file(state_dict, str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"))

    return str(model_dir)


def _get_zimage_model(tmp_path: Path) -> str:
    # Allow overriding the model for local/offline environments.
    # Can be either a HuggingFace repo id or a local path.
    override = os.environ.get("VLLM_TEST_ZIMAGE_MODEL")
    if override:
        return override

    if _is_ci():
        return _create_random_zimage_model(tmp_path / "zimage_random_model")

    return "Tongyi-MAI/Z-Image-Turbo"


def _extract_single_image(outputs) -> Image.Image:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _write_tp_stage_config_yaml(path: Path, *, model: str, devices: str, tp_size: int) -> None:
    # NOTE: Z-Image can ship with a built-in stage config ("z_image.yaml") which
    # defaults to single GPU + TP=1. For TP e2e, force our own stage config.
    path.write_text(
        "\n".join(
            [
                "stage_args:",
                "  - stage_id: 0",
                "    stage_type: diffusion",
                "    runtime:",
                "      process: true",
                f"      devices: \"{devices}\"",
                "      max_batch_size: 1",
                "    engine_args:",
                f"      model: \"{model}\"",
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
    stage_cfg_path = tmp_path / "zimage_tp2.yaml"
    _write_tp_stage_config_yaml(
        stage_cfg_path,
        model=model,
        devices="0,1",
        tp_size=2,
    )
    m = Omni(model=model, stage_configs_path=str(stage_cfg_path))
    try:
        generate_kwargs = dict(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            seed=seed,
            num_outputs_per_prompt=1,
        )
        if _is_ci() and not os.environ.get("VLLM_TEST_ZIMAGE_MODEL"):
            # CI uses a tiny random Z-Image model without a text encoder/tokenizer.
            # Provide a dummy prompt embedding to skip prompt encoding.
            generate_kwargs["prompt_embeds"] = [torch.randn(1, 32)]

        outputs = m.generate(PROMPT, **generate_kwargs)
        tp2_img = _extract_single_image(outputs)
    finally:
        m.close()

    tp2_path = tmp_path / "zimage_tp2.png"
    tp2_img.save(tp2_path)

    assert tp2_img.width == width and tp2_img.height == height

    # Keep TP=1 vs TP=2 perf/quality comparison out of CI gating to avoid
    # flakes and potential OOMs on smaller GPUs. Use local benchmarks instead.

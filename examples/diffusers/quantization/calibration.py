# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from diffusers.utils import load_image
from models_utils import MODEL_DEFAULTS, ModelType
from pipeline_manager import PipelineManager
from quantize_config import CalibrationConfig
from tqdm import tqdm
from utils import load_calib_prompts


class Calibrator:
    """Handles model calibration for quantization."""

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        config: CalibrationConfig,
        model_type: ModelType,
        logger: logging.Logger,
    ):
        """
        Initialize calibrator.

        Args:
            pipeline_manager: Pipeline manager with main and upsampler pipelines
            config: Calibration configuration
            model_type: Type of model being calibrated
            logger: Logger instance
        """
        self.pipeline_manager = pipeline_manager
        self.pipe = pipeline_manager.pipe
        self.pipe_upsample = pipeline_manager.pipe_upsample
        self.config = config
        self.model_type = model_type
        self.logger = logger

    def load_and_batch_prompts(self) -> list[list[str]]:
        """
        Load calibration prompts from file.

        Returns:
            List of batched calibration prompts
        """
        self.logger.info(f"Loading calibration prompts from {self.config.prompts_dataset}")
        if isinstance(self.config.prompts_dataset, Path):
            return load_calib_prompts(
                self.config.batch_size,
                self.config.prompts_dataset,
            )

        return load_calib_prompts(
            self.config.batch_size,
            self.config.prompts_dataset["name"],
            self.config.prompts_dataset["split"],
            self.config.prompts_dataset["column"],
        )

    def load_edit_examples(self) -> list[dict[str, Any]]:
        """Load image-edit calibration records from a JSON metadata file."""
        if self.config.edit_metadata is None:
            raise RuntimeError("edit_metadata is not configured")
        metadata_path = self.config.edit_metadata
        with metadata_path.open() as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = records.get("records", records.get("data"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {metadata_path}")

        examples = []
        for index, record in enumerate(records[: self.config.calib_size]):
            if not isinstance(record, dict) or not record.get("prompt"):
                raise ValueError(f"Record {index} in {metadata_path} needs a non-empty 'prompt'")
            image_value = next(
                (
                    record.get(key)
                    for key in (
                        "edit_image",
                        "input_image",
                        "input_images",
                        "condition_images",
                        "images",
                    )
                    if record.get(key) is not None
                ),
                None,
            )
            if image_value is None:
                raise ValueError(
                    f"Record {index} in {metadata_path} needs 'edit_image', 'input_images', or 'images'"
                )
            image_paths = [image_value] if isinstance(image_value, str) else image_value
            if not isinstance(image_paths, list) or not image_paths:
                raise ValueError(f"Record {index} in {metadata_path} has no input images")
            images = []
            for image_path in image_paths:
                path = Path(image_path)
                if not path.is_absolute():
                    path = metadata_path.parent / path
                if not path.exists():
                    raise FileNotFoundError(f"Missing image for record {index}: {path}")
                images.append(load_image(str(path)))
            examples.append({"prompt": record["prompt"], "images": images})
        if not examples:
            raise ValueError(f"No calibration records found in {metadata_path}")
        self.logger.info(
            "Loaded %d image-edit records (%d condition images in the first record)",
            len(examples),
            len(examples[0]["images"]),
        )
        return examples

    def load_calibration_data(self) -> list[Any]:
        """Load either text prompts or image-edit records."""
        if self.config.edit_metadata is not None:
            return self.load_edit_examples()
        return self.load_and_batch_prompts()

    def run_calibration(self, calibration_data: list[Any]) -> None:
        """
        Run calibration steps on the pipeline.

        Args:
            batched_prompts: List of batched calibration prompts
        """
        if self.config.edit_metadata is not None:
            self._run_edit_calibration(calibration_data)
            return

        self.logger.info(f"Starting calibration with {self.config.num_batches} batches")
        extra_args = MODEL_DEFAULTS.get(self.model_type, {}).get("inference_extra_args", {})

        with tqdm(total=self.config.num_batches, desc="Calibration", unit="batch") as pbar:
            for i, prompt_batch in enumerate(calibration_data):
                if i >= self.config.num_batches:
                    break

                if self.model_type == ModelType.LTX2:
                    self._run_ltx2_calibration(prompt_batch, extra_args)
                elif self.model_type == ModelType.LTX_VIDEO_DEV:
                    # Special handling for LTX-Video
                    self._run_ltx_video_calibration(prompt_batch, extra_args)
                elif self.model_type in [ModelType.WAN22_T2V_14b, ModelType.WAN22_T2V_5b]:
                    # Special handling for WAN video models
                    self._run_wan_video_calibration(prompt_batch, extra_args)
                else:
                    common_args = {
                        "prompt": prompt_batch,
                        "num_inference_steps": self.config.n_steps,
                    }
                    self.pipe(**common_args, **extra_args).images
                pbar.update(1)
                self.logger.debug(f"Completed calibration batch {i + 1}/{self.config.num_batches}")
        self.logger.info("Calibration completed successfully")

    def _run_edit_calibration(self, examples: list[dict[str, Any]]) -> None:
        """Calibrate Qwen-Image with one or more condition images per record."""
        if self.model_type != ModelType.QWEN_IMAGE:
            raise ValueError("Image-edit metadata calibration currently supports qwen-image only")
        extra_args = dict(MODEL_DEFAULTS.get(self.model_type, {}).get("inference_extra_args", {}))
        # Let Qwen-Image derive dimensions from each condition image's aspect ratio.
        extra_args.pop("height", None)
        extra_args.pop("width", None)
        sample_sigmas = getattr(self.pipe.config, "sample_sigmas", None)
        if sample_sigmas is not None:
            sample_sigmas = list(sample_sigmas)
            if len(sample_sigmas) != self.config.n_steps:
                raise ValueError(
                    "The pipeline sample_sigmas schedule has "
                    f"{len(sample_sigmas)} steps, but --n-steps is {self.config.n_steps}. "
                    "Set --n-steps to the checkpoint schedule length."
                )
            extra_args["sigmas"] = sample_sigmas
            self.logger.info(
                "Using checkpoint sample_sigmas for image-edit calibration: %s", sample_sigmas
            )
        self.logger.info(
            "Starting image-edit calibration with %d records and %d denoising steps",
            len(examples),
            self.config.n_steps,
        )
        with tqdm(total=len(examples), desc="Image-edit calibration", unit="sample") as pbar:
            for example in examples:
                self.pipe(
                    prompt=example["prompt"],
                    image=example["images"],
                    num_inference_steps=self.config.n_steps,
                    output_type="latent",
                    **extra_args,
                )
                pbar.update(1)
        self.logger.info("Image-edit calibration completed successfully")

    def _run_wan_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        extra_params = self.pipeline_manager.config.extra_params
        kwargs = {}
        kwargs["negative_prompt"] = extra_args["negative_prompt"]
        kwargs["height"] = extra_params.get("height", extra_args["height"])
        kwargs["width"] = extra_params.get("width", extra_args["width"])
        kwargs["num_frames"] = extra_params.get("num_frames", extra_args["num_frames"])
        kwargs["guidance_scale"] = extra_args["guidance_scale"]
        if "guidance_scale_2" in extra_args:
            kwargs["guidance_scale_2"] = extra_args["guidance_scale_2"]
        kwargs["num_inference_steps"] = self.config.n_steps

        self.pipe(prompt=prompt_batch, **kwargs).frames

    def _run_ltx2_calibration(self, prompt_batch: list[str], extra_args: dict[str, Any]) -> None:
        warnings.warn(
            "LTX-2 packages (ltx-core, ltx-pipelines, ltx-trainer) are provided by Lightricks and are NOT "
            "covered by the Apache 2.0 license governing NVIDIA Model Optimizer. You MUST comply "
            "with the LTX Community License Agreement when installing and using LTX-2 with NVIDIA "
            "Model Optimizer. Any derivative models or fine-tuned weights from LTX-2 remain "
            "subject to the LTX Community License Agreement, not Apache 2.0. "
            "See: https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
            UserWarning,
            stacklevel=2,
        )
        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils.constants import (
            DEFAULT_AUDIO_GUIDER_PARAMS,
            DEFAULT_VIDEO_GUIDER_PARAMS,
        )

        prompt = prompt_batch[0]
        extra_params = self.pipeline_manager.config.extra_params
        kwargs = {
            "negative_prompt": extra_args.get(
                "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
            ),
            "seed": extra_params.get("seed", 0),
            "height": extra_params.get("height", extra_args.get("height", 1024)),
            "width": extra_params.get("width", extra_args.get("width", 1536)),
            "num_frames": extra_params.get("num_frames", extra_args.get("num_frames", 121)),
            "frame_rate": extra_params.get("frame_rate", extra_args.get("frame_rate", 24.0)),
            "num_inference_steps": self.config.n_steps,
            "video_guider_params": DEFAULT_VIDEO_GUIDER_PARAMS,
            "audio_guider_params": DEFAULT_AUDIO_GUIDER_PARAMS,
            "images": extra_params.get("images", []),
            "tiling_config": extra_params.get("tiling_config", TilingConfig.default()),
        }
        decoded_video, decoded_audio = self.pipe(prompt=prompt, **kwargs)
        # vae_decode_video returns a lazy generator — consume it so the
        # video decoder's forward() actually runs during calibration.
        for _ in decoded_video:
            pass

    def _run_ltx_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        """
        Run calibration for LTX-Video model using the full multi-stage pipeline.

        Args:
            prompt_batch: Batch of prompts
            extra_args: Model-specific arguments
        """
        # Extract specific args for LTX-Video
        expected_height = extra_args.get("height", 512)
        expected_width = extra_args.get("width", 704)
        num_frames = extra_args.get("num_frames", 121)
        negative_prompt = extra_args.get(
            "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
        )

        def round_to_nearest_resolution_acceptable_by_vae(height, width):
            height = height - (height % self.pipe.vae_spatial_compression_ratio)
            width = width - (width % self.pipe.vae_spatial_compression_ratio)
            return height, width

        downscale_factor = 2 / 3
        # Part 1: Generate video at smaller resolution
        downscaled_height, downscaled_width = (
            int(expected_height * downscale_factor),
            int(expected_width * downscale_factor),
        )
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
            downscaled_height, downscaled_width
        )

        # Generate initial latents at lower resolution
        latents = self.pipe(
            conditions=None,
            prompt=prompt_batch,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=self.config.n_steps,
            output_type="latent",
        ).frames

        # Part 2: Upscale generated video using latent upsampler (if available)
        if self.pipe_upsample is not None:
            _ = self.pipe_upsample(latents=latents, output_type="latent").frames

            # Part 3: Denoise the upscaled video with few steps to improve texture
            # However, in this example code, we will omit the upscale step since its optional.

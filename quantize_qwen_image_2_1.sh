#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 修改这里即可：metadata.json 中每条记录包含 prompt 和 edit_image/input_images。
# Qwen-Image-2.1-Turbo 是 8 步蒸馏 checkpoint；校准使用其 model_index.json
# 中的 sample_sigmas，并完整执行 8 次 DiT forward。
MODEL_PATH="${MODEL_PATH:-/models/Qwen-Image-2.1-Turbo}"
EDIT_METADATA="${EDIT_METADATA:-/datasets/codes_zsqiao/image_gen/layered_dataset/train/training_aug_all/metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-workspaces/qwen-image-2.1-turbo-nvfp4}"
CALIB_SIZE="${CALIB_SIZE:-128}"
N_STEPS="${N_STEPS:-8}"           # Turbo checkpoint 的 sample_sigmas 长度
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/datasets/codes_zsqiao/training_infra/diffusers/src}"
if [[ ! -f "$DIFFUSERS_SRC/diffusers/__init__.py" ]]; then
    echo "Local Diffusers source not found: $DIFFUSERS_SRC" >&2
    exit 1
fi
export PYTHONPATH="$DIFFUSERS_SRC:examples/diffusers/quantization${PYTHONPATH:+:$PYTHONPATH}"

# 依赖：当前仓库安装的 ModelOpt，以及提供 QwenImage21Pipeline 的 diffusers。
# 复用仓库校准/导出入口；下面的 Python 仅适配 2.1 和量化范围。
# A800 上用 BF16 前向收集校准数据，导出 NVFP4，不启动推理服务。
python - \
    --model qwen-image \
    --override-model-path "$MODEL_PATH" \
    --model-dtype BFloat16 \
    --trt-high-precision-dtype BFloat16 \
    --format fp4 --quant-algo max --block-size 16 \
    --batch-size 1 --calib-size "$CALIB_SIZE" --n-steps "$N_STEPS" \
    --edit-metadata "$EDIT_METADATA" \
    --hf-ckpt-dir "$OUTPUT_DIR/hf" "$@" <<'PY'
import models_utils
import quantize
from diffusers import QwenImage21Pipeline

model_type = models_utils.ModelType.QWEN_IMAGE
models_utils.MODEL_PIPELINE[model_type] = QwenImage21Pipeline
models_utils.MODEL_DEFAULTS[model_type]["block_range"] = {
    "exclude_first_n": 0,
    "exclude_last_n": 0,
    "block_module": "transformer_blocks",
}

# 同时覆盖校准后的旧版 Qwen-Image filter，避免再次排除前两个 block。
def keep_high_precision(name):
    return not name.startswith("transformer_blocks.") or "norm" in name


models_utils._FILTER_FUNC_MAP[model_type] = keep_high_precision
original_get_quant_config = quantize.Quantizer.get_quant_config


def get_quant_config(self, n_steps, backbone):
    config = original_get_quant_config(self, n_steps, backbone)
    config["quant_cfg"].append(
        {"quantizer_name": "*norm*", "enable": False}
    )
    return config


# 32 个 block 的全部 Linear 做 W4A4；Norm、block 外层、VAE、文本编码器不量化。
quantize.Quantizer.get_quant_config = get_quant_config
quantize.main()
PY

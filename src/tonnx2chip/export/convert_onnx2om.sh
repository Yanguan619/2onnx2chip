#!/bin/bash
# Convert ONNX models from qwen35_onnx.py export to OM format for Ascend NPU
#
# Usage:
#   ./convert_onnx2om.sh                              # use defaults
#   ./convert_onnx2om.sh <onnx_dir> <om_dir> <soc_version>
#   bash convert_onnx2om.sh  /data/workspace/weight/Qwen3.5-2B-Edge/onnx-output-opt/  /data/workspace/weight/Qwen3.5-2B-Edge/om-310p Ascend310P7
#
# Defaults:
#   ONNX_DIR:    /data2/qwen-onnx/output/onnx20260618
#   OM_DIR:      /data2/qwen-onnx/output/om20260618
#   SOC_VERSION: Ascend910B4-1

# FAQ
# ln -s /usr/local/python3.11.10/bin/python3.11-config /usr/local/python3.11.10/bin/python3-config

set -euo pipefail

ONNX_DIR="${1:-output/onnx-output-opt/}"
OM_DIR="${2:-output/om/}"
SOC_VERSION="${3:-Ascend910_9362}"

mkdir -p "$OM_DIR"
echo "════════════════════════"
echo "ONNX_DIR:    $ONNX_DIR"
echo "OM_DIR:      $OM_DIR"
echo "SOC_VERSION: $SOC_VERSION"
echo "════════════════════════"

# ---- Helper: probe ONNX input shapes from model file ----
# For dynamic dims (dim_value=0 or dim_param non-empty), output "-1" so ATC
# treats them as dynamic.
probe_shapes() {
    local model="$1"
    python3 -c "
import sys

import onnx

m = onnx.load('$model')
pairs = []
for inp in m.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(str(d.dim_value))
        else:
            dims.append('-1')
    pairs.append(inp.name + ':' + ','.join(dims))
print(';'.join(pairs))
"
}

# ---- Conversion commands ----
VIT_PATH=$ONNX_DIR/vision_encoder/vision_encoder_final.onnx
CMD_VIT="atc --model $VIT_PATH \
    --output $OM_DIR/vision_encoder \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape='$(probe_shapes "$VIT_PATH")' \
    --keep_dtype=config/keep_dtype.cfg \
    --device 0"

EMBED_PATH=$ONNX_DIR/embedding/embedding_final.onnx
CMD_EMBED="atc --model $EMBED_PATH \
    --output $OM_DIR/embedding \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape='$(probe_shapes "$ONNX_DIR/embedding/embedding_final.onnx")' \
    --device 0"

# output 多一层级是因为保存了外置权重且外置权重路径必定在weight目录下
# multimodal_attention_mask:1,256;position_ids:3,1,256;multimodal_embeddings:1,256,2048
PREFILL_PATH=$ONNX_DIR/decoder_model_prefill/decoder_model_prefill_final.onnx
CMD_PREFILL="ASCEND_SLOG_PRINT_TO_STDOUT=0 atc --model $PREFILL_PATH \
    --output $OM_DIR/decoder_model_prefill/decoder_model_prefill \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape='$(probe_shapes "$PREFILL_PATH")' \
    --precision_mode=force_fp32 --external_weight=1 \
    --device 0"

# output 多一层级是因为保存了外置权重且外置权重路径必定在weight目录下，避免与decoder_prefill重复
# inputs_embeds:1,1,2048;multimodal_attention_mask:1,257;position_ids:1,1;past_state_0_conv:1,6144,4;past_state_0_rec:1,16,128,128;past_state_1_conv:1,6144,4;past_state_1_rec:1,16,128,128;past_state_2_conv:1,6144,4;past_state_2_rec:1,16,128,128;past_3_key:1,2,256,256;past_3_value:1,2,256,256;past_state_4_conv:1,6144,4;past_state_4_rec:1,16,128,128;past_state_5_conv:1,6144,4;past_state_5_rec:1,16,128,128;past_state_6_conv:1,6144,4;past_state_6_rec:1,16,128,128;past_7_key:1,2,256,256;past_7_value:1,2,256,256;past_state_8_conv:1,6144,4;past_state_8_rec:1,16,128,128;past_state_9_conv:1,6144,4;past_state_9_rec:1,16,128,128;past_state_10_conv:1,6144,4;past_state_10_rec:1,16,128,128;past_11_key:1,2,256,256;past_11_value:1,2,256,256;past_state_12_conv:1,6144,4;past_state_12_rec:1,16,128,128;past_state_13_conv:1,6144,4;past_state_13_rec:1,16,128,128;past_state_14_conv:1,6144,4;past_state_14_rec:1,16,128,128;past_15_key:1,2,256,256;past_15_value:1,2,256,256;past_state_16_conv:1,6144,4;past_state_16_rec:1,16,128,128;past_state_17_conv:1,6144,4;past_state_17_rec:1,16,128,128;past_state_18_conv:1,6144,4;past_state_18_rec:1,16,128,128;past_19_key:1,2,256,256;past_19_value:1,2,256,256;past_state_20_conv:1,6144,4;past_state_20_rec:1,16,128,128;past_state_21_conv:1,6144,4;past_state_21_rec:1,16,128,128;past_state_22_conv:1,6144,4;past_state_22_rec:1,16,128,128;past_23_key:1,2,256,256;past_23_value:1,2,256,256
DECODE_PATH=$ONNX_DIR/decoder_model_decode/decoder_model_decode_final.onnx
CMD_DECODE="ASCEND_SLOG_PRINT_TO_STDOUT=0 atc --model $DECODE_PATH \
    --output $OM_DIR/decoder_model_decode/decoder_model_decode \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape='$(probe_shapes "$DECODE_PATH")' \
    --precision_mode=force_fp32 --external_weight=1 \
    --device 0"

# ---- Execution ----
echo -e "════════\n$CMD_VIT" && time eval "$CMD_VIT" # ≈40s ≈2min31s
echo -e "════════\n$CMD_EMBED" && time eval "$CMD_EMBED" # ≈23S ≈1m20s
echo -e "════════\n$CMD_PREFILL" && time eval "$CMD_PREFILL" # ≈2m19s ≈21m55s
echo -e "════════\n$CMD_DECODE" && time eval "$CMD_DECODE" # ≈49s ≈4m15s

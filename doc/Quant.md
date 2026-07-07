# Quantization

## Step 1: Install dependencies

```bash
[ -f Ascend-cann-amct_8.5.0_linux-aarch64.tar.gz ] || wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-amct_8.5.0_linux-aarch64.tar.gz
mkdir -p amct_source
tar -zxvf Ascend-cann-amct*linux*.tar.gz -C amct_source
(
    cd amct_source/amct_onnx/
    pip uninstall -y amct_onnx
    # https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/amct/atlasamct_16_0030.html
    pip install numpy==1.26.4 onnxruntime==1.16.0 onnx==1.14.0 protobuf==3.20.3 opencv-python-headless amct_onnx-*-py3-none-linux_*.whl --user

    # 重建 amct_onnx_op
    # https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/amct/atlasamct_16_0034.html
    rm -rf amct_onnx_op
    tar -zxvf amct_onnx_op.tar.gz
    for f in onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_c_api.h onnxruntime_session_options_config_keys.h onnxruntime_float16.h;
    do
        unzip -j v1.16.0.zip "onnxruntime-1.16.0/include/onnxruntime/core/session/$f" -d amct_onnx_op/inc/;
    done
    ls -l amct_onnx_op/inc/
    cd amct_onnx_op/
    python3 setup.py build
    echo "======== Test ========" && python3 -c "import amct_onnx as amct; print(f'{amct.AMCT_SO=}')"
)
```

## Step 2: Quantize the model

```bash
# 参考1
python quant_amct.py quantize \
    --model-path output/onnx-output-opt/decoder_model_prefill/decoder_model_prefill_final_pad2slice.onnx \
    --qwen-path ~/Qwen3.5-2B \
    --img-path ./224x224.png \
    --save-dir ./amct_results \
    --device npu &> log_quant_amct.log &

# 参考2
python quant_amct.py quantize \
    --model-path /data/workspace/weight/Qwen3.5-2B-Edge/onnx-output-opt/decoder_model_prefill/decoder_model_prefill_final_pad2slice.onnx \
    --qwen-path /data/workspace/weight/Qwen3.5-2B \
    --img-path ./224x224.png \
    --save-dir /data/workspace/weight/Qwen3.5-2B-Edge/onnx-amct \
    --device npu &> amct.log &
```

查看日志：

```bash
tail -f log_quant_amct.log
```

```bash
python3 -c "
import onnx
from collections import Counter
model = onnx.load('output/amct_results_deploy_model.onnx')
types = [node.op_type for node in model.graph.node]
for t, c in Counter(types).most_common():
    print(f'{t}: {c}')
"
```

```log
Slice: 2366
Mul: 1898
Gather: 1642
Unsqueeze: 1509
Add: 1495
Reshape: 1465
ScatterND: 1208
ReduceSum: 1170
AscendQuant: 849
MatMul: 614
AscendDequant: 578
Transpose: 337
Cast: 336
Exp: 252
Sub: 162
Sqrt: 115
Div: 115
Where: 91
Sigmoid: 84
Pow: 79
ReduceMean: 79
Trilu: 36
Neg: 30
Concat: 25
Conv: 18
Softplus: 18
Split: 18
CumSum: 18
Expand: 12
Softmax: 6
Flatten: 1
And: 1
Cos: 1
Sin: 1
```

合并所有tensor为一个：

```bash
python3 << 'PYTHON_EOF'
import os
import onnx

onnx_path = "output/amct/decoder_model_prefill_final_pad2slice.onnx"

onnx.save(
    onnx.load("uniform_results_deploy_model.onnx"),
    onnx_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=os.path.basename(onnx_path) + "_data",
    size_threshold=1024,
    convert_attribute=False,
)

print(f"Model saved to {onnx_path}")
PYTHON_EOF
```

```bash
python3 << 'PYTHON_EOF'
import os
import onnx

onnx_path = "output/amct/decoder_model_decode_final.onnx"

onnx.save(
    onnx.load("uniform_results_deploy_model.onnx"),
    onnx_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=os.path.basename(onnx_path) + "_data",
    size_threshold=1024,
    convert_attribute=False,
)

print(f"Model saved to {onnx_path}")
PYTHON_EOF
```

```bash
(
OM_DIR="${2:-output/om_quantized/}"
SOC_VERSION="${3:-Ascend910B4-1}"

mkdir -p "$OM_DIR"
echo "════════════════════════"
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
# output 多一层级是因为保存了外置权重且外置权重路径必定在weight目录下
PREFILL_PATH=output/amct/decoder_model_prefill_final_pad2slice.onnx
CMD_PREFILL="ASCEND_SLOG_PRINT_TO_STDOUT=0 atc --model $PREFILL_PATH \
    --output $OM_DIR/decoder_model_prefill/decoder_model_prefill \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape '$(probe_shapes "$PREFILL_PATH")' \
    --precision_mode_v2 origin --external_weight=1 \
    --device 0"

# output 多一层级是因为保存了外置权重且外置权重路径必定在weight目录下，避免与decoder_prefill重复
DECODE_PATH=output/amct/decoder_model_decode_final.onnx
CMD_DECODE="ASCEND_SLOG_PRINT_TO_STDOUT=0 atc --model $DECODE_PATH \
    --output $OM_DIR/decoder_model_decode/decoder_model_decode \
    --framework 5 \
    --soc_version $SOC_VERSION \
    --input_shape '$(probe_shapes "$DECODE_PATH")' \
    --precision_mode_v2 origin --external_weight=1 \
    --device 0"

# ---- Execution ----
echo -e "════════\n$CMD_PREFILL" && time eval "$CMD_PREFILL" # ≈3m35s
echo -e "════════\n$CMD_DECODE" && time eval "$CMD_DECODE" # ≈1m12s
)
```

```bash
python val_qwen3p5_om.py   --vit-path output/om/vision_encoder.om   --embedding-path output/om/embedding.om   --decoder-prefill-path output/om_quantized/decoder_model_prefill/decoder_model_prefill.om   --decoder-decode-path output/om_quantized/decoder_model_decode/decoder_model_decode.om   --qwen-path ~/Qwen3.5-2B
```

## Step 3: Analyze the quantized output

量化后会得到两个模型：

1. `*_deploy_model.onnx`: 量化部署模型，即量化后的可在昇腾 AI 处理器部署的模型文件。
2. `*_fake_quant_model.onnx`: 量化仿真模型，即量化后的可在 ONNX 执行框架 ONNX Runtime 进行精度仿真的模型文件。

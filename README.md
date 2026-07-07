# tonnx2chip

Qwen3.5 ONNX → ATC → OM 端到端流水线，用于在华为昇腾 NPU 上部署 Qwen3.5-2B 多模态模型。

## 安装

### 1. 安装 Python 包

```bash
pip install git+<repo-url>
```

### 2. 安装昇腾环境依赖

```bash
curl -sSL https://raw.githubusercontent.com/Yanguan619/2onnx2chip/main/repair_python_env.sh | bash
```

该脚本自动完成：

1. 安装 `requirements.txt` 中的核心依赖
2. 从源码编译安装 `auto_optimizer`（msit 仓库）
3. 从源码编译安装 OM 运行时（ais_bench 等）

### 3. 安装量化工具

参考 `doc/Quant.md` 安装 AMCT（amct_onnx）。

## 目录结构

```
src/tonnx2chip/
├── __init__.py
├── cli.py                                 # CLI 入口（统一子命令）
├── constants.py                           # 模型常量
├── export/
│   ├── export_onnx.py                     # ONNX 导出（ViT / Embed / Prefill / Decode）
│   └── convert_onnx2om.sh                # ATC 脚本：ONNX → OM 转换
├── optimize/
│   ├── pad_to_slice.py                    # Pad → Slice 重写（避免 EZ9999）
│   └── optimize_onnx.py                   # ONNX 优化（onnxslim + auto_optimizer）
├── validate/
│   ├── val_onnx.py                        # ONNX 验证 + PyTorch 基线对比
│   └── val_om.py                          # OM 验证 + 内存监控
├── infer/
│   ├── infer_om.py                        # 全 OM 推理
│   ├── infer_om_fast.py                   # OM 快速推理（zero-copy）
│   ├── infer_om_prefill_onnx_decode.py    # Prefill(OM) + Decode(ONNX)
│   ├── infer_onnx_prefill_om_decode.py    # Prefill(ONNX) + Decode(OM)
│   └── infer_onnx_vit_om_rest.py          # ViT(ONNX) + 其余(OM)
├── quantize/
│   ├── quant_uniform.py                   # AMCT 均匀量化
│   └── quant_nuq.py                       # 非均匀量化（NUQ）
├── tools/
│   ├── memory_monitor.py                  # Host & NPU 内存监控
│   └── monitor_mem.sh                     # Shell 内存监控
└── config/
    ├── keep_dtype.list                    # 保留数据类型节点列表
    └── nuq_base.cfg                       # NUQ 量化配置模板
```

## CLI 命令

```bash
tonnx2chip --help          # 查看所有子命令
```

| 子命令 | 说明 |
|---|---|
| `tonnx2chip export` | ONNX 导出 |
| `tonnx2chip optimize-pad` | Pad → Slice 重写 |
| `tonnx2chip optimize-onnx` | ONNX 优化（onnxslim + auto_optimizer） |
| `tonnx2chip val-onnx` | ONNX 推理验证 |
| `tonnx2chip val-om` | OM 推理验证 |
| `tonnx2chip infer-om` | OM 推理 |
| `tonnx2chip infer-fast` | OM 快速推理 |
| `tonnx2chip quant` | AMCT 均匀量化 |
| `tonnx2chip quant-nuq` | 非均匀量化 |

## 操作流程

### Step 1: 环境准备

下载模型权重：

```bash
modelscope download Qwen/Qwen3.5-2B --local_dir ~/Qwen3.5-2B
```

### Step 2: ONNX 导出

```bash
tonnx2chip export \
  --qwen-path ~/Qwen3.5-2B \
  --export-path output/onnx-output \
  --img-path ./assets/224x224.png
```

输出：

```
onnx-output/
├── vision_encoder.onnx (+ _data)
├── embedding.onnx (+ _data)
├── decoder_model_prefill.onnx (+ _data)
└── decoder_model_decode.onnx (+ _data)
```

### Step 3: ONNX 优化

使用 `ascend-onnx-atc-pipeline` skill 的 `scripts/optimize_onnx.py` 对每个子模型执行优化，然后对 prefill 模型执行 Pad→Slice 重写：

```bash
tonnx2chip optimize \
  --input-path output/onnx-output-opt/decoder_model_prefill/decoder_model_prefill_final.onnx \
  --output-path output/onnx-output-opt/decoder_model_prefill/decoder_model_prefill_final.replacedpad.onnx
```

### Step 4: ONNX 验证

```bash
tonnx2chip val-onnx \
  --vit-path output/onnx-output-opt/vision_encoder/vision_encoder_final.onnx \
  --embed-path output/onnx-output-opt/embedding/embedding_final.onnx \
  --decoder-prefill-path output/onnx-output-opt/decoder_model_prefill/decoder_model_prefill_final.replacedpad.onnx \
  --decoder-decode-path output/onnx-output-opt/decoder_model_decode/decoder_model_decode_final.onnx \
  --qwen-path ~/Qwen3.5-2B
```

### Step 5: ATC 转 OM

```bash
bash src/tonnx2chip/export/convert_onnx2om.sh \
  output/onnx-output-opt \
  output/om \
  {SOC_VERSION}
```

输出：

```
om-output/
├── vision_encoder.om
├── embed.om
├── decoder_prefill/
│   ├── decoder_prefill.om
│   └── weight/
└── decoder_decode/
    ├── decoder_decode.om
    └── weight/
```

### Step 6: OM 验证

```bash
tonnx2chip val-om \
  --vit-path output/om/vision_encoder.om \
  --embedding-path output/om/embedding.om \
  --decoder-prefill-path output/om/decoder_model_prefill/decoder_model_prefill.om \
  --decoder-decode-path output/om/decoder_model_decode/decoder_model_decode.om \
  --qwen-path ~/Qwen3.5-2B
```

## 模型架构要点

- **24 层 Transformer**: 6 层 Full Attention (3,7,11,15,19,23) + 18 层 Linear Attention (Gated Delta Net)
- **3D M-RoPE**: 位置编码 3 维 (temporal, height, width)，图片 token 编码空间位置
- **Vision Encoder**: 输出 `seq_len x 1536` 特征，通过 `masked_scatter` 替换 input_ids 中的 image token placeholder
- **External Weight**: Decoder 权重约 3.5GB，ATC 使用 `--external_weight=1` 让 Prefill 和 Decode 共享权重

## 常见问题

- **EZ9999 te_padv3**: 必须将 prefill ONNX 中的 Pad 改写为 Slice，否则 CANN 某些版本触发 `EZ9999: DDR address out of range`
- **动态 Shape**: ATC 侧映射为 `-1`，OM 侧通过 `reshape_outputs()` 处理
- **权重共享**: Prefill 和 Decode 共享同一份 Decoder 权重（约 3.5GB），避免 HBM 重复加载

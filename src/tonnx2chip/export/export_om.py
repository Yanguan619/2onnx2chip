#!/usr/bin/env python3
"""
Convert ONNX models from qwen35_onnx.py export to OM format for Ascend NPU.

Usage:
    python convert_onnx2om.py                              # use defaults
    python convert_onnx2om.py <onnx_dir> <om_dir> <soc_version>
    python convert_onnx2om.py /data/workspace/weight/Qwen3.5-2B-Edge/onnx-output-opt/ /data/workspace/weight/Qwen3.5-2B-Edge/om-310p Ascend310P7

Defaults:
    ONNX_DIR:    output/onnx-output-opt/
    OM_DIR:      output/om/
    SOC_VERSION: Ascend910_9362
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import onnx


def probe_shapes(model_path: str) -> str:
    """Probe ONNX input shapes from model file.

    For dynamic dims (dim_value=0 or dim_param non-empty), output "-1" so ATC
    treats them as dynamic.
    """
    model = onnx.load(model_path)
    pairs = []
    for inp in model.graph.input:
        dims = []
        shape = inp.type.tensor_type.shape
        for d in shape.dim:
            if d.dim_value > 0:
                dims.append(str(d.dim_value))
            else:
                dims.append("-1")
        pairs.append(f"{inp.name}:{','.join(dims)}")
    return ";".join(pairs)


def run_atc(cmd: str, description: str):
    """Execute an ATC command and print timing."""
    print(f"\n{'═' * 60}")
    print(f"[{description}]")
    print(f"Command: {cmd}")
    print(f"{'═' * 60}")

    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"❌ [{description}] FAILED with return code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"✅ [{description}] completed in {elapsed:.1f}s")


def atc(
    onnx_path: str,
    om_path: str | Path,
    soc_version: str,
    extra_args: str = "",
):
    shapes = probe_shapes(str(onnx_path))
    if "--external_weight=1" in extra_args:
        om_dir = f"{om_path}/" + Path(onnx_path).stem
        os.makedirs(om_dir, exist_ok=True)
        om_path = Path(om_dir) / f"{Path(onnx_path).stem}.om"

    cmd = (
        f"atc --model {onnx_path} "
        f"--output {str(om_path)} "
        f"--framework 5 "
        f"--soc_version {soc_version} "
        f"--input_shape='{shapes}' "
        f"--device 0 "
        f"{extra_args}"
    )

    return cmd


def main(
    onnx_dir: str | Path,
    om_dir: str | Path,
    soc_version: str,
) -> None:
    """
    onnx_dir 下必须有以下文件：
    - vision_encoder/vision_encoder_final.onnx
    - embedding/embedding_final.onnx
    - decoder_model_prefill/decoder_model_prefill_final_pad2slice.onnx
    - decoder_model_decode/decoder_model_decode_final.onnx
    """
    os.makedirs(Path(om_dir).parent, exist_ok=True)

    print(f"{'═' * 50}")
    print(f"ONNX_DIR:    {onnx_dir}")
    print(f"OM_DIR:      {om_dir}")
    print(f"SOC_VERSION: {soc_version}")
    print(f"{'═' * 50}")
    onnx_dir = Path(onnx_dir)

    vit_path = onnx_dir / "vision_encoder/vision_encoder_final.onnx"
    embed_path = onnx_dir / "embedding/embedding_final.onnx"
    prefill_path = onnx_dir / "decoder_model_prefill/decoder_model_prefill_final_pad2slice.onnx"
    decode_path = onnx_dir / "decoder_model_decode/decoder_model_decode_final.onnx"
    assert Path(vit_path).exists(), f"Vision Encoder ONNX not found: {vit_path}"
    assert Path(embed_path).exists(), f"Embedding ONNX not found: {embed_path}"
    assert Path(prefill_path).exists(), f"Decoder Prefill ONNX not found: {prefill_path}"
    assert Path(decode_path).exists(), f"Decoder Decode ONNX not found: {decode_path}"
    # --- Vision Encoder ---
    vit_out_path = f"{om_dir}/vision_encoder"
    keep_dtype_path = Path(__file__).parent.parent / "config" / "keep_dtype.list"
    run_atc(
        atc(str(vit_path), vit_out_path, soc_version, f"--keep_dtype={keep_dtype_path}"),
        "Vision Encoder",
    )

    # --- Embedding ---
    embed_out_path = f"{om_dir}/embedding"
    run_atc(atc(str(embed_path), embed_out_path, soc_version), "Embedding")

    # --- Prefill Decoder ---
    prefill_out_path = f"{om_dir}/decoder_model_prefill"
    run_atc(
        atc(
            str(prefill_path),
            prefill_out_path,
            soc_version,
            "--precision_mode=force_fp32 --external_weight=1",
        ),
        "Decoder Prefill",
    )

    # --- Decode Decoder ---
    decode_out_path = f"{om_dir}/decoder_model_decode"
    run_atc(
        atc(
            str(decode_path),
            decode_out_path,
            soc_version,
            "--precision_mode=force_fp32 --external_weight=1",
        ),
        "Decoder Decode",
    )

    print(f"\n{'═' * 30}")
    print("✅ All conversions completed successfully!")
    print(f"{'═' * 30}")


if __name__ == "__main__":
    pass

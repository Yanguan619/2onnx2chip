"""
昇腾 AMCT 非均匀量化 (NUQ) 脚本
参考: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/amct/atlasamct_16_0145.html

命令:
  uniform    均匀量化 → deploy_model.onnx + fake_quant_model.onnx
  nuq        ATC 转融合 JSON + 非均匀量化 → deploy + fake_quant

约束:
  - 仅支持 Conv (dilation=1, group=1, filter=4D) 和 Gemm (transA=false, alpha=beta=1.0)
  - 硬件限制: 该版本不建议使用非均匀量化功能, 获取不到性能收益

Usage:
  # 1. 均匀量化
  python quant_nuq.py uniform \
      --model-path <onnx> --qwen-path <qwen> --save-dir ./uniform

  # 2. 非均匀量化 (自动执行 ATC + NUQ)
  python quant_nuq.py nuq \
      --model-path <onnx> \
      --deploy-model ./uniform/<model>_deploy_model.onnx \
      --qwen-path <qwen> --save-dir ./nuq
"""

import os
import string
import subprocess
from pathlib import Path

import amct_onnx as amct

from tonnx2chip.quantize.quant_uniform import (
    EXPECTED_KL_DIVERGENCE,
    QwenEvaluator,
    clean_temp_dirs,
    guess_model_name,
    quantize_one,
)

_HERE = Path(__file__).resolve().parent
_NUQ_CFG_TEMPLATE = _HERE.parent / "config" / "nuq_base.cfg"


# ── ATC 生成融合 JSON ─────────────────────────────────────────────


def atc_to_json(deploy_model: str, output_json: str):
    """调用 ATC --mode=1 将 deploy 模型转为融合 JSON 文件."""
    import shlex

    cmd = f"atc --mode 1 --om {deploy_model} --json {output_json} --framework 5 --soc_version TsnsC --enable_compress_weight"
    print(f"[INFO][ATC] {cmd}")
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ATC failed:\n{result.stderr}")
    print(f"[INFO][ATC] → {output_json}")
    return output_json


# ── 生成非均匀量化 cfg ────────────────────────────────────────────


def gen_nuq_cfg(
    mapping_json: str,
    output_cfg: str,
    num_steps: int = 32,
    num_of_iteration: int = 0,
    activation_offset: bool = True,
    batch_num: int = 2,
):
    """从 nuq_base.cfg 模板生成非均匀量化配置文件."""
    with open(_NUQ_CFG_TEMPLATE) as f:
        template = string.Template(f.read())
    content = template.safe_substitute(
        mapping_json=mapping_json,
        num_steps=num_steps,
        num_of_iteration=num_of_iteration,
        activation_offset="true" if activation_offset else "false",
        batch_num=batch_num,
    )
    with open(output_cfg, "w") as f:
        f.write(content)
    print(f"[CFG] NUQ config written to {output_cfg}")


IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"


# ── CLI: 非均匀量化 ────────────────────────────────────────────────
def nuq(
    model_path: str = typer.Option(..., help="原始 ONNX 模型"),
    deploy_model: str = typer.Option(
        ..., help="均匀量化后的 deploy_model.onnx (用于 ATC 转融合 JSON)"
    ),
    qwen_path: str = typer.Option(..., help="Qwen3.5 模型目录"),
    save_dir: str = typer.Option("./nuq_results", help="输出目录"),
    img_path: str = typer.Option(IMAGE_PATH, help="校准图片路径"),
    device: str = typer.Option("npu", help="Torch 设备"),
    num_steps: int = typer.Option(32, help="NUQ 台阶数 (越小压缩率越高)"),
    num_iter: int = typer.Option(0, help="NUQ 迭代次数 (0=默认)"),
    batch_num: int = typer.Option(1, help="校准 batch 数"),
    expected_acc_loss: float = typer.Option(EXPECTED_KL_DIVERGENCE, help="KL 散度阈值"),
    decode_steps: int = typer.Option(256, help="Decode 校准步数"),
):
    """非均匀量化: ATC 转融合 JSON → NUQ cfg → accuracy_based_auto_calibration."""
    os.makedirs(save_dir, exist_ok=True)
    clean_temp_dirs(Path(os.getcwd()))
    model_name = guess_model_name(model_path)

    # 1. ATC 将均匀量化的 deploy model 转为融合 JSON
    mapping_json = os.path.join(save_dir, "uniform_quanatized.json")
    atc_to_json(deploy_model, mapping_json)

    # 2. 生成 NUQ cfg 并创建量化配置
    cfg_file = os.path.join(save_dir, "nuq_config.cfg")
    gen_nuq_cfg(
        mapping_json=mapping_json,
        output_cfg=cfg_file,
        num_steps=num_steps,
        num_of_iteration=num_iter,
        batch_num=batch_num,
    )

    config_file = os.path.join(save_dir, "nuq_quant_config.json")
    amct.create_quant_config(
        config_file=config_file,
        model_file=model_path,
        config_defination=cfg_file,
    )

    # 3. 使用 QwenEvaluator + accuracy_based_auto_calibration
    evaluator = QwenEvaluator(
        model_name=model_name,
        qwen_path=qwen_path,
        img_path=img_path,
        device=device,
        expected_acc_loss=expected_acc_loss,
        decode_steps=decode_steps,
    )

    amct.accuracy_based_auto_calibration(
        model_file=model_path,
        model_evaluator=evaluator,
        config_file=config_file,
        record_file=os.path.join(save_dir, "scale_offset_record.txt"),
        save_dir=save_dir,
        strategy="BinarySearch",
        sensitivity="CosineSimilarity",
    )

    print(f"\n[Done] NUQ quantization results in {save_dir}:")
    for onnx_file in sorted(Path(save_dir).glob("*.onnx")):
        print(f"  {onnx_file}")


if __name__ == "__main__":
    pass

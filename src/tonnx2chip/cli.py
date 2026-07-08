"""tonnx2chip CLI – unified entry point with subcommands."""

from pathlib import Path

import typer

app = typer.Typer(
    name="tonnx2chip",
    help="Qwen3.5 ONNX → ATC → OM pipeline for Ascend NPU.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)

IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"
PROMPT = "Describe this image."


@app.command()
def export_onnx(
    qwen_path: str = typer.Option(..., help="Qwen3.5-2B model path"),
    export_path: str = typer.Option(..., help="Export path"),
    img_path: str = typer.Option(..., help="Input path"),
    prompt=typer.Option(PROMPT, help="Prompt"),
    context_length: int = typer.Option(256, help="Context length for prefill and decode"),
):
    "Step1: ONNX export (Vision Encoder / Embedding / Prefill / Decode)."
    from tonnx2chip.export.export_onnx import main as export

    export(
        qwen_path=qwen_path,
        export_path=export_path,
        img_path=img_path,
        text=prompt,
        context_length=context_length,
    )


@app.command()
def optimize_onnx(
    model: str = typer.Option(..., help="Path to the source ONNX model"),
    save_dir: str = typer.Option("", help="Directory for optimized ONNX artifacts"),
    probe_shape_profile: list[str] = typer.Option(
        [],
        help="Concrete input profile used to probe graph output shapes before optimization",
    ),
    skip_output_shape_probe: bool = typer.Option(
        False, help="Skip the pre-optimization ORT probe and graph output shape patch"
    ),
    autoopt_knowledges: str = typer.Option(
        "",
        help="Comma-separated knowledge names for auto_optimizer (default: all active knowledges)",
    ),
) -> None:
    "Step2: ONNX optimization pipeline (onnxslim + auto_optimizer)."
    from tonnx2chip.optimize.optimize_onnx import main as optimize_onnx

    optimize_onnx(
        model=model,
        save_dir=save_dir,
        probe_shape_profile=probe_shape_profile,
        skip_output_shape_probe=skip_output_shape_probe,
        autoopt_knowledges=autoopt_knowledges,
    )


@app.command()
def modify_onnx_pad(
    input_path: str = typer.Option(..., help="Input ONNX model"),
    output_path: str | None = typer.Option(None, help="Output ONNX path"),
):
    "Step2.1(Optional): Replace Pad nodes with Slice nodes for better NPU compatibility."
    from tonnx2chip.optimize.pad_to_slice import main as modify_onnx_pad

    modify_onnx_pad(input_path=input_path, output_path=output_path)


@app.command()
def val_onnx(
    vit_path: str = typer.Option(..., help="Dir with ONNX files"),
    embed_path: str = typer.Option(..., help="Dir with ONNX files"),
    decoder_prefill_path: str = typer.Option(..., help="Dir with ONNX files"),
    decoder_decode_path: str = typer.Option(..., help="Dir with ONNX files"),
    qwen_path: str = typer.Option(..., help="Original Qwen model dir"),
    prompt: str = typer.Option(PROMPT, help="Prompt"),
    image_path: str = typer.Option(IMAGE_PATH, help="Image path"),
    max_new_tokens: int = typer.Option(20, help="Max new tokens"),
):
    "Validate OM inference + PyTorch baseline comparison."
    from tonnx2chip.validate.val_onnx import main as val_onnx_main

    val_onnx_main(
        vit_path=vit_path,
        embed_path=embed_path,
        decoder_prefill_path=decoder_prefill_path,
        decoder_decode_path=decoder_decode_path,
        qwen_path=qwen_path,
        prompt=prompt,
        image_path=image_path,
        max_new_tokens=max_new_tokens,
    )


@app.command()
def quantize_onnx(
    model_path: str = typer.Option(..., help="Path to the ONNX model"),
    qwen_path: str = typer.Option(..., help="Original Qwen3.5 model directory"),
    save_dir: str = typer.Option("./output/amct_results", help="Output directory"),
    img_path: str = typer.Option(None, help="Calibration image path"),
    device: str = typer.Option("npu", help="Torch device"),
    expected_acc_loss: float = typer.Option(0.01, help="KL threshold"),
    activation_offset: bool = typer.Option(True, help="Enable activation offset"),
    decode_steps: int = typer.Option(256, help="Decode calibration steps"),
):
    """Step2.2(Optional): Quantize a single decoder_prefill or decoder_decode ONNX model."""

    from tonnx2chip.quantize.quant_uniform import main as quantize_one

    IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"

    def _default_img_path() -> str:
        if not IMAGE_PATH.exists():
            raise FileNotFoundError(f"默认校准图片不存在: {IMAGE_PATH}")
        return str(IMAGE_PATH)

    quantize_one(
        model_path=model_path,
        qwen_path=qwen_path,
        save_dir=save_dir,
        img_path=img_path or _default_img_path(),
        device=device,
        expected_acc_loss=expected_acc_loss,
        activation_offset=activation_offset,
        decode_steps=decode_steps,
    )


@app.command()
def export_om_all(
    onnx_dir: str = typer.Option(..., help="Directory containing ONNX models"),
    soc_version: str = typer.Option(..., help="Ascend SoC version"),
    om_dir: str = typer.Option("output/om", help="Output directory for OM models"),
):
    "Step3: Convert ONNX to OM (ATC)."
    from tonnx2chip.export.export_om import main as convert_om

    convert_om(onnx_dir=onnx_dir, om_dir=om_dir, soc_version=soc_version)


@app.command()
def val_om(
    vit_path: str = typer.Option(
        ...,
        help="VIT OM path (use 'none' to disable for text-only)",
    ),
    embedding_path: str = typer.Option(
        ...,
        help="Embedding OM path",
    ),
    decoder_prefill_path: str = typer.Option(
        ...,
        help="Decoder prefill OM path (pad2slice version recommended)",
    ),
    decoder_decode_path: str = typer.Option(
        ...,
        help="Decoder decode OM path",
    ),
    qwen_path: str = typer.Option(..., help="Original Qwen model dir"),
    prompt: str = typer.Option(PROMPT, help="Prompt"),
    image_path: str = typer.Option(IMAGE_PATH, help="Image path (or 'none')"),
    max_new_tokens: int = typer.Option(2, help="Max new tokens"),
    device_id: int = typer.Option(0, help="NPU device ID"),
    baseline: bool = typer.Option(True, help="Also run PyTorch baseline for comparison"),
):
    "Validate OM inference + PyTorch baseline comparison."
    from tonnx2chip.validate.val_om import main as val_om_main

    val_om_main(
        vit_path=vit_path,
        embedding_path=embedding_path,
        decoder_prefill_path=decoder_prefill_path,
        decoder_decode_path=decoder_decode_path,
        qwen_path=qwen_path,
        prompt=prompt,
        image_path=image_path,
        max_new_tokens=max_new_tokens,
        device_id=device_id,
        baseline=baseline,
    )


# app.add_typer(
#     quant_nuq,
#     name="quant-nuq",
#     help="Quantization (nuq).",
# )

if __name__ == "__main__":
    app()

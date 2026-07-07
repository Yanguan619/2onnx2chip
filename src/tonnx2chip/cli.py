"""tonnx2chip CLI – unified entry point with subcommands."""

import typer

from tonnx2chip.export.export_onnx import app as export
from tonnx2chip.optimize.optimize_onnx import app as optimize_onnx
from tonnx2chip.optimize.pad_to_slice import app as optimize_pad
from tonnx2chip.quantize.quant_nuq import app as quant_nuq
from tonnx2chip.quantize.quant_uniform import app as quant
from tonnx2chip.validate.val_om import app as val_om
from tonnx2chip.validate.val_onnx import app as val_onnx

app = typer.Typer(
    name="tonnx2chip",
    help="Qwen3.5 ONNX → ATC → OM pipeline for Ascend NPU.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)

app.add_typer(
    export, name="export", help="ONNX export (Vision Encoder / Embedding / Prefill / Decode)."
)
app.add_typer(
    optimize_pad,
    name="optimize-pad",
    help="Pad → Slice rewrite (avoid EZ9999 te_padv3).",
)
app.add_typer(
    optimize_onnx,
    name="optimize-onnx",
    help="ONNX optimization pipeline (onnxslim + auto_optimizer).",
)
app.add_typer(
    val_onnx,
    name="val-onnx",
    help="Validate ONNX inference + PyTorch baseline comparison.",
)
app.add_typer(
    val_om,
    name="val-om",
    help="Validate OM inference on Ascend NPU.",
)
app.add_typer(
    quant,
    name="quant",
    help="Quantization (uniform + nuq).",
)
app.add_typer(
    quant_nuq,
    name="quant-nuq",
    help="Quantization (nuq).",
)

if __name__ == "__main__":
    app()

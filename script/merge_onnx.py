import os
import onnx

onnx_path = "output/onnx-amct/decoder_model_prefill.onnx"
tmp_path = "output/amct_results_deploy_model.onnx"

onnx_model = onnx.load(tmp_path)
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
print(f"Saving onnx (external data) to {onnx_path}")
onnx.save(
    onnx_model,
    onnx_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=os.path.basename(onnx_path) + "_data",
    size_threshold=1024,
    convert_attribute=False,
)
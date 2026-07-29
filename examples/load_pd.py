import time
from pathlib import Path
from ais_bench.infer.interface import InferSession

decoder_prefill_path = "output/om/decoder_model_prefill/decoder_model_prefill.om"
decoder_decode_path = "output/om/decoder_model_decode/decoder_model_decode.om"
device_id = 0  # GPU device id

prefill_weight_dir = str(Path(decoder_prefill_path).parent / "weight")
decode_weight_dir = str(Path(decoder_decode_path).parent / "weight")
prefill = InferSession(device_id, decoder_prefill_path, weight_dir=prefill_weight_dir, debug=True)

time.sleep(2)
decode = InferSession(device_id, decoder_decode_path, weight_dir=decode_weight_dir, debug=True)

time.sleep(2)
"""
Qwen3.5 AMCT 精度感知自动量化脚本.
仅支持固定 seq_len=256 的 ONNX 子模型 (decoder_prefill / decoder_decode).

使用 KL 散度 (负值, higher=better) 作为评估指标,
配合 AMCT 的 accuracy_based_auto_calibration 自动回退敏感层.

Usage:
    python quant_amct.py quantize-all \
        --onnx-dir <onnx_output_dir> \
        --qwen-path <qwen_model_dir> \
        [--img-path <calibration_image>] \
        [--save-dir ./amct_results] \
        [--device cpu]

    python quant_amct.py quantize \
        --model-path <single_onnx_file> \
        --qwen-path <qwen_model_dir> \
        [--img-path <calibration_image>] \
        [--save-dir ./amct_results] \
        [--device cpu]
"""

import os
import shutil
from pathlib import Path

import amct_onnx as amct
import numpy as np
import onnxruntime as ort
import torch
import typer
from amct_onnx.common.auto_calibration import AutoCalibrationEvaluatorBase
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

app = typer.Typer(pretty_exceptions_enable=False)

TARGET_SEQ_LEN = 256
EXPECTED_KL_DIVERGENCE = 0.01  # KL threshold; tune empirically
IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"


def _default_img_path() -> str:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"默认校准图片不存在: {IMAGE_PATH}")
    return str(IMAGE_PATH)


def guess_model_name(model_path: str) -> str:
    stem = Path(model_path).stem.lower()
    if "prefill" in stem:
        return "decoder_prefill"
    if "decode" in stem:
        return "decoder_decode"
    raise ValueError(f"Cannot infer model type from path: {model_path}")


def clean_temp_dirs(root: Path):
    for pattern in [".external", "temp*"]:
        for d in root.glob(pattern):
            shutil.rmtree(d, ignore_errors=True)


def create_session(model_file: str) -> ort.InferenceSession:
    session_options = amct.AMCT_SO
    providers = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU":
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(model_file, session_options, providers=providers)


class CalibInputBuilder:
    """Build fixed-length calibration inputs for decoder_prefill / decoder_decode."""

    def __init__(self, qwen_path: str, device: str = "cpu"):
        self.qwen_path = qwen_path
        self.device = device
        self.processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
        self.config = AutoConfig.from_pretrained(qwen_path, local_files_only=True)
        if not hasattr(self.config, "text_config"):
            raise ValueError("Model config must have text_config")

        self.text_config = self.config.text_config
        self.hidden_size = self.text_config.hidden_size
        self.num_layers = self.text_config.num_hidden_layers
        self._full_model = None

    @property
    def full_model(self) -> Qwen3_5ForConditionalGeneration:
        if self._full_model is None:
            self._full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.qwen_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        return self._full_model

    @property
    def model_device(self):
        return next(self.full_model.parameters()).device

    @staticmethod
    def _to_numpy(tensor: torch.Tensor, dtype: np.floating = np.float16) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(dtype)

    def build_prefill_inputs(self, img_path: str = "") -> tuple[dict, int, int]:
        print("  [DataBuilder] 构建 prefill 校准输入...")
        def _build_token_inputs(img_path: str = "", prompt: str = "Describe this image.") -> dict:
            content = []
            if img_path and Path(img_path).exists():
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = _build_token_inputs(img_path)
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        d = self.model_device
        model = self.full_model.model

        with torch.no_grad():
            inputs_embeds = self.full_model.get_input_embeddings()(input_ids.to(d))
            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = model.get_image_features(
                    pixel_values.to(d),
                    image_grid_thw.to(d),
                    return_dict=True,
                ).pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = model.get_placeholder_mask(
                    input_ids.to(d),
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            position_ids = model.compute_3d_position_ids(
                input_ids=input_ids.to(d),
                image_grid_thw=image_grid_thw.to(d) if image_grid_thw is not None else None,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"].to(d),
                past_key_values=None,
                mm_token_type_ids=mm_token_type_ids.to(d)
                if mm_token_type_ids is not None
                else None,
            )

            if position_ids is None:
                attn_mask = inputs["attention_mask"].to(d)
                position_ids = attn_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attn_mask == 0, 0)
                position_ids = position_ids.view(1, 1, -1).expand(3, -1, -1)

            orig_seq_len = min(inputs_embeds.shape[1], TARGET_SEQ_LEN)
            inputs_embeds = inputs_embeds[:, :orig_seq_len, :]
            attention_mask = inputs["attention_mask"][:, :orig_seq_len].to(d)
            position_ids = position_ids[:, :, :orig_seq_len]

            if orig_seq_len < TARGET_SEQ_LEN:
                pad_len = TARGET_SEQ_LEN - orig_seq_len
                pad_emb = torch.zeros(
                    (1, pad_len, self.hidden_size), dtype=inputs_embeds.dtype, device=d
                )
                inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                pad_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype, device=d)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
                pad_pos = torch.full(
                    (3, 1, pad_len),
                    orig_seq_len - 1,
                    dtype=position_ids.dtype,
                    device=d,
                )
                position_ids = torch.cat([position_ids, pad_pos], dim=2)

            out = self.full_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            golden_token = int(out.logits[0, orig_seq_len - 1].argmax())

        print(f"  [DataBuilder] prefill 输入构建完成 (seq_len={orig_seq_len}, golden_token={golden_token})")
        return (
            {
                "multimodal_attention_mask": self._to_numpy(attention_mask, np.int64),
                "position_ids": self._to_numpy(position_ids, np.int64),
                "multimodal_embeddings": self._to_numpy(inputs_embeds, np.float16),
            },
            golden_token,
            orig_seq_len,
        )

    def build_decode_inputs(self, img_path: str = "") -> tuple[dict, int]:
        print("  [DataBuilder] 构建 decode 校准输入...")
        print("  [DataBuilder] 步骤 1/3: 构建 prefill 输入...")
        prefill_inputs, _, _ = self.build_prefill_inputs(img_path)
        # prefill_len = number of non-padded tokens (mask中1的个数)
        prefill_len = int(prefill_inputs["multimodal_attention_mask"].sum())
        d = self.model_device

        embeddings_t = torch.from_numpy(prefill_inputs["multimodal_embeddings"]).to(d).half()
        mask_t = torch.from_numpy(prefill_inputs["multimodal_attention_mask"]).to(d).long()
        pos_t = torch.from_numpy(prefill_inputs["position_ids"]).to(d).long()

        print("  [DataBuilder] 步骤 2/3: PyTorch 推理获取 KV cache...")
        with torch.no_grad():
            out = self.full_model(
                attention_mask=mask_t,
                position_ids=pos_t,
                inputs_embeds=embeddings_t,
                use_cache=True,
            )
            pkv = out.past_key_values
            next_token_id = int(out.logits[0, prefill_len - 1].argmax())
            next_embeds_t = self.full_model.get_input_embeddings()(
                torch.tensor([[next_token_id]], device=d),
            )
            next_embeds = self._to_numpy(next_embeds_t, np.float16)

        decode_inputs = {
            "inputs_embeds": next_embeds,
            "multimodal_attention_mask": np.concatenate(
                [
                    prefill_inputs["multimodal_attention_mask"],
                    np.ones((1, 1), dtype=np.int64),
                ],
                axis=1,
            ),
            # position_ids shape [1,1]: decode阶段每步只处理1个token，与ONNX模型输入契约一致
            "position_ids": np.array([[prefill_len]], dtype=np.int64),
        }

        layer_types = self.text_config.layer_types
        num_kv_heads = getattr(
            self.text_config,
            "num_key_value_heads",
            self.text_config.num_attention_heads,
        )
        head_dim = self.text_config.hidden_size // self.text_config.num_attention_heads
        conv_dim = 3 * self.text_config.hidden_size
        conv_kernel = getattr(self.text_config, "linear_conv_kernel_dim", 4)
        num_kv_heads_linear = getattr(self.text_config, "linear_num_key_heads", 16)
        linear_key_head_dim = getattr(self.text_config, "linear_key_head_dim", 128)
        linear_value_head_dim = getattr(self.text_config, "linear_value_head_dim", 128)

        is_new_cache = hasattr(pkv, "layers")
        print(f"  [DataBuilder] 步骤 3/3: 提取 {self.num_layers} 层 KV cache...")
        for layer in range(self.num_layers):
            print(f"  [DataBuilder]   layer {layer}/{self.num_layers} ({layer_types[layer]})")
            if layer_types[layer] == "full_attention":
                if is_new_cache:
                    layer_cache = pkv.layers[layer]
                    k_t = layer_cache.keys
                    v_t = layer_cache.values
                else:
                    k_t = pkv.key_cache[layer]
                    v_t = pkv.value_cache[layer]
                if k_t is None or v_t is None:
                    print(
                        f"  [Warning] KV cache at layer {layer} is None on device={self.device}, "
                        f"initializing with zeros (seq_len={prefill_len}). "
                        f"Calibration data may be suboptimal; use --device cpu for correctness."
                    )
                    k_t = torch.zeros(
                        1,
                        num_kv_heads,
                        prefill_len,
                        head_dim,
                        dtype=torch.float16,
                        device=d,
                    )
                    v_t = torch.zeros(
                        1,
                        num_kv_heads,
                        prefill_len,
                        head_dim,
                        dtype=torch.float16,
                        device=d,
                    )
                k = self._to_numpy(k_t, np.float16)
                v = self._to_numpy(v_t, np.float16)
                if k.shape[2] > TARGET_SEQ_LEN:
                    k = k[:, :, -TARGET_SEQ_LEN:, :]
                    v = v[:, :, -TARGET_SEQ_LEN:, :]
                decode_inputs[f"past_{layer}_key"] = k
                decode_inputs[f"past_{layer}_value"] = v
            else:
                if (
                    not is_new_cache
                    and hasattr(pkv, "conv_states")
                    and pkv.conv_states[layer] is not None
                    and hasattr(pkv, "recurrent_states")
                    and pkv.recurrent_states[layer] is not None
                ):
                    conv_t = pkv.conv_states[layer]
                    rec_t = pkv.recurrent_states[layer]
                else:
                    print(
                        f"  [Warning] Mamba state at layer {layer} is None on device={self.device}, "
                        f"initializing with zeros. Calibration data may be suboptimal; "
                        f"use --device cpu for correctness."
                    )
                    conv_t = torch.zeros(1, conv_dim, conv_kernel, dtype=torch.float16, device=d)
                    rec_t = torch.zeros(
                        1,
                        num_kv_heads_linear,
                        linear_key_head_dim,
                        linear_value_head_dim,
                        dtype=torch.float32,
                        device=d,
                    )
                decode_inputs[f"past_state_{layer}_conv"] = self._to_numpy(conv_t, np.float16)
                decode_inputs[f"past_state_{layer}_rec"] = self._to_numpy(rec_t, np.float32)

        decode_mask_t = torch.from_numpy(decode_inputs["multimodal_attention_mask"]).to(d).long()
        decode_pos_t = torch.from_numpy(decode_inputs["position_ids"]).to(d).long()
        with torch.no_grad():
            decode_out = self.full_model(
                attention_mask=decode_mask_t,
                position_ids=decode_pos_t,
                inputs_embeds=next_embeds_t,
                past_key_values=pkv,
                use_cache=False,
            )
            golden_token = int(decode_out.logits[0, -1].argmax())

        print(f"  [DataBuilder] decode 输入构建完成 (golden_token={golden_token})")
        return decode_inputs, golden_token


class QwenEvaluator(AutoCalibrationEvaluatorBase):
    """AMCT evaluator for Qwen3.5 ONNX models.

    Uses KL divergence (returned as negative for higher=better)
    to detect distribution shifts from INT8 quantization.
    """

    def __init__(
        self,
        model_name: str,
        qwen_path: str,
        img_path: str = "",
        device: str = "npu",
        decode_steps: int = 256,
        expected_acc_loss: float = EXPECTED_KL_DIVERGENCE,
    ):
        super().__init__()
        self.diff = expected_acc_loss
        self.model_name = model_name
        self._decode_steps = decode_steps if model_name == "decoder_decode" else 0
        self._is_decode = self._decode_steps > 0

        self._builder: CalibInputBuilder | None = None
        self.calib_inputs, self.golden_token = self._build_inputs(
            model_name, qwen_path, img_path, device
        )
        self._orig_logits: list[np.ndarray] | np.ndarray | None = None
        self._teacher_tokens: list[int] | None = None
        self._prefill_seq_len: int = 0

        print(f"[Evaluator] Built inputs for {model_name}: {list(self.calib_inputs.keys())}")
        for k, v in self.calib_inputs.items():
            print(f"  {k}: {v.shape} {v.dtype}")
        print(f"  golden_token: {self.golden_token}")
        if self._is_decode:
            print(f"  decode_steps: {self._decode_steps}")

    def _build_inputs(
        self, model_name: str, qwen_path: str, img_path: str, device: str
    ) -> tuple[dict, int]:
        builder = CalibInputBuilder(qwen_path, device)
        self._builder = builder
        if model_name == "decoder_prefill":
            inputs, token, seq_len = builder.build_prefill_inputs(img_path)
            self._prefill_seq_len = seq_len
            return inputs, token
        if model_name == "decoder_decode":
            return builder.build_decode_inputs(img_path)
        raise ValueError(f"Unknown model_name: {model_name}")

    def _get_embedding(self, token_id: int) -> np.ndarray:
        embed = self._builder.full_model.get_input_embeddings()
        t = torch.tensor([[token_id]], device=self._builder.model_device)
        return embed(t).detach().cpu().numpy().astype(np.float16)

    def _decode_multi_step(
        self,
        session,
        initial_inputs: dict,
        teacher_tokens: list[int] | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        inputs = {k: v.copy() for k, v in initial_inputs.items()}
        layer_types = self._builder.text_config.layer_types
        logits_list: list[np.ndarray] = []
        tokens: list[int] = []
        cum_mask = inputs["multimodal_attention_mask"][:, :-1]

        label = "teacher-forced" if teacher_tokens is not None else "greedy"
        print(f"  [ONNXDecode] 开始 {self._decode_steps} 步 {label} 推理...")
        for step in range(self._decode_steps):
            print(f"  [ONNXDecode] step {step + 1}/{self._decode_steps}", end="\r")
            outputs = session.run(None, inputs)
            step_logits = outputs[0]
            logits_list.append(step_logits[0, -1].copy())

            if teacher_tokens is not None and step < len(teacher_tokens):
                next_tok = teacher_tokens[step]
            else:
                next_tok = int(step_logits[0, -1].argmax())
            tokens.append(next_tok)

            inputs["inputs_embeds"] = self._get_embedding(next_tok)
            inputs["position_ids"] = np.array([[inputs["position_ids"][0, 0] + 1]], dtype=np.int64)

            for i, t in enumerate(layer_types):
                base = 1 + i * 2
                if t == "full_attention":
                    k = outputs[base]
                    v = outputs[base + 1]
                    if k.shape[2] > TARGET_SEQ_LEN:
                        k = k[:, :, -TARGET_SEQ_LEN:, :].copy()
                        v = v[:, :, -TARGET_SEQ_LEN:, :].copy()
                    inputs[f"past_{i}_key"] = k
                    inputs[f"past_{i}_value"] = v
                else:
                    inputs[f"past_state_{i}_conv"] = outputs[base]
                    inputs[f"past_state_{i}_rec"] = outputs[base + 1]

            if cum_mask.shape[1] > TARGET_SEQ_LEN:
                cum_mask = cum_mask[:, -TARGET_SEQ_LEN:].copy()
            cum_mask = np.concatenate([cum_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
            inputs["multimodal_attention_mask"] = cum_mask

        print(f"  [ONNXDecode] 完成，共生成 {len(logits_list)} 个 token 的 logits")
        return logits_list, tokens

    def calibration(self, model_file: str):
        session = create_session(model_file)
        if self._is_decode:
            self._decode_multi_step(session, self.calib_inputs)
        else:
            session.run(None, self.calib_inputs)

    def evaluate(self, model_file: str) -> float:
        """Return higher = better metric. Negative KL is used."""
        session = create_session(model_file)

        if self._is_decode:
            if self._orig_logits is None:
                logits_list, tokens = self._decode_multi_step(session, self.calib_inputs)
                self._orig_logits = logits_list
                self._teacher_tokens = tokens
                return 0.0

            teacher_logits, _ = self._decode_multi_step(
                session, self.calib_inputs, teacher_tokens=self._teacher_tokens
            )
            kl_total = 0.0
            n = len(self._orig_logits)
            for step in range(n):
                kl_total += self._kl_divergence(teacher_logits[step], self._orig_logits[step])
            return -kl_total / n  # negative so higher = better

        outputs = session.run(None, self.calib_inputs)
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1, outputs[0].shape[-1])
        if self._prefill_seq_len > 0:
            real_logits = logits[: self._prefill_seq_len]
        else:
            real_logits = logits[-1:]
        n = len(real_logits)

        if self._orig_logits is None:
            self._orig_logits = real_logits.copy()
            return 0.0

        kl_total = 0.0
        for i in range(n):
            kl_total += self._kl_divergence(real_logits[i], self._orig_logits[i])
        return -kl_total / n

    @staticmethod
    def _kl_divergence(quant_logits: np.ndarray, orig_logits: np.ndarray) -> float:
        log_p = torch.log_softmax(torch.from_numpy(orig_logits), dim=-1)
        log_q = torch.log_softmax(torch.from_numpy(quant_logits), dim=-1)
        q = torch.softmax(torch.from_numpy(quant_logits), dim=-1)
        return (q * (log_q - log_p)).sum().item()

    def metric_eval(self, original_metric: float, new_metric: float):
        loss = original_metric - new_metric  # higher metric = better
        ok = loss < self.diff
        print(
            f"[MetricEval] original={original_metric:.6f}, quantized={new_metric:.6f}, "
            f"loss={loss:.6f}, threshold={self.diff}, ok={ok}"
        )
        return ok, loss


def quantize_one(
    model_path: str,
    qwen_path: str,
    save_dir: str,
    img_path: str = "",
    device: str = "cpu",
    expected_acc_loss: float = EXPECTED_KL_DIVERGENCE,
    activation_offset: bool = True,
    decode_steps: int = 256,
):
    clean_temp_dirs(Path(os.getcwd()))
    os.makedirs(save_dir, exist_ok=True)
    model_name = guess_model_name(model_path)

    print(f"\nQuantizing {model_name}: {model_path}, save_dir={save_dir}")

    config_path = os.path.join(save_dir, "quant_config.json")
    amct.create_quant_config(
        config_path,
        model_path,
        skip_layers=[],
        batch_num=1,
        activation_offset=activation_offset,
    )

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
        config_file=config_path,
        record_file=os.path.join(save_dir, "scale_offset_record.txt"),
        save_dir=save_dir,
        strategy="BinarySearch",
        sensitivity="CosineSimilarity",
    )

    print(f"Done. Generated models in {save_dir}:")
    for onnx_file in sorted(Path(save_dir).glob("*.onnx")):
        print(f"  {onnx_file}")


@app.command()
def quantize(
    model_path: str = typer.Option(..., help="Path to the ONNX model"),
    qwen_path: str = typer.Option(..., help="Original Qwen3.5 model directory"),
    save_dir: str = typer.Option("./output/amct_results", help="Output directory"),
    img_path: str = typer.Option(None, help="Calibration image path"),
    device: str = typer.Option("npu", help="Torch device"),
    expected_acc_loss: float = typer.Option(EXPECTED_KL_DIVERGENCE, help="KL threshold"),
    activation_offset: bool = typer.Option(True, help="Enable activation offset"),
    decode_steps: int = typer.Option(256, help="Decode calibration steps"),
):
    """Quantize a single decoder_prefill or decoder_decode ONNX model."""
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
def quantize_all(
    onnx_dir: str = typer.Option(..., help="Directory containing ONNX submodels"),
    qwen_path: str = typer.Option(..., help="Original Qwen3.5 model path"),
    save_dir_str: str = typer.Option(
        "./output/amct_results", "--save-dir", help="Root save directory"
    ),
    img_path: str = typer.Option(None, help="Calibration image path"),
    device: str = typer.Option("npu", help="Torch device"),
    expected_acc_loss: float = typer.Option(EXPECTED_KL_DIVERGENCE, help="KL threshold"),
    activation_offset: bool = typer.Option(True, help="Enable activation offset"),
    decode_steps: int = typer.Option(256, help="Decode calibration steps"),
):
    """Quantize all decoder_prefill and decoder_decode ONNX models."""
    save_dir = Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_img_path = img_path or _default_img_path()

    submodels = {
        "decoder_prefill": Path(onnx_dir)
        / "decoder_model_prefill"
        / "decoder_model_prefill_final.onnx",
        "decoder_decode": Path(onnx_dir)
        / "decoder_model_decode"
        / "decoder_model_decode_final.onnx",
    }

    results = {}
    for name, mp in submodels.items():
        if not mp.exists():
            print(f"Skipping {name}: {mp} not found")
            continue

        sub_save = save_dir / name
        sub_save.mkdir(parents=True, exist_ok=True)
        clean_temp_dirs(save_dir)

        try:
            quantize_one(
                model_path=str(mp),
                qwen_path=qwen_path,
                save_dir=str(sub_save),
                img_path=resolved_img_path,
                device=device,
                expected_acc_loss=expected_acc_loss,
                activation_offset=activation_offset,
                decode_steps=decode_steps,
            )
            results[name] = "success"
        except Exception as e:
            results[name] = f"failed: {e}"
            print(f"[Error] {name} failed: {e}")
        finally:
            clean_temp_dirs(save_dir)

    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    app()

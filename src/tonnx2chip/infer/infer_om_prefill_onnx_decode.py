"""Hybrid Qwen3.5 pipeline: OM prefill + ONNX decode.

This mirrors infer_qwen3p5_om.py but swaps the decoder_model_decode backend
from Ascend OM (ais_bench InferSession) to ONNX Runtime on CPU.

Motivation: isolate whether the decode OM graph introduces precision issues.
Everything upstream (vision encoder, token embedding, 3D M-RoPE, prefill
decoder) still runs on NPU via OM, so any mismatch with the pure-OM or
pure-ONNX reference must originate in the decode stage.

Differences from infer_qwen3p5_om.py:
  - `decode` is an onnxruntime.InferenceSession (CPU), not an OM InferSession.
  - Prefill kernel/linear states are pulled to host as numpy right after
    prefill, since ONNX Runtime consumes numpy arrays (name-keyed dict).
  - Decode inputs use the ONNX positional/named schema:
      inputs_embeds, multimodal_attention_mask, position_ids,
      past_{i}_key / past_{i}_value (full-attn layers),
      past_state_{i}_conv / past_state_{i}_rec (gated-delta layers).
  - KV cache trim to TARGET_SEQ_LEN is done in numpy on the host.
  - No device-tensor zero-copy path; correctness > throughput here.
"""

import itertools
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import typer
from ais_bench.infer.interface import InferSession
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

app = typer.Typer(pretty_exceptions_enable=False)

from tonnx2chip.constants import (
    EMBED_SEQ_LEN,
    FULL_ATTN_LAYERS,
    HIDDEN_SIZE,
    LINEAR_ATTN_LAYERS,
    TARGET_SEQ_LEN,
    VOCAB_SIZE,
)

ONNX_PROVIDER = ["CPUExecutionProvider"]


def reshape_outputs(outs, out_shapes):
    """Reshape flat OM outputs back to declared shapes (handles -1 dims)."""
    result = []
    for o, s in zip(outs, out_shapes):
        o = np.asarray(o)
        known = 1
        neg_idx = -1
        for i, d in enumerate(s):
            if d > 0:
                known *= d
            else:
                neg_idx = i
        if neg_idx >= 0 and known != 0:
            real = [d if d > 0 else o.size // known for d in s]
            o = o.reshape(real)
        elif s != o.shape:
            o = o.reshape(s)
        if not o.flags["C_CONTIGUOUS"]:
            o = np.ascontiguousarray(o)
        result.append(o)
    return result


def run_om(session, inputs, out_shapes, stage="infer"):
    outs = session.infer(inputs)
    outs = [np.asarray(o) for o in outs]
    if out_shapes is not None:
        outs = reshape_outputs(outs, out_shapes)
    return outs


class Qwen35Hybrid:
    def __init__(
        self,
        vit_path: str,
        embed_path: str,
        decoder_prefill_path: str,
        decoder_decode_onnx_path: str,
        qwen_path: str,
        device_id=0,
    ):
        self.device_id = device_id

        print("Loading processor & config...", flush=True)
        self.processor = AutoProcessor.from_pretrained(qwen_path)
        config = AutoConfig.from_pretrained(qwen_path)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.eos_id = self.processor.tokenizer.eos_token_id

        # === NPU side: vision encoder, embedding, prefill (OM) ===
        print("\nLoading OM models (vit / embed / prefill)...", flush=True)
        if vit_path and Path(vit_path).exists():
            self.vit = InferSession(device_id, vit_path)
        else:
            self.vit = None
            print("  Vision encoder skipped (text-only mode)", flush=True)
        self.emb = InferSession(device_id, embed_path)
        prefill_weight_dir = str(Path(decoder_prefill_path).parent / "weight")
        self.prefill = InferSession(device_id, decoder_prefill_path, weight_dir=prefill_weight_dir)
        self.prefill_out_shapes = [tuple(o.shape) for o in self.prefill.get_outputs()]

        # === CPU side: decode (ONNX Runtime) ===
        print(f"\nLoading ONNX decode model: {decoder_decode_onnx_path}", flush=True)
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.intra_op_num_threads = 1
        self.decode = ort.InferenceSession(
            decoder_decode_onnx_path, providers=ONNX_PROVIDER, session_options=so
        )

        # Build the ONNX input-name index: {name -> position} to validate
        # that the OM prefill state layout (layer0 a/b, layer1 a/b, ...) maps
        # onto the ONNX graph's named past_* slots.
        self.decode_input_names = [i.name for i in self.decode.get_inputs()]
        self._embed_cache = {}

    # ---------- upstream (OM) helpers ----------
    def vision_encode(self, pixel_values):
        px = pixel_values.numpy().astype(np.float16)
        out = run_om(self.vit, [px], None)
        return out[0]

    def embed_tokens(self, input_ids_np):
        """Embed up to EMBED_SEQ_LEN tokens. Returns [1, seq_len, hidden]."""
        bs, seq_len = input_ids_np.shape
        if seq_len > EMBED_SEQ_LEN:
            raise ValueError(f"seq_len={seq_len} > embedding fixed len {EMBED_SEQ_LEN}")
        if seq_len == 1:
            tok_id = int(input_ids_np[0, 0])
            cached = self._embed_cache.get(tok_id)
            if cached is not None:
                return cached
        padded = np.zeros((1, EMBED_SEQ_LEN), dtype=np.int64)
        padded[0, :seq_len] = input_ids_np.astype(np.int64)
        out = run_om(self.emb, [padded], None)
        result = np.ascontiguousarray(out[0][:, :seq_len, :])
        if seq_len == 1:
            self._embed_cache[tok_id] = result
        return result

    def compute_3d_rope_torch(self, seq_len, mm_token_type_ids=None, image_grid_thw=None):
        """Replicate Qwen3.5Model.get_rope_index for 3D M-RoPE.

        Returns pos_3d of shape [3, 1, seq_len] int64.
        """
        device = mm_token_type_ids.device if mm_token_type_ids is not None else "cpu"
        pos_3d = torch.zeros((3, 1, seq_len), dtype=torch.int64, device=device)

        if mm_token_type_ids is not None and image_grid_thw is not None:
            token_types = mm_token_type_ids[0, :seq_len]
            diff = torch.cat(
                [torch.tensor([1], device=device), token_types[1:] != token_types[:-1]]
            )
            change_indices = torch.where(diff)[0]
            groups = []
            for i, idx in enumerate(change_indices):
                start = idx.item()
                end = change_indices[i + 1].item() if i + 1 < len(change_indices) else seq_len
                mtype = token_types[start].item()
                groups.append((mtype, start, end))

            grid_iter = iter(image_grid_thw.cpu().numpy())
            cur_pos = 0
            for mtype, start, end in groups:
                length = end - start
                if mtype == 0:
                    row = torch.arange(length, dtype=torch.int64, device=device) + cur_pos
                    pos_3d[:, 0, start:end] = row.unsqueeze(0)
                    cur_pos += length
                elif mtype == 1:
                    grid_thw = next(grid_iter)
                    _, h, w = grid_thw
                    merged_h = h // self.spatial_merge_size
                    merged_w = w // self.spatial_merge_size
                    h_pos = torch.repeat_interleave(
                        torch.arange(merged_h, dtype=torch.int64, device=device), merged_w
                    )
                    w_pos = torch.tile(
                        torch.arange(merged_w, dtype=torch.int64, device=device), (merged_h,)
                    )
                    pos_3d[0, 0, start:end] = (
                        torch.arange(length, dtype=torch.int64, device=device) + cur_pos
                    )
                    pos_3d[1, 0, start:end] = h_pos
                    pos_3d[2, 0, start:end] = w_pos
                    cur_pos += length
        else:
            pos_range = torch.arange(seq_len, dtype=torch.int64, device=device)
            pos_3d[0, 0, :seq_len] = pos_range
            pos_3d[1, 0, :seq_len] = pos_range
            pos_3d[2, 0, :seq_len] = pos_range
        return pos_3d

    def build_prefill_inputs(self, token_embeds, seq_len, pos_3d):
        if seq_len > TARGET_SEQ_LEN:
            raise ValueError(f"seq_len={seq_len} > target {TARGET_SEQ_LEN}")
        emb = np.zeros((1, TARGET_SEQ_LEN, HIDDEN_SIZE), dtype=token_embeds.dtype)
        emb[:, :seq_len, :] = token_embeds[:, :seq_len, :]
        mask = np.zeros((1, TARGET_SEQ_LEN), dtype=np.int64)
        mask[:, :seq_len] = 1
        pad_pos_val = (seq_len - 1) if seq_len > 0 else 0
        pos = np.full((3, 1, TARGET_SEQ_LEN), pad_pos_val, dtype=np.int64)
        pos[:, :, :seq_len] = pos_3d[:, :, :seq_len]
        return mask, pos, emb

    def extract_states_host(self, prefill_outs_raw):
        """Pull prefill KV/linear states to host as numpy.

        Mirrors extract_states() in infer_qwen3p5_om.py, but the resulting
        arrays are host numpy (since ONNX Runtime takes numpy inputs).
        Each entry: [type, a_np, b_np] kept as a list for in-place updates.
        """
        states = [None] * 24
        for layer in range(24):
            base = 1 + layer * 2
            typ = "attn" if layer in FULL_ATTN_LAYERS else "linear"
            a_t = prefill_outs_raw[base]
            b_t = prefill_outs_raw[base + 1]
            a_t.to_host()
            b_t.to_host()
            a = np.ascontiguousarray(np.array(a_t))
            b = np.ascontiguousarray(np.array(b_t))
            states[layer] = [typ, a, b]
        return states

    # ---------- decode (ONNX) ----------
    def build_decode_inputs(self, inp_embeds, decode_attn_mask, decode_pos_ids, states):
        """Build the name-keyed dict consumed by decoder_model_decode.onnx."""
        feeds = {
            "inputs_embeds": inp_embeds,
            "multimodal_attention_mask": decode_attn_mask.astype(np.int64),
            "position_ids": decode_pos_ids.astype(np.int64),
        }
        for layer in range(24):
            typ, a, b = states[layer]
            if typ == "attn":
                feeds[f"past_{layer}_key"] = a
                feeds[f"past_{layer}_value"] = b
            else:
                feeds[f"past_state_{layer}_conv"] = a
                feeds[f"past_state_{layer}_rec"] = b
        return feeds

    def generate(self, prompt, image_path=None, max_new_tokens=128):
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image", "image": image_path}] if image_path else [])
                + [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        px = inputs.get("pixel_values", None)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        bs, seq_len = input_ids.shape
        print(f"Input: {seq_len} tokens, image={'yes' if px is not None else 'no'}")

        # === Step 1: Vision encoder (OM, optional) ===
        if px is not None:
            with tqdm(range(1), desc="Running OM vision encoder..", unit="tok") as pbar:
                img_feats = self.vision_encode(px)
                pbar.set_postfix({"vision_features": img_feats.shape})
        else:
            img_feats = None
        del self.vit

        # === Step 2-4: Token embedding, image merge, 3D M-RoPE ===
        token_embeds = self.embed_tokens(input_ids.numpy())
        print(f"  token embeddings: {token_embeds.shape}")
        if img_feats is not None:
            img_tok_id = self.processor.image_token_id
            mask = input_ids.numpy() == img_tok_id
            token_embeds[mask] = img_feats.astype(token_embeds.dtype)
            print(f"  merged {mask.sum()} image tokens into embeddings")
            del img_feats, mask
        pos_3d = self.compute_3d_rope_torch(seq_len, mm_token_type_ids, image_grid_thw)
        del px, mm_token_type_ids, image_grid_thw, input_ids

        # === Step 5: Prefill (OM, on device) ===
        with tqdm(range(1), desc="Prefilling (OM)", unit="tok") as pbar:
            mask, pos, emb = self.build_prefill_inputs(token_embeds, seq_len, pos_3d)
            prefill_outs_raw = self.prefill.infer(
                [mask, pos, emb],
                out_array=False,
            )
            logits_tensor = prefill_outs_raw[0]
            logits_tensor.to_host()
            logits = np.array(logits_tensor)
            next_tok = int(logits[0, seq_len - 1].argmax())
            pbar.set_postfix(
                {"First token_id": next_tok, "token": self.processor.tokenizer.decode([next_tok])}
            )
            del token_embeds, emb, pos, logits_tensor, logits

        # Pull state tensors to host for ONNX decode
        states = self.extract_states_host(prefill_outs_raw)
        del prefill_outs_raw, self.prefill

        # === Step 6: Decode loop (ONNX, CPU) ===
        gen_ids = []
        max_possible_mask_len = TARGET_SEQ_LEN + max_new_tokens
        decode_mask = np.ones((1, max_possible_mask_len), dtype=np.int64)
        decode_mask[:, : mask.shape[1]] = mask
        decode_mask_len = mask.shape[1]
        del mask

        start = time.time()
        with tqdm(range(max_new_tokens), desc="Decoding (ONNX)", unit="tok") as pbar:
            for step in pbar:
                if next_tok == self.eos_id:
                    break
                gen_ids.append(next_tok)
                cur_pos = seq_len + step
                inp_ids = np.array([[next_tok]], dtype=np.int64)

                # Trim attn KV cache + cumulative mask when exceeding window.
                past_len = states[FULL_ATTN_LAYERS[0]][1].shape[2]
                if past_len > TARGET_SEQ_LEN:
                    for layer in FULL_ATTN_LAYERS:
                        sl = states[layer]
                        sl[1] = np.ascontiguousarray(sl[1][:, :, -TARGET_SEQ_LEN:, :])
                        sl[2] = np.ascontiguousarray(sl[2][:, :, -TARGET_SEQ_LEN:, :])
                    shift = past_len - TARGET_SEQ_LEN
                    decode_mask[:, :TARGET_SEQ_LEN] = decode_mask[:, shift:past_len]
                    decode_mask_len = TARGET_SEQ_LEN
                    past_len = TARGET_SEQ_LEN

                decode_attn_mask = decode_mask[:, : decode_mask_len + 1]
                decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
                inp_embeds = self.embed_tokens(inp_ids)

                feeds = self.build_decode_inputs(
                    inp_embeds, decode_attn_mask, decode_pos_ids, states
                )
                del inp_embeds

                if step == 0:
                    for layer in FULL_ATTN_LAYERS[:3]:
                        print(f"  past_{layer}_k: {states[layer][1].shape}", flush=True)
                    for layer in LINEAR_ATTN_LAYERS[:3]:
                        print(
                            f"  past_state_{layer}_conv: {states[layer][1].shape}",
                            flush=True,
                        )
                    # sanity check: all required ONNX names are fed
                    missing = [n for n in self.decode_input_names if n not in feeds]
                    if missing:
                        raise RuntimeError(
                            f"ONNX decode graph expects inputs not provided: {missing}"
                        )

                decode_outputs = self.decode.run(None, feeds)
                next_tok = int(decode_outputs[0][0, -1].argmax())

                # Update layer states (host numpy; copy to avoid aliasing).
                for layer in range(24):
                    base = 1 + layer * 2
                    sl = states[layer]
                    sl[1] = np.ascontiguousarray(decode_outputs[base])
                    sl[2] = np.ascontiguousarray(decode_outputs[base + 1])

                decode_mask_len += 1

                pbar.set_postfix(
                    {"next_tok": next_tok, "token": self.processor.tokenizer.decode([next_tok])}
                )

        elapsed = time.time() - start

        for layer in FULL_ATTN_LAYERS[:3]:
            print(f"  past_{layer}_k: {states[layer][1].shape}", flush=True)
        for layer in LINEAR_ATTN_LAYERS[:3]:
            print(
                f"  past_state_{layer}_conv: {states[layer][1].shape}",
                flush=True,
            )

        return gen_ids, elapsed


IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"
PROMPT = "Describe this image."


@app.command()
def infer(
    vit_path: str = typer.Option(..., help="VIT OM path (use 'none' to disable for text-only)"),
    embedding_path: str = typer.Option(..., help="Embedding OM path"),
    decoder_prefill_path: str = typer.Option(..., help="Decoder prefill OM path"),
    decoder_decode_onnx_path: str = typer.Option(
        ..., help="decoder_model_decode.onnx path (used in place of OM decode)"
    ),
    qwen_path: str = typer.Option(..., help="Original Qwen model dir"),
    prompt: str = typer.Option(PROMPT, help="Prompt"),
    image_path: str = typer.Option(IMAGE_PATH, help="Image path (or 'none')"),
    max_new_tokens: int = typer.Option(10, help="Max new tokens"),
    device_id: int = typer.Option(0, help="NPU device ID"),
):
    for f in [embedding_path, decoder_prefill_path, decoder_decode_onnx_path]:
        assert Path(f).exists(), f"Missing {f}"

    vit = None if (vit_path.lower() == "none" or not Path(vit_path).exists()) else vit_path
    img = None if image_path.lower() == "none" else image_path

    runner = Qwen35Hybrid(
        vit_path=vit,
        embed_path=embedding_path,
        decoder_prefill_path=decoder_prefill_path,
        decoder_decode_onnx_path=decoder_decode_onnx_path,
        qwen_path=qwen_path,
        device_id=device_id,
    )
    tokens, elapsed = runner.generate(prompt, img, max_new_tokens)
    text = runner.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    print("════════" * 4)
    print(f"Hybrid OM-prefill + ONNX-decode ({len(tokens)} tok, {elapsed:.2f}s):")
    print(text)
    print("════════" * 4)


if __name__ == "__main__":
    app()

"""End-to-end Qwen3.5 validation pipeline using Ascend OM models.

This mirrors src/qwen_onnx/val_qwen3p5_with_onnx.py but runs inference on
NPU via ais_bench InferSession instead of ONNX Runtime on CPU.

Key differences from the ONNX variant:
  - OM inputs are positional lists (ais_bench API), not name-keyed dicts.
  - OM outputs may be flattened; we reshape via session.get_outputs() shapes.
  - prefill uses decoder_prefill_pad2slice.om (Pad nodes rewritten to Slice
    to avoid the EZ9999 te_padv3 runtime error).
  - decode uses decoder_decode.om and maintains per-layer KV/linear states
    across decode steps, trimming to target_seq_len when exceeded.
"""

import itertools
import time
from pathlib import Path

import numpy as np
from ais_bench.infer.interface import InferSession
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor


from tonnx2chip.constants import (
    EMBED_SEQ_LEN,
    FULL_ATTN_LAYERS,
    HIDDEN_SIZE,
    LINEAR_ATTN_LAYERS,
    TARGET_SEQ_LEN,
    VOCAB_SIZE,
)


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


class Qwen35OM:
    def __init__(
        self,
        vit_path: str |None,
        embed_path: str,
        decoder_prefill_path: str,
        decoder_decode_path: str,
        qwen_path: str,
        device_id=0,
    ):
        self.device_id = device_id

        print("Loading processor & config...", flush=True)
        self.processor = AutoProcessor.from_pretrained(qwen_path)
        config = AutoConfig.from_pretrained(qwen_path)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.eos_id = self.processor.tokenizer.eos_token_id

        print("\nLoading OM models...", flush=True)
        if vit_path and Path(vit_path).exists():
            self.vit = InferSession(device_id, vit_path)
        else:
            self.vit = None
            print("  Vision encoder skipped (text-only mode)", flush=True)
        self.emb = InferSession(device_id, embed_path)
        prefill_weight_dir = str(Path(decoder_prefill_path).parent / "weight")
        decode_weight_dir = str(Path(decoder_decode_path).parent / "weight")
        self.prefill = InferSession(device_id, decoder_prefill_path, weight_dir=prefill_weight_dir)
        self.decode = InferSession(device_id, decoder_decode_path, weight_dir=decode_weight_dir)

        # Cache declared output shapes (positional InferSession I/O)
        self.prefill_out_shapes = [tuple(o.shape) for o in self.prefill.get_outputs()]
        self.decode_out_shapes = [tuple(o.shape) for o in self.decode.get_outputs()]

        self._embed_cache = {}

    def vision_encode(self, pixel_values):
        px = pixel_values.numpy().astype(np.float16)
        out = run_om(self.vit, [px], None)
        return out[0]

    def embed_tokens(self, input_ids_np):
        """Embed up to EMBED_SEQ_LEN tokens. Returns [1, seq_len, hidden]."""
        bs, seq_len = input_ids_np.shape
        if seq_len > EMBED_SEQ_LEN:
            raise ValueError(f"seq_len={seq_len} > embedding fixed len {EMBED_SEQ_LEN}")
        # Single-token decode path: cache by token id
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

    def compute_3d_rope(self, seq_len, mm_token_type_ids=None, image_grid_thw=None):
        """Replicate Qwen3.5Model.get_rope_index for 3D M-RoPE.

        Returns pos_3d of shape [3, 1, seq_len] int64.
        """
        pos_3d = np.zeros((3, 1, seq_len), dtype=np.int64)
        if mm_token_type_ids is not None and image_grid_thw is not None:
            token_types = mm_token_type_ids[0, :seq_len].tolist()
            groups = []
            for key, group in itertools.groupby(enumerate(token_types), lambda x: x[1]):
                group_list = list(group)
                groups.append((key, group_list[0][0], group_list[-1][0] + 1))
            grid_iter = iter(image_grid_thw.numpy())
            cur_pos = 0
            for mtype, start, end in groups:
                length = end - start
                if mtype == 0:
                    row = np.arange(length, dtype=np.int64) + cur_pos
                    pos_3d[:, 0, start:end] = row[np.newaxis, :]
                    cur_pos += length
                elif mtype == 1:
                    grid_thw = next(grid_iter)
                    _, h, w = grid_thw
                    merged_h = h // self.spatial_merge_size
                    merged_w = w // self.spatial_merge_size
                    h_pos = np.repeat(np.arange(merged_h, dtype=np.int64), merged_w)
                    w_pos = np.tile(np.arange(merged_w, dtype=np.int64), merged_h)
                    pos_3d[0, 0, start:end] = np.arange(length, dtype=np.int64) + cur_pos
                    pos_3d[1, 0, start:end] = h_pos
                    pos_3d[2, 0, start:end] = w_pos
                    cur_pos += length
        else:
            pos_3d[0, 0, :seq_len] = np.arange(seq_len, dtype=np.int64)
            pos_3d[1, 0, :seq_len] = np.arange(seq_len, dtype=np.int64)
            pos_3d[2, 0, :seq_len] = np.arange(seq_len, dtype=np.int64)
        return pos_3d

    def build_prefill_inputs(self, token_embeds, seq_len, pos_3d):
        """Pad embeddings, mask, and positions to TARGET_SEQ_LEN (single allocation)."""
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

    def extract_states(self, prefill_outputs):
        """Extract per-layer states from prefill outputs.

        Outputs are: [logits, (layer0 a, layer0 b), (layer1 a, layer1 b), ...]
        Full-attn layers store (key, value); linear layers store (conv, rec).

        Returns a list of mutable [type, a, b] for in-place updates.
        """
        states = [None] * 24
        for layer in range(24):
            base = 1 + layer * 2
            typ = "attn" if layer in FULL_ATTN_LAYERS else "linear"
            states[layer] = [typ, prefill_outputs[base], prefill_outputs[base + 1]]
        return states

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

        # === Step 1: Vision encoder (optional) ===
        if px is not None:
            with tqdm(range(1), desc="Running OM vision encoder..", unit="tok") as pbar:
                img_feats = self.vision_encode(px)
                pbar.set_postfix({"vision_features": img_feats.shape})
        else:
            img_feats = None

        # === Step 2-4: Token embedding, image merge, 3D M-RoPE ===
        token_embeds = self.embed_tokens(input_ids.numpy())
        print(f"  token embeddings: {token_embeds.shape}")
        if img_feats is not None:
            img_tok_id = self.processor.image_token_id
            mask = input_ids.numpy() == img_tok_id
            token_embeds[mask] = img_feats.astype(token_embeds.dtype)
            print(f"  merged {mask.sum()} image tokens into embeddings")
            del img_feats, mask
        pos_3d = self.compute_3d_rope(seq_len, mm_token_type_ids, image_grid_thw)
        del px, mm_token_type_ids, image_grid_thw, input_ids

        # === Step 5: Prefill (keep KV cache on device) ===
        with tqdm(range(1), desc="Prefilling", unit="tok") as pbar:
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
            del token_embeds, emb, pos
            del logits_tensor
            del logits

        # KV cache stays on device; state list holds [type, device_tensor_k, device_tensor_v]
        states = self.extract_states(prefill_outs_raw)
        del prefill_outs_raw

        # === Step 6: Decode loop (zero-copy KV cache) ===
        gen_ids = []
        # Pre-allocate decode mask to avoid per-step concatenation
        max_possible_mask_len = TARGET_SEQ_LEN + max_new_tokens
        decode_mask = np.ones((1, max_possible_mask_len), dtype=np.int64)
        decode_mask[:, : mask.shape[1]] = mask
        decode_mask_len = mask.shape[1]
        del mask

        start = time.time()
        with tqdm(range(max_new_tokens), desc="Decoding", unit="tok") as pbar:
            for step in pbar:
                if next_tok == self.eos_id:
                    break
                gen_ids.append(next_tok)
                cur_pos = seq_len + step
                inp_ids = np.array([[next_tok]], dtype=np.int64)

                past_len = states[FULL_ATTN_LAYERS[0]][1].shape[2]
                if past_len > TARGET_SEQ_LEN:
                    for layer in FULL_ATTN_LAYERS:
                        sl = states[layer]
                        sl[1].to_host()
                        k = np.ascontiguousarray(np.array(sl[1])[:, :, -TARGET_SEQ_LEN:, :])
                        sl[1] = self.decode.create_tensor_from_arrays_to_device(k)
                        sl[2].to_host()
                        v = np.ascontiguousarray(np.array(sl[2])[:, :, -TARGET_SEQ_LEN:, :])
                        sl[2] = self.decode.create_tensor_from_arrays_to_device(v)
                    shift = past_len - TARGET_SEQ_LEN
                    decode_mask[:, :TARGET_SEQ_LEN] = decode_mask[:, shift:past_len]
                    decode_mask_len = TARGET_SEQ_LEN
                    past_len = TARGET_SEQ_LEN

                decode_attn_mask = decode_mask[:, : decode_mask_len + 1]
                decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)
                inp_embeds = self.embed_tokens(inp_ids)

                # Convert small inputs to device tensors
                inp_embeds_dev = self.decode.create_tensor_from_arrays_to_device(inp_embeds)
                del inp_embeds
                mask_dev = self.decode.create_tensor_from_arrays_to_device(decode_attn_mask)
                pos_dev = self.decode.create_tensor_from_arrays_to_device(decode_pos_ids)

                decode_inputs_dev = [inp_embeds_dev, mask_dev, pos_dev]
                for layer in range(24):
                    sl = states[layer]
                    decode_inputs_dev.append(sl[1])
                    decode_inputs_dev.append(sl[2])

                decode_outs_raw = self.decode.run_from_tensors(decode_inputs_dev, out_array=False)

                logits_dev = decode_outs_raw[0]
                logits_dev.to_host()
                next_tok = int(np.array(logits_dev)[0, -1].argmax())

                for layer in range(24):
                    base = 1 + layer * 2
                    sl = states[layer]
                    sl[1] = decode_outs_raw[base]
                    sl[2] = decode_outs_raw[base + 1]

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

def infer(
    vit_path: str,
    embedding_path: str,
    decoder_prefill_path: str,
    decoder_decode_path: str,
    qwen_path: str,
    prompt: str = PROMPT,
    image_path: str = str(IMAGE_PATH),
    max_new_tokens: int = 64,
    device_id: int = 0,
):
    for f in [embedding_path, decoder_prefill_path, decoder_decode_path]:
        assert Path(f).exists(), f"Missing {f}"

    vit = None if (vit_path.lower() == "none" or not Path(vit_path).exists()) else vit_path
    img = None if image_path.lower() == "none" else image_path

    runner = Qwen35OM(
        vit_path=vit,
        embed_path=embedding_path,
        decoder_prefill_path=decoder_prefill_path,
        decoder_decode_path=decoder_decode_path,
        qwen_path=qwen_path,
        device_id=device_id,
    )
    tokens, elapsed = runner.generate(prompt, img, max_new_tokens)
    text = runner.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    print("════════" * 4)
    print(text)
    print("════════" * 4)
    breakpoint()


if __name__ == "__main__":
    # infer(
    #     vit_path="/data2/Qwen3.5-2B-Edge/om-int8/vision_encoder.om",
    #     embedding_path="/data2/Qwen3.5-2B-Edge/om-int8/embedding.om",
    #     decoder_prefill_path="/data2/Qwen3.5-2B-Edge/om-int8/decoder_model_prefill/decoder_model_prefill.om",
    #     decoder_decode_path="/data2/Qwen3.5-2B-Edge/om-int8/decoder_model_decode/decoder_model_decode.om",
    #     qwen_path="/data2/Qwen3.5-2B",
    # )
    infer(
        vit_path="/data/workspace/weight/Qwen3.5-2B-Edge/om-310p/vision_encoder.om",
        embedding_path="/data/workspace/weight/Qwen3.5-2B-Edge/om-310p/embedding.om",
        decoder_prefill_path="/data/workspace/weight/Qwen3.5-2B-Edge/om-310p/decoder_model_prefill/decoder_model_prefill.om",
        decoder_decode_path="/data/workspace/weight/Qwen3.5-2B-Edge/om-310p/decoder_model_decode/decoder_model_decode.om",
        qwen_path="/data/workspace/weight/Qwen3.5-2B",
    )

import itertools
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import typer
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

app = typer.Typer(pretty_exceptions_enable=False)
ONNX_Provider = ["CPUExecutionProvider"]


class Qwen35ONNX:
    def __init__(
        self,
        vit_path: str,
        embed_path: str,
        decoder_prefill_path: str,
        decoder_decode_parh: str,
        qwen_path: str,
    ):
        self.processor = AutoProcessor.from_pretrained(qwen_path)

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.intra_op_num_threads = 1

        print("Loading vision_encoder.onnx...")
        self.vis = ort.InferenceSession(vit_path, providers=ONNX_Provider, session_options=so)
        print("Loading embedding.onnx...")
        self.emb = ort.InferenceSession(embed_path, providers=ONNX_Provider, session_options=so)
        print("Loading decoder_model_prefill.onnx...")
        self.prefill = ort.InferenceSession(
            decoder_prefill_path, providers=ONNX_Provider, session_options=so
        )
        print("Loading decoder_model_decode.onnx...")
        self.decode = ort.InferenceSession(
            decoder_decode_parh, providers=ONNX_Provider, session_options=so
        )

        self._full_attn_layers = [3, 7, 11, 15, 19, 23]
        self._linear_attn_layers = [i for i in range(24) if i not in self._full_attn_layers]
        print(f"Full attention layers: {self._full_attn_layers}")
        print(f"Gated-delta layers: {self._linear_attn_layers}")

        self.vocab_size = 248320
        self.hidden_size = 2048
        self.target_seq_len = 256
        self.eos_id = self.processor.tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(qwen_path)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def generate(self, prompt: str, image_path: str | None = None, max_new_tokens: int = 128):
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
        _attn_mask = inputs["attention_mask"]
        px = inputs.get("pixel_values", None)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        bs, seq_len = input_ids.shape
        print(f"Input: {seq_len} tokens, image={'yes' if px is not None else 'no'}")

        # === Step 1: Vision encoder (ONNX) ===
        if px is not None:
            print("Running ONNX vision encoder...")
            out_vis = self.vis.run(
                None,
                {
                    "pixel_values": px.numpy().astype(np.float16),
                },
            )
            img_feats = out_vis[0]
            print(f"  vision_features: {img_feats.shape}")
        else:
            img_feats = None

        # === Step 2: Embedding (ONNX, pad to fixed 128) ===
        EXPORT_SEQ_LEN = 128
        if seq_len > EXPORT_SEQ_LEN:
            raise ValueError(
                f"Input seq_len={seq_len} > embedding ONNX fixed seq_len={EXPORT_SEQ_LEN}"
            )
        padded_ids = np.zeros((1, EXPORT_SEQ_LEN), dtype=np.int64)
        padded_ids[0, :seq_len] = input_ids.numpy().astype(np.int64)
        print(f"Running ONNX embedding (padded {seq_len} -> {EXPORT_SEQ_LEN})...")
        token_embeds = self.emb.run(None, {"input_ids": padded_ids})[0]
        token_embeds = token_embeds[:, :seq_len, :]
        print(f"  token embeddings: {token_embeds.shape}")

        # === Step 3: Merge image features ===
        if img_feats is not None:
            img_tok_id = self.processor.image_token_id
            mask = input_ids.numpy() == img_tok_id
            token_embeds[mask] = img_feats.astype(token_embeds.dtype)
            print(f"  merged {mask.sum()} image tokens into embeddings")

        # === Step 4: Compute 3D M-RoPE position_ids ===
        # Replicates Qwen3.5Model.get_rope_index logic for spatial encoding
        prefill_len = min(seq_len, self.target_seq_len)
        pos_3d = np.zeros((3, 1, prefill_len), dtype=np.int64)
        if img_feats is not None and mm_token_type_ids is not None:
            token_types = mm_token_type_ids[0, :prefill_len].tolist()
            # Group consecutive same-type tokens
            groups = []
            for key, group in itertools.groupby(enumerate(token_types), lambda x: x[1]):
                group_list = list(group)
                groups.append((key, group_list[0][0], group_list[-1][0] + 1))
            grid_iter = iter(image_grid_thw.numpy())
            cur_pos = 0
            for mtype, start, end in groups:
                length = end - start
                if mtype == 0:
                    # Text tokens: all 3 dims = linear position
                    row = np.arange(length, dtype=np.int64) + cur_pos
                    pos_3d[:, 0, start:end] = row[np.newaxis, :]
                    cur_pos += length
                elif mtype == 1:
                    # Image tokens: dim0=pos, dim1=height, dim2=width
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
            # Text-only: all 3 dims = linear position
            pos_3d[0, 0, :prefill_len] = np.arange(prefill_len, dtype=np.int64)
            pos_3d[1, 0, :prefill_len] = np.arange(prefill_len, dtype=np.int64)
            pos_3d[2, 0, :prefill_len] = np.arange(prefill_len, dtype=np.int64)

        # Pad to target_seq_len
        if prefill_len < self.target_seq_len:
            padding_len = self.target_seq_len - prefill_len
            pad_pos = np.full((3, 1, padding_len), prefill_len - 1, dtype=np.int64)
            pos_3d = np.concatenate([pos_3d, pad_pos], axis=2)

            pad_emb = np.zeros((1, padding_len, self.hidden_size), dtype=token_embeds.dtype)
            multimodal_embeddings = np.concatenate([token_embeds, pad_emb], axis=1)

            pad_mask = np.zeros((1, padding_len), dtype=np.int64)
            multimodal_attention_mask = np.concatenate(
                [np.ones((1, prefill_len), dtype=np.int64), pad_mask], axis=1
            )
        else:
            multimodal_embeddings = token_embeds[:, :prefill_len, :]
            multimodal_attention_mask = np.ones((1, prefill_len), dtype=np.int64)

        # === Step 5: ONNX prefill ===
        print("Running ONNX prefill...")
        prefill_outputs = self.prefill.run(
            None,
            {
                "multimodal_attention_mask": multimodal_attention_mask,
                "position_ids": pos_3d,
                "multimodal_embeddings": multimodal_embeddings,
            },
        )
        logits = prefill_outputs[0]
        # last REAL token position -> next-token logits
        next_tok = int(logits[0, prefill_len - 1].argmax())

        # Extract all layer states from prefill output
        # prefill_outputs[0] = logits, then for each layer 0..23: 2 outputs
        # gated-delta: conv_state, rec_state; full-attn: key, value
        states = {}
        for layer in range(24):
            base = 1 + layer * 2
            if layer in self._full_attn_layers:
                states[layer] = (
                    "attn",
                    prefill_outputs[base],
                    prefill_outputs[base + 1],
                )
            else:
                states[layer] = (
                    "linear",
                    prefill_outputs[base],
                    prefill_outputs[base + 1],
                )

        print(f"  first token: {next_tok} ({self.processor.tokenizer.decode([next_tok])!r})")

        # === Step 6: ONNX decode loop ===
        # KV cache (attn layers only) is trimmed to at most target_seq_len entries.
        gen_ids = []
        decode_mask_cum = multimodal_attention_mask.copy()
        for step in tqdm(range(max_new_tokens), desc="Generating tokens"):
            if next_tok == self.eos_id:
                break
            gen_ids.append(next_tok)
            cur_pos = seq_len + step
            _inp_ids = np.array([[next_tok]], dtype=np.int64)

            # Embed the single token using the ONNX embedding model
            padded_inp = np.zeros((1, EXPORT_SEQ_LEN), dtype=np.int64)
            padded_inp[0, 0] = next_tok
            inp_embeds = self.emb.run(None, {"input_ids": padded_inp})[0][:, 0:1, :]

            # Trim attn KV cache + cumulative mask (linear states are fixed-size, no trim needed)
            past_len = states[self._full_attn_layers[0]][1].shape[2]
            if past_len > self.target_seq_len:
                for layer in self._full_attn_layers:
                    typ, k, v = states[layer]
                    states[layer] = (
                        typ,
                        k[:, :, -self.target_seq_len :, :].copy(),
                        v[:, :, -self.target_seq_len :, :].copy(),
                    )
                decode_mask_cum = decode_mask_cum[:, -self.target_seq_len :].copy()
                past_len = self.target_seq_len

            # Build mask: cumulative (trimmed) + 1 for the new token
            decode_attn_mask = np.concatenate(
                [decode_mask_cum, np.ones((1, 1), dtype=np.int64)], axis=1
            )
            decode_pos_ids = np.array([[cur_pos]], dtype=np.int64)

            # Build decode inputs for all 24 layers
            decode_inputs = {
                "inputs_embeds": inp_embeds,
                "multimodal_attention_mask": decode_attn_mask,
                "position_ids": decode_pos_ids,
            }
            for layer in range(24):
                typ, a, b = states[layer]
                if typ == "attn":
                    decode_inputs[f"past_{layer}_key"] = a
                    decode_inputs[f"past_{layer}_value"] = b
                else:
                    decode_inputs[f"past_state_{layer}_conv"] = a
                    decode_inputs[f"past_state_{layer}_rec"] = b

            if step == 0:
                for layer in self._full_attn_layers:
                    print(f"  past_{layer}_k: {states[layer][1].shape}", flush=True)
                for layer in self._linear_attn_layers[:3]:
                    print(
                        f"  past_state_{layer}_conv: {states[layer][1].shape}",
                        flush=True,
                    )

            decode_outputs = self.decode.run(None, decode_inputs)
            next_tok = int(decode_outputs[0][0, -1].argmax())

            # Update all layer states from decode outputs
            # decode_outputs[0] = logits, then for each layer 0..23: 2 outputs
            for layer in range(24):
                base = 1 + layer * 2
                typ, _, _ = states[layer]
                new_a, new_b = decode_outputs[base], decode_outputs[base + 1]
                states[layer] = (typ, new_a.copy(), new_b.copy())

            # Update cumulative mask
            decode_mask_cum = np.concatenate(
                [decode_mask_cum, np.ones((1, 1), dtype=np.int64)], axis=1
            )

        return gen_ids


IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"
PROMPT = "Describe this image."


@app.command()
def main(
    vit_path: str = typer.Option(..., help="Dir with ONNX files"),
    embed_path: str = typer.Option(..., help="Dir with ONNX files"),
    decoder_prefill_path: str = typer.Option(..., help="Dir with ONNX files"),
    decoder_decode_path: str = typer.Option(..., help="Dir with ONNX files"),
    qwen_path: str = typer.Option(..., help="Original Qwen model dir"),
    prompt: str = typer.Option(PROMPT, help="Prompt"),
    image_path: str = typer.Option(IMAGE_PATH, help="Image path"),
    max_new_tokens: int = typer.Option(20, help="Max new tokens"),
):
    runner = Qwen35ONNX(vit_path, embed_path, decoder_prefill_path, decoder_decode_path, qwen_path)
    start = time.time()
    tokens = runner.generate(prompt, image_path, max_new_tokens)
    elapsed = time.time() - start
    text = runner.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    print(
        f"\nGenerated from onnx ({len(tokens)} tok, {elapsed:.2f}s, {len(tokens) / elapsed:.2f} tok/s):"
    )
    print(text)

    # === Pure PyTorch baseline (for comparison) ===
    print("\n=== Pure PyTorch baseline ===")
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen_path,
        local_files_only=False,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    messages = [
        {
            "role": "user",
            "content": ([{"type": "image", "image": image_path}] if image_path else [])
            + [{"type": "text", "text": prompt}],
        }
    ]
    pt_inputs = runner.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    start_pt = time.time()
    pt_ids = full.generate(
        **pt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )[0]
    pt_elapsed = time.time() - start_pt
    pt_tokens = pt_ids[len(pt_inputs["input_ids"][0]) :].tolist()
    pt_text = runner.processor.tokenizer.decode(pt_tokens, skip_special_tokens=True)
    print(f"Generated from pytorch ({len(pt_tokens)} tok, {pt_elapsed:.2f}s):")
    print(pt_text)

    # Compare
    if tokens == pt_tokens:
        print("\n✅ ONNX pipeline output matches pure PyTorch!")
    else:
        print(f"\n⚠️  Mismatch: ONNX={len(tokens)} tok, PT={len(pt_tokens)} tok")
        print(f"  ONNX: {tokens}")
        print(f"  PyTorch: {pt_tokens}")
        common = set(tokens) & set(pt_tokens)
        print(f"  Common tokens: {len(common)}/{max(len(tokens), len(pt_tokens))}")


if __name__ == "__main__":
    app()

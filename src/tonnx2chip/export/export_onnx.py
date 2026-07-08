import os
import types
import warnings
from pathlib import Path
from typing import Sequence

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Monkey-patch: TorchScript tracing converts tensor.shape[N] to 0-D tensors,
# but transformers 5.4.0 _preprocess_mask_arguments returns q_length/kv_length
# as 0-D tensors, which then crashes in sdpa_mask BC check (line 492).
# Fix: ensure q_length/kv_length/q_offset/kv_offset are Python ints.
import transformers.masking_utils as _mu
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5VisionModel,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    l2norm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_mask_to_padding_states as _apply_mask,
)

# ==================== Patch
_orig_preprocess = _mu._preprocess_mask_arguments
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


class Qwen3_5VisionModelOpt(Qwen3_5VisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states


def _patched_preprocess(*args, **kwargs):
    result = _orig_preprocess(*args, **kwargs)
    (
        early_exit,
        attention_mask,
        packed_sequence_mask,
        q_length,
        kv_length,
        q_offset,
        kv_offset,
    ) = result
    if isinstance(q_length, torch.Tensor):
        q_length = int(q_length.item())
    if isinstance(kv_length, torch.Tensor):
        kv_length = int(kv_length.item())
    if isinstance(q_offset, torch.Tensor):
        q_offset = int(q_offset.item())
    if isinstance(kv_offset, torch.Tensor):
        kv_offset = int(kv_offset.item())
    return (
        early_exit,
        attention_mask,
        packed_sequence_mask,
        q_length,
        kv_length,
        q_offset,
        kv_offset,
    )


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    **kwargs,
) -> tuple[torch.Tensor]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    **kwargs,
) -> tuple[torch.Tensor]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_causal_conv1d_update_onnx(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_conv_state = hidden_states_new[:, :, -state_len:].contiguous()
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out, new_conv_state


def gated_delta_net_forward_onnx(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
):
    hidden_states = _apply_mask(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1
    )

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv, new_conv_state = torch_causal_conv1d_update_onnx(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
        cache_params.conv_states[self.layer_idx] = new_conv_state
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


_mu._preprocess_mask_arguments = _patched_preprocess


# ==================== Wrapper Classes ====================
class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, visual) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        return Qwen3_5VisionModelOpt.forward(self.visual, pixel_values, image_grid_thw)


class Qwen35PrefillWrapper(nn.Module):
    """Wrapper for prefill model to enable ONNX export."""

    def __init__(self, causal_lm: Qwen3_5ForCausalLM) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        config = causal_lm.config if hasattr(causal_lm, "config") else causal_lm.model.config
        self.layer_types = config.layer_types

    def forward(
        self,
        multimodal_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
    ):
        """Forward pass for prefill stage."""
        outputs = self.causal_lm(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            use_cache=True,
        )
        pkv = outputs.past_key_values
        result = [outputs.logits]
        for i, t in enumerate(self.layer_types):
            if t == "full_attention":
                result.extend([pkv.key_cache[i], pkv.value_cache[i]])
            else:
                result.extend([pkv.conv_states[i], pkv.recurrent_states[i]])
        return tuple(result)


class Qwen35DecoderWrapper(nn.Module):
    """Wrapper for decoder model to enable ONNX export."""

    def __init__(self, causal_lm: Qwen3_5ForCausalLM) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        self.layer_types = causal_lm.model.config.layer_types

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *flat_tensors,
    ):
        """Forward pass for decoder stage.

        接收已过 embedding 层的 inputs_embeds（由单独导出的 embedding 模型
        预先计算），从而避免 decode 图内重复携带 embed_tokens 权重（约 970MB）。

        flat_tensors layout (chronological layer 0..23):
          for each layer i:
            full_attention: [key, value]
            linear_attention: [conv, rec]
          Total = 51 tensors (12 KV pairs + 18 conv + 18 rec)
        """
        config = self.causal_lm.model.config

        pkv = Qwen3_5DynamicCache(config)
        fi = 0
        for i, t in enumerate(self.layer_types):
            if t == "full_attention":
                pkv.key_cache[i] = flat_tensors[fi]
                pkv.value_cache[i] = flat_tensors[fi + 1]
                fi += 2
            else:
                pkv.conv_states[i] = flat_tensors[fi]
                pkv.recurrent_states[i] = flat_tensors[fi + 1]
                fi += 2

        outputs = self.causal_lm(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            labels=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        out_pkv = outputs.past_key_values
        present = [outputs.logits]
        for i, t in enumerate(self.layer_types):
            if t == "full_attention":
                present.extend([out_pkv.key_cache[i], out_pkv.value_cache[i]])
            else:
                present.extend([out_pkv.conv_states[i], out_pkv.recurrent_states[i]])
        return tuple(present)


# 1. 导出 Embedding 层
class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


def get_model_input(qwen_path, imgs_paths, text, batch_size, device):
    processor = AutoProcessor.from_pretrained(qwen_path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img_path} for img_path in imgs_paths]
            + [{"type": "text", "text": text}],
        }
    ]

    # Check context
    assert len(messages) == batch_size, (
        f"messages number should be equal to batch_size, but now messages batch size = {len(messages)}, config batch_size = {batch_size}"
    )

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def onnx_export_all_tensors_to_one_file(
    model: nn.Module,
    args: tuple,
    onnx_path: Path,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
):
    tmp_path = onnx_path
    all_tensors_to_one_file_flag = False

    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    threshold = 2 * 1024**3  # 2GB
    if total_bytes > threshold:
        all_tensors_to_one_file_flag = True
        tmp_path = Path("/tmp") / onnx_path.name

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        with torch.no_grad():
            torch.onnx.export(
                model,
                args,
                tmp_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=14,
                export_params=True,
                do_constant_folding=True,
                dynamo=False,
            )

    if all_tensors_to_one_file_flag:
        print("Mergeing all tensors to one file")
        onnx_model = onnx.load(tmp_path)

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
        print(f"Saved onnx to {onnx_path}")
    return onnx_path


def export_vit(model: nn.Module, torch_input, export_path: Path):
    pbar = tqdm(total=2, desc="处理中", disable=True)
    pbar.set_description("Exporting model to onnx")

    vision_encoder = VisionEncoderWrapper(model.visual)
    vision_encoder.eval()
    # seq_len x 1536
    dummy_pixel_values = (
        torch_input["pixel_values"].clone().to(dtype=torch.float16)
    )  # seq_len x 1536
    dummy_grid_thw = torch_input["image_grid_thw"].clone()  # img_num x 3

    print(f"\n{dummy_pixel_values.shape=}\n{dummy_grid_thw.shape=}")

    onnx_export_all_tensors_to_one_file(
        vision_encoder,
        (dummy_pixel_values, dummy_grid_thw),
        export_path / "vision_encoder.onnx",
        input_names=["pixel_values", "grid_thw"],
        output_names=["vision_features"],
    )
    print("✅ 视觉编码器导出完成")
    pbar.update(1)

    embedding_model = model.get_input_embeddings().eval()
    dummy_ids = torch.randint(0, 248320, (1, 128), dtype=torch.int64)

    print(f"\n{dummy_ids.shape=}")

    onnx_export_all_tensors_to_one_file(
        embedding_model,
        dummy_ids,
        export_path / "embedding.onnx",
        input_names=["input_ids"],
        output_names=["embeddings"],
    )
    print("✅ Embedding 层导出完成")
    pbar.update(1)
    pbar.close()


def vit2llm(model: nn.Module, dummy_inputs):
    dummy_input_ids = dummy_inputs["input_ids"]
    dummy_pixel_values = dummy_inputs["pixel_values"]
    dummy_attention_mask = dummy_inputs["attention_mask"]
    dummy_grid_thw = dummy_inputs["image_grid_thw"].clone()
    mm_token_type_ids = dummy_inputs["mm_token_type_ids"].clone()
    print(
        f"\n{dummy_input_ids.shape=}\n{dummy_pixel_values.shape=}\n{dummy_attention_mask.shape=}",
        flush=True,
    )

    inputs_embeds = model.get_input_embeddings()(dummy_input_ids)
    if dummy_pixel_values is not None:
        image_outputs = model.get_image_features(
            dummy_pixel_values, dummy_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(
            dummy_input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    position_ids = model.compute_3d_position_ids(
        input_ids=dummy_input_ids,
        image_grid_thw=dummy_grid_thw,
        video_grid_thw=None,
        inputs_embeds=inputs_embeds,
        attention_mask=dummy_attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    return (inputs_embeds, dummy_attention_mask, position_ids)


def export_prefill(
    model: nn.Module,
    causal_lm: nn.Module,
    dummy_inputs,
    export_path: Path,
    target_seq_len: int = 256,
) -> tuple:
    """Export prefill subgraph only.

    Returns (multimodal_embeddings, multimodal_attention_mask,
             multimodal_position_ids, orig_seq_len) for serial reuse by export_decode.
    """
    language_config = causal_lm.config
    multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = vit2llm(
        model, dummy_inputs
    )

    orig_seq_len = multimodal_embeddings.shape[1]
    assert orig_seq_len <= target_seq_len, (
        f"Input seq_len {orig_seq_len} exceeds target_seq_len {target_seq_len}"
    )
    padding_len = target_seq_len - orig_seq_len
    if padding_len > 0:
        pad_mask = torch.zeros((1, padding_len), dtype=multimodal_attention_mask.dtype)
        multimodal_attention_mask = torch.cat([multimodal_attention_mask, pad_mask], dim=1)

        pad_pos = torch.full(
            (3, 1, padding_len), orig_seq_len - 1, dtype=multimodal_position_ids.dtype
        )
        multimodal_position_ids = torch.cat([multimodal_position_ids, pad_pos], dim=2)

        pad_emb = torch.zeros(
            (1, padding_len, multimodal_embeddings.shape[2]),
            dtype=multimodal_embeddings.dtype,
        )
        multimodal_embeddings = torch.cat([multimodal_embeddings, pad_emb], dim=1)

    layer_types = language_config.layer_types
    prefill_input_names = [
        "multimodal_attention_mask",
        "position_ids",
        "multimodal_embeddings",
    ]
    prefill_output_names = ["logits"]
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            prefill_output_names += [f"present_{i}_key", f"present_{i}_value"]
        else:
            prefill_output_names += [
                f"present_state_{i}_conv",
                f"present_state_{i}_rec",
            ]
    prefill_wrapper = Qwen35PrefillWrapper(causal_lm)
    prefill_wrapper.eval()

    onnx_export_all_tensors_to_one_file(
        prefill_wrapper,
        (multimodal_attention_mask, multimodal_position_ids, multimodal_embeddings),
        onnx_path=export_path / "decoder_model_prefill.onnx",
        input_names=prefill_input_names,
        output_names=prefill_output_names,
    )
    print("✅ Decoder(prefill) 层导出完成")

    return (
        multimodal_embeddings,
        multimodal_attention_mask,
        multimodal_position_ids,
        orig_seq_len,
    )


def export_decode(
    model: nn.Module,
    causal_lm: nn.Module,
    dummy_inputs,
    export_path: Path,
    target_seq_len: int = 256,
    precomputed: tuple | None = None,
):
    """Export decode subgraph only.

    Args:
        precomputed: Optional (multimodal_embeddings, multimodal_attention_mask,
                      multimodal_position_ids, orig_seq_len) from export_prefill
                      to reuse vit2llm outputs.
    """
    language_config = causal_lm.config
    layer_types = language_config.layer_types

    if precomputed is not None:
        (
            multimodal_embeddings,
            multimodal_attention_mask,
            multimodal_position_ids,
            orig_seq_len,
        ) = precomputed
    else:
        multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = vit2llm(
            model, dummy_inputs
        )
        orig_seq_len = multimodal_embeddings.shape[1]
        padding_len = target_seq_len - orig_seq_len
        if padding_len > 0:
            pad_mask = torch.zeros((1, padding_len), dtype=multimodal_attention_mask.dtype)
            multimodal_attention_mask = torch.cat([multimodal_attention_mask, pad_mask], dim=1)
            pad_emb = torch.zeros(
                (1, padding_len, multimodal_embeddings.shape[2]),
                dtype=multimodal_embeddings.dtype,
            )
            multimodal_embeddings = torch.cat([multimodal_embeddings, pad_emb], dim=1)

    hidden_size = language_config.hidden_size
    num_attention_heads = language_config.num_attention_heads
    num_key_value_heads = getattr(language_config, "num_key_value_heads", num_attention_heads)
    head_dim = hidden_size // num_attention_heads
    vocab_size = language_config.vocab_size
    decoder_wrapper = Qwen35DecoderWrapper(causal_lm)
    dummy_batch_size = 1
    dummy_cache_len = target_seq_len

    dummy_decoder_input_ids = torch.randint(0, vocab_size, (dummy_batch_size, 1), dtype=torch.int64)
    dummy_decoder_inputs_embeds = causal_lm.get_input_embeddings()(dummy_decoder_input_ids).to(
        dtype=multimodal_embeddings.dtype
    )
    new_mask_segment = torch.tensor([[1]], dtype=torch.int64)
    dummy_attention_mask_decoder = torch.cat((multimodal_attention_mask, new_mask_segment), dim=1)
    new_pos_id = torch.tensor([[orig_seq_len]], dtype=multimodal_position_ids.dtype)

    num_kv_heads_linear = getattr(language_config, "linear_num_key_heads", 16)
    linear_key_head_dim = getattr(language_config, "linear_key_head_dim", 128)
    linear_value_head_dim = getattr(language_config, "linear_value_head_dim", 128)
    conv_dim = 3 * hidden_size
    conv_kernel = getattr(language_config, "linear_conv_kernel_dim", 4)

    dummy_all_flat = []
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            dummy_all_flat.append(
                torch.randn(
                    dummy_batch_size,
                    num_key_value_heads,
                    dummy_cache_len,
                    head_dim,
                    dtype=torch.float16,
                )
            )
            dummy_all_flat.append(
                torch.randn(
                    dummy_batch_size,
                    num_key_value_heads,
                    dummy_cache_len,
                    head_dim,
                    dtype=torch.float16,
                )
            )
        else:
            dummy_all_flat.append(torch.randn(1, conv_dim, conv_kernel, dtype=torch.float16))
            dummy_all_flat.append(
                torch.randn(
                    1,
                    num_kv_heads_linear,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    dtype=torch.float32,
                )
            )

    decoder_input_names = ["inputs_embeds", "multimodal_attention_mask", "position_ids"]
    decoder_output_names = ["logits"]
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            decoder_input_names += [f"past_{i}_key", f"past_{i}_value"]
            decoder_output_names += [f"present_{i}_key", f"present_{i}_value"]
        else:
            decoder_input_names += [f"past_state_{i}_conv", f"past_state_{i}_rec"]
            decoder_output_names += [
                f"present_state_{i}_conv",
                f"present_state_{i}_rec",
            ]

    onnx_export_all_tensors_to_one_file(
        decoder_wrapper,
        (
            dummy_decoder_inputs_embeds,
            dummy_attention_mask_decoder,
            new_pos_id,
            *dummy_all_flat,
        ),
        onnx_path=export_path / "decoder_model_decode.onnx",
        input_names=decoder_input_names,
        output_names=decoder_output_names,
    )
    print("✅ Decoder(decode) 层导出完成")


IMAGE_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "224x224.png"


def _load_model_and_inputs(qwen_path: str, img_path: str, text: str, device: str = "cpu"):
    """Load model and prepare inputs. Returns (model, causal_lm, torch_input)."""
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen_path,
        local_files_only=True,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = full_model.model

    text_config = full_model.config.text_config
    causal_lm = Qwen3_5ForCausalLM(text_config)
    causal_lm.model = model.language_model
    # torch.onnx.export 通过 tracing 来导出模型，它只会保存实际被 forward 函数使用到的权重和操作。
    # 因此，不必要替换embed_tokens 为 nn.Identity()。
    # causal_lm.model.embed_tokens = nn.Identity()
    causal_lm.lm_head = full_model.lm_head

    for decoder_layer in causal_lm.model.layers:
        if decoder_layer.layer_type == "linear_attention":
            decoder_layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            decoder_layer.linear_attn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            decoder_layer.linear_attn.forward = types.MethodType(
                gated_delta_net_forward_onnx, decoder_layer.linear_attn
            )

    causal_lm.eval()
    torch_input = get_model_input(qwen_path, [img_path], text, 1, device)
    return model, causal_lm, torch_input


def main(
    qwen_path: str,
    export_path: str,
    img_path: str,
    text: str,
    context_length: int = 256,
):
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    model, causal_lm, torch_input = _load_model_and_inputs(qwen_path, img_path, text)

    print(f"Exporting to {export_dir} ...")
    export_vit(model, torch_input, export_dir)
    precomputed = export_prefill(model, causal_lm, torch_input, export_dir, context_length)
    export_decode(model, causal_lm, torch_input, export_dir, context_length, precomputed)


if __name__ == "__main__":
    pass

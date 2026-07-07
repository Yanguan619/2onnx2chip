# Memory Optimization Notes

## Overview

对 `val_qwen3p5_om.py` 进行了 7 项内存优化，其中一项触发精度回退。

## 优化清单

| # | 优化 | 方法 | 状态 |
| --- | ------ | ------ | ------ |
| 2 | Decode mask 预分配 | `np.concatenate` → 固定数组 + 长度计数器 | **已修正精度 Bug** |
| 3 | Prefill logits 即时释放 | `del logits_tensor; del logits` | ✅ |
| 5 | 中间数组及时释放 | `del inp_embeds` (device copy 后) | ✅ |
| 6 | 跳过无用 contiguous copy | `reshape_outputs` 仅非 contiguous 时调用 | ✅ |
| 7 | Prefill 输入单次分配 | 3 次 `np.concatenate` → 预分配 + 直接填充 | ✅ |
| - | KV cache trim 后滑动 mask | `decode_mask[:, :256] = decode_mask[:, shift:256+shift]` | ✅ |
| - | 阶梯释放 host/npu tensor | `del prefill_outs_raw, token_embeds, pos, emb, mask, px, input_ids, ...` | ✅ |

## 精度 Bug: Decode Mask Trim 不对齐

### 根因

Decode 过程中 KV cache 长度超过 `TARGET_SEQ_LEN=256` 时触发 trim：截去前 `shift = past_len - 256` 个 token 的 KV 状态。**attention mask 必须同步截去前 `shift` 个位置**，否则 padding 0 与 KV cache 内容错位。

原代码（正确）：

```python
decode_mask_cum = decode_mask_cum[:, -TARGET_SEQ_LEN:].copy()
```

# 257 行从累积 mask 截取**最后** 256 个位置，与 KV cache 对齐

新代码（错误）：

```python
decode_mask_len = TARGET_SEQ_LEN  # 仅重置计数器
```

只重置了长度，未滑动 mask 数据窗口。低位数组内容仍是原始的 prefill mask（含 padding 0），与已前移的 KV cache 不匹配。

### 症状

模型生成的 attention mask 在 padding 位置错误地 mask 掉真实 token，导致上下文丢失。输出表现为：

- **生成重复短语**
- **内容变短、空洞**
- 同一 prompt 产生不同（且更差）的结果

### 修复

```python
shift = past_len - TARGET_SEQ_LEN
decode_mask[:, :TARGET_SEQ_LEN] = decode_mask[:, shift:past_len]
decode_mask_len = TARGET_SEQ_LEN
```

滑动 mask 窗口与 KV cache 同步。

## 教训

1. **View vs Copy 语义要保持**：原 `decode_mask_cum[:, -N:].copy()` 是独立的数组切片；预分配方案用长度计数器替代数组复制，必须保证数据内容同步。
2. **`del` 引用不影响精度**：只要在数组最后一次使用之后调用 `del`，对计算结果无影响。本次精度问题与 `del` 无关。
3. **可预分配不一定优于拼接**：`np.concatenate` 虽然每次分配新数组，语义直白不易出错。预分配 + 长度计数器提升性能但引入数据一致性成本。
4. **NPU 侧优化 ≠ host 侧优化**：Host 侧 numpy 操作不影响 NPU 计算精度，但涉及 attention mask、position ids 等语义数据的操作必须保持与原逻辑严格等价。

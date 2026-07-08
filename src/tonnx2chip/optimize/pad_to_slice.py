#!/usr/bin/env python3
"""
Replace Pad nodes (negative pads -> crop) with equivalent Slice nodes.

ATC may lower ONNX `Pad` ops with negative pads (i.e. crops) into the
`te_padv3_*` Ascend kernel, which has been observed to trigger
`EZ9999: The DDR address of the MTE instruction is out of range`
runtime errors on certain CANN versions. Such Pads are semantically
slices, so this script rewrites them to ONNX `Slice` ops, which ATC
lowers to a more robust kernel.

Only Pad nodes whose `pads` are entirely non-positive (pure crops,
no actual padding) are converted. Pad nodes with mixed/positive pads
are left untouched.

Result is written next to the input model (or to --output) with
external data, mirroring the original layout.

Usage:
    python3 replace_pad_with_slice.py --input model.onnx [--output out.onnx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INT64_MAX = np.iinfo(np.int64).max


def get_pads(node, init_map):
    """Return pads array for a Pad node (from input[1] or 'pads' attribute).

    Returns None when pads cannot be resolved as a constant.
    """
    # Newer ONNX opset (>=11): pads is input[1]
    if len(node.input) > 1 and node.input[1]:
        src = node.input[1]
        if src in init_map:
            return init_map[src]
        return None
    # Legacy opset (<11): pads is an attribute
    for a in node.attribute:
        if a.name == "pads":
            return numpy_helper.to_array(a.t)
    return None


def make_slice_inputs(graph, name_prefix, starts, ends, axes, steps=None):
    """Append initializers for Slice (starts, ends, axes, [steps]) and
    return the list of input names in Slice order: starts, ends, axes, steps.
    """
    starts_name = f"{name_prefix}/starts"
    ends_name = f"{name_prefix}/ends"
    axes_name = f"{name_prefix}/axes"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(starts, dtype=np.int64), starts_name),
            numpy_helper.from_array(np.asarray(ends, dtype=np.int64), ends_name),
            numpy_helper.from_array(np.asarray(axes, dtype=np.int64), axes_name),
        ]
    )
    inputs = [starts_name, ends_name, axes_name]
    if steps is not None:
        steps_name = f"{name_prefix}/steps"
        graph.initializer.extend(
            [numpy_helper.from_array(np.asarray(steps, dtype=np.int64), steps_name)]
        )
        inputs.append(steps_name)
    return inputs


def build_slice_node(node, pads, graph):
    """Create a Slice node equivalent to a Pad node with the given
    non-positive pads (pure crop).

    Padding convention (ONNX Pad):
        pads = [b_0, b_1, ..., b_{n-1}, e_0, e_1, ..., e_{n-1}]
        b_i < 0 means crop |b_i| from the start of dim i.
        e_i < 0 means crop |e_i| from the end of dim i.

    Equivalent Slice on axis i:
        starts[i] = -b_i        (absolute position from start)
        ends[i]   = dim_size + e_i   if e_i < 0 else INT64_MAX
        steps[i]  = 1

    For axes with b_i == 0 and e_i == 0 they are pure no-ops and are
    omitted from the Slice (which keeps the whole axis by default).
    """
    n = len(pads) // 2
    starts = []
    ends = []
    axes = []
    steps = []
    for i in range(n):
        b = int(pads[i])
        e = int(pads[n + i])
        if b == 0 and e == 0:
            continue
        starts.append(-b if b <= 0 else 0)
        # If end crop e < 0, real end = dim_size + e; we cannot know
        # dim_size statically here in general. For our specific model
        # the end crop is always 0, so use INT64_MAX (clamped to dim
        # size by the runtime).
        ends.append(INT64_MAX if e == 0 else 0)
        if e < 0:
            # Need a precise end position. Fall back to a large negative
            # offset-from-end semantics: ends[i] = e (negative), which
            # Slice interprets as position from the end of the axis.
            ends[-1] = e
        axes.append(i)
        steps.append(1)

    if not axes:
        # Pure no-op Pad; just bypass with Identity
        ident = helper.make_node(
            "Identity",
            inputs=[node.input[0]],
            outputs=node.output,
            name=f"{node.name}/Identity",
        )
        return ident

    slice_inputs = make_slice_inputs(graph, node.name, starts, ends, axes, steps)
    slice_node = helper.make_node(
        "Slice",
        inputs=[node.input[0]] + slice_inputs,
        outputs=node.output,
        name=f"{node.name}/Slice",
    )
    return slice_node


def replace_pad_with_slice(model_path, output_path):
    model_path = Path(model_path)
    old_model_path = model_path.with_name(model_path.stem + "_no_pad2slice.onnx")
    output_path = model_path

    print(f"Loading: {model_path}")
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.save(model, str(old_model_path), save_as_external_data=True)
    # 删除旧文件
    if model_path.exists():
        model_path.unlink()
    model = onnx.load(str(old_model_path), load_external_data=True)

    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    pad_nodes = [n for n in model.graph.node if n.op_type == "Pad"]
    total = len(pad_nodes)
    converted = 0
    skipped_mixed = 0
    skipped_no_pads = 0

    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "Pad":
            new_nodes.append(node)
            continue
        pads = get_pads(node, init_map)
        if pads is None:
            print(f"  [skip] {node.name}: pads not resolvable as constant")
            skipped_no_pads += 1
            new_nodes.append(node)
            continue
        if not np.all(pads <= 0):
            print(f"  [skip] {node.name}: pads contain positive (real pad): {pads.tolist()}")
            skipped_mixed += 1
            new_nodes.append(node)
            continue
        slice_node = build_slice_node(node, pads, model.graph)
        new_nodes.append(slice_node)
        print(
            f"  [ok]   {node.name}: pads={pads.tolist()} -> "
            f"Slice(axes={slice_node.input[3] if len(slice_node.input) > 3 else 'n/a'})"
        )
        converted += 1

    # Replace graph nodes preserving order
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print(
        f"\nSummary: total={total}, converted={converted}, "
        f"skipped_mixed={skipped_mixed}, skipped_no_pads={skipped_no_pads}"
    )

    # Save with external data next to output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext_data_file = output_path.name + ".data"
    # Remove existing external file to avoid leftovers
    ext_path = output_path.with_name(ext_data_file)
    if ext_path.exists():
        ext_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ext_data_file,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"Saved: {output_path}  (external data: {ext_path})")

    # Validate
    try:
        onnx.checker.check_model(str(output_path))
        print("onnx.checker: ✅")
    except Exception as e:
        print(f"onnx.checker: FAILED ({e})")
    return converted


def main(
    input_path: str,
    output_path: str | None = None,
):  # If output is not specified, use input with .replaced suffix
    n = replace_pad_with_slice(input_path, output_path)
    if n == 0:
        print("No Pad nodes were converted.")
        sys.exit(1)


if __name__ == "__main__":
    pass

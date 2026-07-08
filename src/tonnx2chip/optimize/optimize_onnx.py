#!/usr/bin/env python3
"""Prepare ONNX optimization candidates for later ATC selection."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import auto_optimizer  # noqa: F401  # pyright: ignore[reportMissingImports]
import onnx


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def external_data_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".data")


def remove_path_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_onnx_model(model_path: Path):
    return onnx.load(str(model_path), load_external_data=True)


def load_onnx_graph_model(model_path: Path):
    return onnx.load(str(model_path), load_external_data=False)


def tensor_uses_external_data(tensor: Any) -> bool:
    if tensor is None:
        return False
    if getattr(tensor, "data_location", None) != onnx.TensorProto.EXTERNAL:
        return False
    return any(
        entry.key == "location" and entry.value for entry in getattr(tensor, "external_data", [])
    )


def graph_uses_external_data(graph: Any) -> bool:
    for initializer in graph.initializer:
        if tensor_uses_external_data(initializer):
            return True
    for sparse_initializer in getattr(graph, "sparse_initializer", []):
        if tensor_uses_external_data(getattr(sparse_initializer, "values", None)):
            return True
        if tensor_uses_external_data(getattr(sparse_initializer, "indices", None)):
            return True
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and tensor_uses_external_data(
                attribute.t
            ):
                return True
            if attribute.type == onnx.AttributeProto.TENSORS:
                if any(tensor_uses_external_data(tensor) for tensor in attribute.tensors):
                    return True
            if attribute.type == onnx.AttributeProto.GRAPH and graph_uses_external_data(
                attribute.g
            ):
                return True
            if attribute.type == onnx.AttributeProto.GRAPHS and any(
                graph_uses_external_data(subgraph) for subgraph in attribute.graphs
            ):
                return True
            if attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                sparse_tensor = attribute.sparse_tensor
                if tensor_uses_external_data(getattr(sparse_tensor, "values", None)):
                    return True
                if tensor_uses_external_data(getattr(sparse_tensor, "indices", None)):
                    return True
    return False


def model_uses_external_data(model: Any) -> bool:
    return graph_uses_external_data(model.graph)


def save_onnx_model(model, model_path: Path) -> None:
    data_path = external_data_path(model_path)
    remove_path_if_exists(data_path)
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=model_path.name + ".data",
        size_threshold=1024,
    )


def clone_onnx_model(source_path: Path, target_path: Path) -> None:
    import shutil

    if source_path == target_path:
        return

    model = load_onnx_graph_model(source_path)
    referenced_locations: set[str] = set()
    target_data_path = external_data_path(target_path)

    def add_tensor_locations(tensor: Any) -> None:
        if tensor is None:
            return
        if getattr(tensor, "data_location", None) != onnx.TensorProto.EXTERNAL:
            return
        for entry in getattr(tensor, "external_data", []):
            if entry.key == "location" and entry.value:
                referenced_locations.add(entry.value)

    def visit_graph(graph: Any) -> None:
        for initializer in graph.initializer:
            add_tensor_locations(initializer)
        for sparse_initializer in getattr(graph, "sparse_initializer", []):
            add_tensor_locations(getattr(sparse_initializer, "values", None))
            add_tensor_locations(getattr(sparse_initializer, "indices", None))
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    add_tensor_locations(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        add_tensor_locations(tensor)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        visit_graph(subgraph)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    sparse_tensor = attribute.sparse_tensor
                    add_tensor_locations(getattr(sparse_tensor, "values", None))
                    add_tensor_locations(getattr(sparse_tensor, "indices", None))

    visit_graph(model.graph)

    ensure_parent(target_path)
    remove_path_if_exists(target_path)
    remove_path_if_exists(target_data_path)
    shutil.copy2(source_path, target_path)

    for location in referenced_locations:
        source_data = source_path.parent / location
        target_data = target_path.parent / location
        if not source_data.exists():
            continue
        if source_data == target_data:
            return
        if source_data.resolve() == target_data.resolve():
            continue
        if target_data.exists():
            target_data.unlink()
        try:
            os.link(source_data, target_data)
        except OSError:
            shutil.copy2(source_data, target_data)


def infer_shapes_inplace(model_path: Path) -> dict[str, Any]:
    graph_model = load_onnx_graph_model(model_path)
    has_external_data = model_uses_external_data(graph_model)
    file_size = model_path.stat().st_size
    prefer_path = has_external_data or file_size >= 1024 * 1024 * 1024

    attempts = ["path", "memory"] if prefer_path else ["memory", "path"]
    last_exc: Exception | None = None
    method = ""
    for attempt in attempts:
        try:
            if attempt == "memory":
                model = onnx.load(str(model_path), load_external_data=True)
                inferred = onnx.shape_inference.infer_shapes(
                    model,
                    strict_mode=False,
                    data_prop=True,
                )
                save_onnx_model(inferred, model_path)
            else:
                with tempfile.TemporaryDirectory(dir=str(model_path.parent)) as tmp_dir:
                    inferred_path = Path(tmp_dir) / model_path.name
                    onnx.shape_inference.infer_shapes_path(
                        str(model_path),
                        str(inferred_path),
                        strict_mode=False,
                        data_prop=True,
                    )
                    if not inferred_path.exists():
                        raise RuntimeError("Path-based shape inference did not produce a model")
                    inferred_external = inferred_path.with_name(inferred_path.name + ".data")
                    target_external = external_data_path(model_path)
                    inferred_graph_model = load_onnx_graph_model(inferred_path)
                    inferred_uses_external_data = model_uses_external_data(inferred_graph_model)
                    inferred_path.replace(model_path)
                    if inferred_external.exists():
                        remove_path_if_exists(target_external)
                        inferred_external.replace(target_external)
                    elif not inferred_uses_external_data:
                        remove_path_if_exists(target_external)
            method = attempt
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    else:
        raise RuntimeError("Shape inference failed in both memory and path modes") from last_exc

    inferred = load_onnx_graph_model(model_path)
    return {
        "value_info_count": len(inferred.graph.value_info),
        "node_count": len(inferred.graph.node),
        "method": method,
        "has_external_data": has_external_data,
        "file_size_bytes": file_size,
        "prefer_path": prefer_path,
    }


def safe_infer_shapes_inplace(model_path: Path) -> dict[str, Any]:
    try:
        result = infer_shapes_inplace(model_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
        }
    result["success"] = True
    return result


def pick_fields(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}


def compact_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = pick_fields(
        record,
        (
            "attempted",
            "success",
            "path",
            "reason",
            "error",
            "known_noop",
            "optimization_effect",
            "slim_strategy",
            "returncode",
            "validation_policy",
        ),
    )

    nested_field_specs = {
        "validation": ("success", "onnx_checker", "onnxruntime_load"),
        "runtime_validation_before_repair": ("onnx_checker", "onnxruntime_load"),
        "shape_inference": (
            "success",
            "method",
            "has_external_data",
            "prefer_path",
            "file_size_bytes",
            "value_info_count",
            "node_count",
        ),
        "shape_inference_after_repair": (
            "success",
            "method",
            "has_external_data",
            "prefer_path",
            "file_size_bytes",
            "value_info_count",
            "node_count",
        ),
        "final_validation": ("onnx_checker", "onnxruntime_load"),
        "shape_index_int_mismatch_repair": ("attempted", "success", "reason"),
    }
    for field, keys in nested_field_specs.items():
        nested = record.get(field)
        if isinstance(nested, dict):
            summary[field] = pick_fields(nested, keys)
            if field == "shape_index_int_mismatch_repair" and "patched_nodes" in nested:
                summary[field]["patched_nodes_count"] = len(nested["patched_nodes"])

    orphan_repair = record.get("orphan_output_repair")
    if isinstance(orphan_repair, dict):
        summary["orphan_output_repair"] = {
            "attempted": orphan_repair.get("attempted", False),
            "orphan_outputs_count": len(orphan_repair.get("orphan_outputs", [])),
            "repaired_count": len(orphan_repair.get("repaired", [])),
            "unresolved_count": len(orphan_repair.get("unresolved", [])),
        }

    remaining_orphans = record.get("remaining_orphan_outputs")
    if isinstance(remaining_orphans, list):
        summary["remaining_orphan_outputs_count"] = len(remaining_orphans)
    return summary


def validate_onnx_model(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    import onnxruntime as ort

    model_path = str(Path(model_path).resolve())
    result: dict[str, Any] = {
        "path": model_path,
        "exists": Path(model_path).exists(),
        "onnx_checker": False,
        "onnxruntime_load": False,
        "input_names": [],
        "output_names": [],
    }
    if not result["exists"]:
        result["error"] = "file_missing"
        return result

    onnx.checker.check_model(model_path)
    result["onnx_checker"] = True

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    result["onnxruntime_load"] = True
    result["input_names"] = [node.name for node in session.get_inputs()]
    result["output_names"] = [node.name for node in session.get_outputs()]
    return result


def inspect_candidate(
    model_path: Path,
    *,
    include_onnxruntime: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(model_path.resolve()),
        "exists": model_path.exists(),
        "onnx_checker": False,
        "onnxruntime_load": None,
        "input_names": [],
        "output_names": [],
    }
    if not result["exists"]:
        result["error"] = "file_missing"
        return result

    if include_onnxruntime:
        try:
            return validate_onnx_model(model_path)
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            return result

    try:
        model = onnx.load(str(model_path), load_external_data=False)
        initializer_names = {init.name for init in model.graph.initializer}
        onnx.checker.check_model(str(model_path))
        result["onnx_checker"] = True
        result["input_names"] = [
            value.name for value in model.graph.input if value.name not in initializer_names
        ]
        result["output_names"] = [value.name for value in model.graph.output]
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def _dim_to_repr(dim) -> int | str:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return "?"


def parse_dim_list(raw_shape: str) -> list[int]:
    dims = []
    for raw_dim in raw_shape.split(","):
        raw_dim = raw_dim.strip()
        if not raw_dim:
            raise ValueError(f"Invalid empty dimension in shape: {raw_shape}")
        dim = int(raw_dim)
        if dim <= 0:
            raise ValueError(f"Probe dimensions must be positive: {raw_shape}")
        dims.append(dim)
    return dims


def parse_shape_profile(raw_profile: str) -> dict[str, list[int]]:
    profile: dict[str, list[int]] = {}
    for item in raw_profile.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid profile item {item!r}; expected name:dim0,dim1,...")
        name, raw_shape = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid empty input name in profile: {raw_profile}")
        if name in profile:
            raise ValueError(f"Duplicate input name {name!r} in profile")
        profile[name] = parse_dim_list(raw_shape)
    if not profile:
        raise ValueError("Shape profile is empty")
    return profile


def parse_shape_profiles(raw_profiles: list[str]) -> list[dict[str, list[int]]]:
    return [parse_shape_profile(raw_profile) for raw_profile in raw_profiles]


def inspect_io_contract(model_path: Path) -> dict[str, Any]:
    model = load_onnx_graph_model(model_path)
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = []
    outputs = []

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        shape = [_dim_to_repr(dim) for dim in tensor_type.shape.dim]
        inputs.append(
            {
                "name": value_info.name,
                "shape": shape,
                "is_dynamic": any(not isinstance(dim, int) for dim in shape),
            }
        )

    for value_info in model.graph.output:
        tensor_type = value_info.type.tensor_type
        shape = [_dim_to_repr(dim) for dim in tensor_type.shape.dim]
        outputs.append(
            {
                "name": value_info.name,
                "shape": shape,
                "is_dynamic": any(not isinstance(dim, int) for dim in shape),
            }
        )

    return {
        "inputs": inputs,
        "outputs": outputs,
        "all_inputs_static": all(not item["is_dynamic"] for item in inputs),
        "has_dynamic_output": any(item["is_dynamic"] for item in outputs),
    }


def input_value_infos(model) -> list[Any]:
    initializer_names = {init.name for init in model.graph.initializer}
    return [
        value_info for value_info in model.graph.input if value_info.name not in initializer_names
    ]


def validate_profile_names(
    profiles: list[dict[str, list[int]]],
    graph_inputs: list[Any],
) -> None:
    input_names = {value_info.name for value_info in graph_inputs}
    for index, profile in enumerate(profiles):
        unknown = sorted(set(profile) - input_names)
        if unknown:
            raise ValueError(f"Profile {index} contains unknown graph inputs: {unknown}")


def shape_from_profile_or_graph(
    value_info: Any,
    profile: dict[str, list[int]],
) -> list[int] | list[int | str]:
    graph_shape = [_dim_to_repr(dim) for dim in value_info.type.tensor_type.shape.dim]
    if value_info.name not in profile:
        if any(not isinstance(dim, int) for dim in graph_shape):
            raise ValueError(
                f"Cannot probe dynamic input {value_info.name} with shape "
                f"{graph_shape}; provide --probe-shape-profile"
            )
        return graph_shape

    shape = profile[value_info.name]
    if len(shape) != len(graph_shape):
        raise ValueError(
            f"Profile shape rank mismatch for {value_info.name}: got {shape}, "
            f"graph rank is {len(graph_shape)}"
        )
    for index, (graph_dim, profile_dim) in enumerate(zip(graph_shape, shape)):
        if isinstance(graph_dim, int) and graph_dim != profile_dim:
            raise ValueError(
                f"Profile shape mismatch for {value_info.name} dim {index}: "
                f"got {profile_dim}, graph requires {graph_dim}"
            )
    return shape


def build_probe_feeds(
    model: Any,
    profile: dict[str, list[int]],
) -> dict[str, Any]:
    import numpy as np

    tensor_type_to_numpy = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
        11: np.float64,
        12: np.uint32,
        13: np.uint64,
    }

    graph_inputs = input_value_infos(model)
    validate_profile_names([profile], graph_inputs)
    feeds = {}
    for value_info in graph_inputs:
        tensor_type = value_info.type.tensor_type
        shape = shape_from_profile_or_graph(value_info, profile)
        dtype = tensor_type_to_numpy.get(tensor_type.elem_type)
        if dtype is None:
            raise ValueError(
                f"Unsupported input elem_type {tensor_type.elem_type} for {value_info.name}"
            )
        feeds[value_info.name] = np.zeros(shape, dtype=dtype)
    return feeds


def output_shape_template_from_probe_runs(
    runs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if not runs:
        return {}

    by_name: dict[str, list[list[int]]] = {}
    for run in runs:
        for output in run["outputs"]:
            by_name.setdefault(output["name"], []).append(output["shape"])

    templates: dict[str, list[dict[str, Any]]] = {}
    for name, shapes in by_name.items():
        ranks = {len(shape) for shape in shapes}
        if len(ranks) != 1:
            raise ValueError(f"Output {name} rank varies across profiles: {shapes}")
        rank = ranks.pop()
        dims = []
        for axis in range(rank):
            values = [shape[axis] for shape in shapes]
            if len(set(values)) == 1:
                dims.append({"kind": "fixed", "value": int(values[0])})
            else:
                dims.append(
                    {
                        "kind": "symbolic",
                        "param": f"{name}_dim{axis}",
                        "values": [int(value) for value in values],
                    }
                )
        templates[name] = dims
    return templates


def profile_coverage_for_dynamic_inputs(
    inspection: dict[str, Any],
    profiles: list[dict[str, list[int]]],
) -> dict[str, Any]:
    if not profiles:
        return {
            "profile_count": 0,
            "uncovered_dynamic_axes": [],
            "covered_dynamic_axes": [],
        }

    profile_names = set(profiles[0])
    for index, profile in enumerate(profiles[1:], start=1):
        if set(profile) != profile_names:
            raise ValueError(f"Profile {index} inputs do not match profile 0 inputs")

    covered = []
    uncovered = []
    for item in inspection["inputs"]:
        name = item["name"]
        shape = item["shape"]
        if name not in profile_names:
            if item["is_dynamic"]:
                uncovered.append({"input": name, "axis": "*", "reason": "missing"})
            continue
        profile_shapes = [profile[name] for profile in profiles]
        ranks = {len(profile_shape) for profile_shape in profile_shapes}
        if len(ranks) != 1:
            raise ValueError(f"Profiles for {name} have varying ranks: {profile_shapes}")
        rank = ranks.pop()
        if rank != len(shape):
            raise ValueError(
                f"Profile rank mismatch for {name}: got {rank}, graph rank is {len(shape)}"
            )
        for axis, graph_dim in enumerate(shape):
            if isinstance(graph_dim, int):
                continue
            values = [profile_shape[axis] for profile_shape in profile_shapes]
            entry = {
                "input": name,
                "axis": axis,
                "graph_dim": graph_dim,
                "values": values,
            }
            if len(set(values)) > 1:
                covered.append(entry)
            else:
                uncovered.append(entry)

    return {
        "profile_count": len(profiles),
        "covered_dynamic_axes": covered,
        "uncovered_dynamic_axes": uncovered,
    }


def patch_decision(
    inspection: dict[str, Any],
    profiles: list[dict[str, list[int]]],
    output_templates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not output_templates:
        return {"patch": False, "reason": "no_output_templates"}
    if inspection["all_inputs_static"]:
        return {"patch": True, "reason": "all_inputs_static"}
    if len(profiles) < 2:
        return {
            "patch": False,
            "reason": "dynamic_inputs_need_at_least_two_profiles",
        }

    coverage = profile_coverage_for_dynamic_inputs(inspection, profiles)
    if coverage["uncovered_dynamic_axes"]:
        return {
            "patch": False,
            "reason": "profiles_do_not_cover_all_dynamic_input_axes",
            "coverage": coverage,
        }
    return {
        "patch": True,
        "reason": "profiles_cover_all_dynamic_input_axes",
        "coverage": coverage,
    }


def probe_output_shape_template(
    model_path: Path,
    profiles: list[dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    model = load_onnx_graph_model(model_path)
    profiles = profiles or [{}]
    validate_profile_names(profiles, input_value_infos(model))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    runs = []
    for profile_index, profile in enumerate(profiles):
        feeds = build_probe_feeds(model, profile)
        output_values = session.run(None, feeds)
        output_meta = session.get_outputs()
        outputs = []
        for meta, value in zip(output_meta, output_values):
            outputs.append(
                {
                    "name": meta.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
        runs.append(
            {
                "profile_index": profile_index,
                "profile": profile,
                "feeds": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in feeds.items()
                },
                "outputs": outputs,
            }
        )
    return {
        "runs": runs,
        "output_shape_template": output_shape_template_from_probe_runs(runs),
    }


def patch_graph_output_shapes_inplace(
    model_path: Path,
    output_templates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    model = load_onnx_model(model_path)
    patched = 0
    for output in model.graph.output:
        template = output_templates.get(output.name)
        if template is None:
            continue
        output.type.tensor_type.shape.ClearField("dim")
        for dim in template:
            new_dim = output.type.tensor_type.shape.dim.add()
            if dim["kind"] == "fixed":
                new_dim.dim_value = int(dim["value"])
            else:
                new_dim.dim_param = dim["param"]
        patched += 1
    save_onnx_model(model, model_path)
    return {
        "patched_outputs": patched,
        "requested_outputs": len(output_templates),
    }


def maybe_patch_dynamic_outputs(
    model_path: Path,
    profiles: list[dict[str, list[int]]],
) -> dict[str, Any]:
    inspection = inspect_io_contract(model_path)
    record: dict[str, Any] = {
        "attempted": False,
        "inspection": inspection,
        "profile_count": len(profiles),
    }
    if not inspection["has_dynamic_output"]:
        record["reason"] = "no_dynamic_graph_outputs"
        return record
    if not inspection["all_inputs_static"] and not profiles:
        record["reason"] = "dynamic_inputs_without_probe_profiles"
        return record

    record["attempted"] = True
    try:
        probe = probe_output_shape_template(model_path, profiles)
        record["success"] = True
        record["probe"] = probe
        decision = patch_decision(
            inspection,
            profiles,
            probe["output_shape_template"],
        )
        record["patch_decision"] = decision
        if decision["patch"]:
            patch_result = patch_graph_output_shapes_inplace(
                model_path,
                probe["output_shape_template"],
            )
            record["patch_result"] = patch_result
            record["post_patch_inspection"] = inspect_io_contract(model_path)
    except Exception as exc:  # noqa: BLE001
        record["success"] = False
        record["error"] = str(exc)
    return record


def produced_tensor_names(model) -> set[str]:
    initializer_names = {init.name for init in model.graph.initializer}
    graph_input_names = {
        value.name for value in model.graph.input if value.name not in initializer_names
    }
    node_output_names = {
        output_name for node in model.graph.node for output_name in node.output if output_name
    }
    return initializer_names | graph_input_names | node_output_names


def find_orphan_graph_outputs(model_path: Path) -> list[str]:
    model = load_onnx_graph_model(model_path)
    available_names = produced_tensor_names(model)
    return [value.name for value in model.graph.output if value.name not in available_names]


def identity_alias_map(model_path: Path) -> dict[str, str]:
    model = load_onnx_graph_model(model_path)
    aliases: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = node.input[0]
        output_name = node.output[0]
        if input_name and output_name:
            aliases[input_name] = output_name
            aliases[output_name] = input_name
    return aliases


def repair_orphan_graph_outputs(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    orphan_outputs = find_orphan_graph_outputs(candidate_path)
    record: dict[str, Any] = {
        "attempted": bool(orphan_outputs),
        "orphan_outputs": orphan_outputs,
        "repaired": [],
        "unresolved": [],
    }
    if not orphan_outputs:
        return record

    alias_map = identity_alias_map(reference_path)
    candidate_model = load_onnx_model(candidate_path)
    available_names = produced_tensor_names(candidate_model)
    inserted_nodes = []

    for output in candidate_model.graph.output:
        if output.name not in orphan_outputs:
            continue
        alias_name = alias_map.get(output.name)
        if alias_name and alias_name in available_names:
            node_name = f"restore_output_alias_{output.name}"
            candidate_model.graph.node.append(
                onnx.helper.make_node(
                    "Identity",
                    inputs=[alias_name],
                    outputs=[output.name],
                    name=node_name,
                )
            )
            inserted_nodes.append(node_name)
            record["repaired"].append(
                {
                    "output": output.name,
                    "source_tensor": alias_name,
                    "method": "identity_restore",
                }
            )
            available_names.add(output.name)
        else:
            record["unresolved"].append(output.name)

    if inserted_nodes:
        record["inserted_nodes"] = inserted_nodes
        save_onnx_model(candidate_model, candidate_path)
    return record


SIMPLICITY_NODE_THRESHOLD = 50


def model_is_simple(model_path: Path) -> tuple[bool, int]:
    model = load_onnx_graph_model(model_path)
    op_types = {node.op_type for node in model.graph.node}
    count = len(op_types)
    return count < SIMPLICITY_NODE_THRESHOLD, count


def should_save_as_external_data(model_path: Path) -> bool:
    if model_path.stat().st_size > 1024 * 1024 * 1024:
        return True
    try:
        model = load_onnx_graph_model(model_path)
    except Exception:  # noqa: BLE001
        return True
    return model_uses_external_data(model)


def run_onnxslim(input_path: Path, output_path: Path) -> dict[str, Any]:
    from onnxslim import slim

    def clear_output() -> None:
        data_path = external_data_path(output_path)
        if output_path.exists():
            output_path.unlink()
        if data_path.exists():
            data_path.unlink()

    save_external = should_save_as_external_data(input_path)
    slim_error: Exception | None = None
    slim_strategy = "shape_infer"
    for no_shape_infer in (False, True):
        clear_output()
        try:
            slim(
                str(input_path),
                output_model=str(output_path),
                save_as_external_data=save_external,
                no_shape_infer=no_shape_infer,
            )
            slim_strategy = "no_shape_infer" if no_shape_infer else "shape_infer"
            break
        except Exception as exc:  # noqa: BLE001
            slim_error = exc
            if no_shape_infer:
                raise
    if slim_error is not None and not output_path.exists():
        raise slim_error
    if not output_path.exists():
        raise RuntimeError("onnxslim did not produce the expected output model.")
    structural_validation = inspect_candidate(
        output_path,
        include_onnxruntime=False,
    )
    if not structural_validation.get("onnx_checker", False):
        raise RuntimeError(
            "onnxslim produced a model that failed ONNX checker: "
            f"{structural_validation.get('error', 'unknown_error')}"
        )
    result: dict[str, Any] = {
        "path": str(output_path.resolve()),
        "slim_strategy": slim_strategy,
        "shape_inference": safe_infer_shapes_inplace(output_path),
        "structural_validation": structural_validation,
    }
    result["orphan_output_repair"] = repair_orphan_graph_outputs(output_path, input_path)
    if result["orphan_output_repair"].get("attempted"):
        result["shape_inference_after_repair"] = safe_infer_shapes_inplace(output_path)
    result["remaining_orphan_outputs"] = find_orphan_graph_outputs(output_path)
    try:
        result["runtime_validation"] = validate_onnx_model(output_path)
    except Exception as exc:  # noqa: BLE001
        result["runtime_validation"] = {
            "path": str(output_path.resolve()),
            "exists": output_path.exists(),
            "onnx_checker": structural_validation.get("onnx_checker", False),
            "onnxruntime_load": False,
            "error": str(exc),
        }
    result["success"] = bool(
        result["runtime_validation"].get("onnxruntime_load")
        and not result["remaining_orphan_outputs"]
    )
    return result


def build_autoopt_command(
    input_path: Path,
    output_path: Path,
    knowledges: str | None = None,
) -> list[str] | None:
    cmd = [
        sys.executable,
        "-m",
        "auto_optimizer",
        "optimize",
        str(input_path),
        str(output_path),
    ]
    if knowledges:
        cmd.extend(["-k", knowledges])
    return cmd


def is_auto_optimizer_no_knowledge_message(messages: str) -> bool:
    return bool(
        re.search(
            r"(unable\s+to\s+optimi[sz]e,\s*)?"
            r"no\s+knowledge(?:s)?\s+(?:matched|can\s+be\s+matched|applicable)",
            messages,
            flags=re.IGNORECASE,
        )
    )


def load_type_maps(model) -> tuple[dict[str, int], dict[str, Any]]:
    value_types: dict[str, int] = {}
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        elem_type = value.type.tensor_type.elem_type
        if elem_type:
            value_types[value.name] = elem_type
    initializers = {init.name: init for init in model.graph.initializer}
    return value_types, initializers


def input_elem_type(
    name: str,
    value_types: dict[str, int],
    initializers: dict[str, Any],
) -> int | None:
    if name in value_types:
        return value_types[name]
    if name in initializers:
        return initializers[name].data_type
    return None


def find_or_create_int32_initializer(
    model,
    initializers: dict[str, Any],
    source_name: str,
) -> tuple[str, bool]:
    import numpy as np
    from onnx import TensorProto, numpy_helper

    source = initializers[source_name]
    source_array = numpy_helper.to_array(source)
    target_dtype = TensorProto.INT32
    target_np_dtype = np.int32

    for init in model.graph.initializer:
        candidate_array = numpy_helper.to_array(init)
        if (
            init.data_type == target_dtype
            and candidate_array.shape == source_array.shape
            and (candidate_array == source_array).all()
        ):
            return init.name, False

    if source_array.size and np.issubdtype(source_array.dtype, np.integer):
        int32_info = np.iinfo(np.int32)
        source_min = source_array.min()
        source_max = source_array.max()
        if source_min < int32_info.min or source_max > int32_info.max:
            raise ValueError(
                f"Initializer {source_name} cannot be narrowed to INT32 safely: "
                f"value range [{source_min}, {source_max}] exceeds int32"
            )

    base_name = f"{source_name}_INT32"
    clone_name = base_name
    suffix = 1
    while clone_name in initializers:
        clone_name = f"{base_name}_{suffix}"
        suffix += 1

    clone_tensor = numpy_helper.from_array(source_array.astype(target_np_dtype), name=clone_name)
    model.graph.initializer.append(clone_tensor)
    initializers[clone_name] = clone_tensor
    return clone_name, True


def find_or_create_int64_initializer(
    model,
    initializers: dict[str, Any],
    source_name: str,
) -> tuple[str, bool]:
    import numpy as np
    from onnx import TensorProto, numpy_helper

    source = initializers[source_name]
    source_array = numpy_helper.to_array(source)
    target_dtype = TensorProto.INT64
    target_np_dtype = np.int64

    for init in model.graph.initializer:
        candidate_array = numpy_helper.to_array(init)
        if (
            init.data_type == target_dtype
            and candidate_array.shape == source_array.shape
            and (candidate_array == source_array).all()
        ):
            return init.name, False

    base_name = f"{source_name}_INT64"
    clone_name = base_name
    suffix = 1
    while clone_name in initializers:
        clone_name = f"{base_name}_{suffix}"
        suffix += 1

    clone_tensor = numpy_helper.from_array(source_array.astype(target_np_dtype), name=clone_name)
    model.graph.initializer.append(clone_tensor)
    initializers[clone_name] = clone_tensor
    return clone_name, True


def find_or_create_typed_initializer(
    model,
    initializers: dict[str, Any],
    source_name: str,
    target_dtype: int,
) -> tuple[str, bool]:
    from onnx import TensorProto

    if target_dtype == TensorProto.INT32:
        return find_or_create_int32_initializer(model, initializers, source_name)
    if target_dtype == TensorProto.INT64:
        return find_or_create_int64_initializer(model, initializers, source_name)

    raise ValueError(
        f"Unsupported target dtype for shape/index repair: {TensorProto.DataType.Name(target_dtype)}"
    )


def choose_target_integer_type(
    node,
    input_names: list[str],
    input_types: list[int | None],
    initializers: dict[str, Any],
    value_types: dict[str, int],
) -> int | None:
    from onnx import TensorProto

    dynamic_input_types = {
        elem_type
        for name, elem_type in zip(input_names, input_types)
        if elem_type in {TensorProto.INT32, TensorProto.INT64} and name not in initializers
    }
    if len(dynamic_input_types) == 1:
        return next(iter(dynamic_input_types))
    if len(dynamic_input_types) > 1:
        return None

    output_types = {
        value_types[name]
        for name in node.output
        if name in value_types and value_types[name] in {TensorProto.INT32, TensorProto.INT64}
    }
    if len(output_types) == 1:
        return next(iter(output_types))
    if len(output_types) > 1:
        return None

    concrete_input_types = [
        elem_type
        for elem_type in input_types
        if elem_type in {TensorProto.INT32, TensorProto.INT64}
    ]
    if not concrete_input_types:
        return None

    int32_count = concrete_input_types.count(TensorProto.INT32)
    int64_count = concrete_input_types.count(TensorProto.INT64)
    if int32_count > int64_count:
        return TensorProto.INT32
    return TensorProto.INT64


def patch_mixed_int_concat_nodes(model) -> list[dict[str, Any]]:
    from onnx import TensorProto

    value_types, initializers = load_type_maps(model)
    patched_nodes: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Concat":
            continue

        input_types = [input_elem_type(name, value_types, initializers) for name in node.input]
        concrete_types = {elem_type for elem_type in input_types if elem_type is not None}
        if concrete_types != {TensorProto.INT32, TensorProto.INT64}:
            continue
        target_dtype = choose_target_integer_type(
            node,
            list(node.input),
            input_types,
            initializers,
            value_types,
        )
        if target_dtype is None:
            continue

        replacements = []
        for index, (name, elem_type) in enumerate(zip(node.input, input_types)):
            if elem_type == target_dtype or name not in initializers:
                continue
            replacement_name, reused_existing = find_or_create_typed_initializer(
                model,
                initializers,
                name,
                target_dtype,
            )
            node.input[index] = replacement_name
            replacements.append(
                {
                    "input_index": index,
                    "from": name,
                    "to": replacement_name,
                    "target_dtype": TensorProto.DataType.Name(target_dtype),
                    "reused_existing_initializer": reused_existing is False,
                }
            )

        if replacements:
            patched_nodes.append(
                {
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "replacements": replacements,
                }
            )

    return patched_nodes


def patch_mixed_int_slice_nodes(model) -> list[dict[str, Any]]:
    from onnx import TensorProto

    value_types, initializers = load_type_maps(model)
    patched_nodes: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue

        indexed_index_inputs = [
            (index, name) for index, name in enumerate(node.input[1:], start=1) if name
        ]
        index_inputs = [name for _, name in indexed_index_inputs]
        index_types = [input_elem_type(name, value_types, initializers) for name in index_inputs]
        concrete_types = {elem_type for elem_type in index_types if elem_type is not None}
        if concrete_types != {TensorProto.INT32, TensorProto.INT64}:
            continue
        target_dtype = choose_target_integer_type(
            node,
            index_inputs,
            index_types,
            initializers,
            value_types,
        )
        if target_dtype is None:
            continue

        replacements = []
        for (local_index, name), elem_type in zip(indexed_index_inputs, index_types):
            if elem_type == target_dtype or name not in initializers:
                continue
            replacement_name, reused_existing = find_or_create_typed_initializer(
                model,
                initializers,
                name,
                target_dtype,
            )
            node.input[local_index] = replacement_name
            replacements.append(
                {
                    "input_index": local_index,
                    "from": name,
                    "to": replacement_name,
                    "target_dtype": TensorProto.DataType.Name(target_dtype),
                    "reused_existing_initializer": reused_existing is False,
                }
            )

        if replacements:
            patched_nodes.append(
                {
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "replacements": replacements,
                }
            )

    return patched_nodes


def repair_shape_index_int_mismatch(
    source_path: Path,
    repaired_path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempted": True,
        "input_path": str(source_path.resolve()),
        "output_path": str(repaired_path.resolve()),
        "patched_nodes": [],
        "success": False,
    }
    model = load_onnx_model(source_path)
    patched_nodes = []
    patched_nodes.extend(patch_mixed_int_concat_nodes(model))
    patched_nodes.extend(patch_mixed_int_slice_nodes(model))
    record["patched_nodes"] = patched_nodes

    if not patched_nodes:
        record["reason"] = "no_supported_mismatch_patterns_found"
        return record

    save_onnx_model(model, repaired_path)
    record["shape_inference"] = safe_infer_shapes_inplace(repaired_path)
    record["inspection"] = inspect_candidate(
        repaired_path,
        include_onnxruntime=True,
    )
    record["success"] = bool(
        record["inspection"].get("onnx_checker") and record["inspection"].get("onnxruntime_load")
    )
    if not record["success"]:
        record["reason"] = "repair_output_failed_runtime_validation"
    return record


def run_auto_optimizer(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    knowledges: str | None = None,
) -> dict[str, Any]:
    command = build_autoopt_command(input_path, output_path, knowledges=knowledges)
    if command is None:
        return {
            "attempted": False,
            "success": False,
            "reason": "auto_optimizer_runner_unavailable",
        }

    external_data_path = output_path.with_name(output_path.name + ".data")
    repaired_output = output_path.with_name(f"{output_path.stem}.repaired{output_path.suffix}")
    repaired_external_data_path = external_data_path.with_name(f"{repaired_output.name}.data")
    if output_path.exists():
        output_path.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    if repaired_output.exists():
        repaired_output.unlink()
    if repaired_external_data_path.exists():
        repaired_external_data_path.unlink()

    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {' '.join(command)}",
                "",
                "[STDOUT]",
                result.stdout,
                "",
                "[STDERR]",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "attempted": True,
        "command": command,
        "returncode": result.returncode,
        "log_path": str(log_path),
        "path": str(output_path.resolve()),
    }
    optimizer_messages = "\n".join([result.stdout, result.stderr])
    no_knowledge_matched = is_auto_optimizer_no_knowledge_message(optimizer_messages)
    if no_knowledge_matched:
        payload["known_noop"] = True
        payload["optimization_effect"] = "no_op"
        payload["reason"] = "auto_optimizer_no_knowledge_matched"
    command_succeeded = result.returncode == 0 and output_path.exists() and not no_knowledge_matched
    if command_succeeded:
        payload["shape_inference"] = safe_infer_shapes_inplace(output_path)
        payload["structural_validation"] = inspect_candidate(output_path, include_onnxruntime=False)
        payload["orphan_output_repair"] = repair_orphan_graph_outputs(output_path, input_path)
        if payload["orphan_output_repair"].get("attempted"):
            payload["shape_inference_after_repair"] = safe_infer_shapes_inplace(output_path)
        payload["remaining_orphan_outputs"] = find_orphan_graph_outputs(output_path)
        payload["runtime_validation_before_repair"] = inspect_candidate(
            output_path,
            include_onnxruntime=True,
        )
        final_path = output_path
        final_validation = payload["runtime_validation_before_repair"]
        payload["shape_index_int_mismatch_repair"] = {
            "attempted": False,
        }
        if not final_validation.get("onnxruntime_load", False):
            repaired_output = output_path.with_name(
                f"{output_path.stem}.repaired{output_path.suffix}"
            )
            payload["shape_index_int_mismatch_repair"] = repair_shape_index_int_mismatch(
                output_path,
                repaired_output,
            )
            if payload["shape_index_int_mismatch_repair"].get("success"):
                final_path = repaired_output
                final_validation = payload["shape_index_int_mismatch_repair"]["inspection"]

        payload["path"] = str(final_path.resolve())
        payload["final_validation"] = final_validation
        payload["success"] = bool(
            payload["structural_validation"].get("onnx_checker", False)
            and final_validation.get("onnx_checker", False)
            and final_validation.get("onnxruntime_load", False)
            and not payload["remaining_orphan_outputs"]
        )
        if not payload["success"]:
            payload["reason"] = "post_optimization_validation_failed"
        else:
            payload["validation_policy"] = "checker_plus_onnxruntime_before_atc"
    else:
        payload["success"] = False
        if not no_knowledge_matched:
            payload["reason"] = "command_failed"
    return payload


def main(
    model: str,
    save_dir: str,
    probe_shape_profile: list[str] = [],
    skip_output_shape_probe=False,
    autoopt_knowledges="",
) -> None:
    model_path = Path(model).resolve()
    model_name = model_path.stem
    assert model_path.exists(), f"Model not found: {model_path}"

    # Stage: output layout
    if save_dir.strip() == "":
        save_dir = str(model_path.parent / model_name)
    output_dir = Path(save_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印输入参数
    print(f"Model: {model}")
    print(f"Save dir: {output_dir}")
    print(f"Probe shape profile: {probe_shape_profile}")
    print(f"Skip output shape probe: {skip_output_shape_probe}")

    probe_shape_profiles = parse_shape_profiles(probe_shape_profile)

    original_copy = model_path
    summary: dict[str, Any] = {
        "source_model": str(model_path),
        "pre_optimization_fixes": {},
        "candidates": {},
    }

    if skip_output_shape_probe:
        summary["pre_optimization_fixes"]["dynamic_output_shape_patch"] = {
            "attempted": False,
            "reason": "skip_output_shape_probe_requested",
            "profile_count": len(probe_shape_profiles),
        }
    else:
        patch_record = maybe_patch_dynamic_outputs(original_copy, probe_shape_profiles)
        summary["pre_optimization_fixes"]["dynamic_output_shape_patch"] = pick_fields(
            patch_record,
            ("attempted", "success", "reason", "profile_count"),
        )
    original_shape_info = safe_infer_shapes_inplace(original_copy)

    # Stage: original candidate
    original_validation = inspect_candidate(original_copy, include_onnxruntime=True)
    summary["candidates"]["original"] = compact_candidate_record(
        {
            "path": str(original_copy),
            "shape_inference": original_shape_info,
            "validation": original_validation,
            "success": bool(
                original_validation.get("onnx_checker")
                and original_validation.get("onnxruntime_load")
            ),
        }
    )

    selected_label = "original"
    selected_path = original_copy

    print("Stage: onnxslim candidate")
    onnxslim_output = output_dir / f"{model_name}_onnxslim.onnx"
    onnxslim_record: dict[str, Any] = {"attempted": True, "path": str(onnxslim_output)}
    try:
        onnxslim_record["validation"] = run_onnxslim(original_copy, onnxslim_output)
        onnxslim_record["success"] = bool(onnxslim_record["validation"].get("success"))
        if onnxslim_record["success"]:
            selected_label = "onnxslim"
            selected_path = onnxslim_output
        else:
            onnxslim_record["reason"] = "post_slim_validation_failed"
    except Exception as exc:  # noqa: BLE001
        onnxslim_record["success"] = False
        onnxslim_record["error"] = str(exc)
    summary["candidates"]["onnxslim"] = compact_candidate_record(onnxslim_record)

    print("Stage: auto-optimizer candidate")
    autoopt_output = output_dir / f"{model_name}_auto_optimizer.onnx"
    autoopt_log = output_dir / "auto_optimizer.log"
    autoopt_input = selected_path
    simple, node_count = model_is_simple(autoopt_input)
    user_specified_knowledges = bool(autoopt_knowledges)
    if simple and not user_specified_knowledges:
        print(
            f"Model has few op types ({node_count} unique types < {SIMPLICITY_NODE_THRESHOLD}), skipping auto_optimizer"
        )
        autoopt_record = {
            "attempted": False,
            "success": False,
            "reason": "model_too_simple",
            "unique_op_types": node_count,
            "threshold": SIMPLICITY_NODE_THRESHOLD,
        }
    else:
        knowledges = autoopt_knowledges or None
        if knowledges:
            print(f"Running auto_optimizer (knowledges={knowledges}), log: {autoopt_log}")
        else:
            print(f"Running auto_optimizer, log: {autoopt_log}")
        autoopt_record = run_auto_optimizer(
            autoopt_input, autoopt_output, autoopt_log, knowledges=knowledges
        )
        if autoopt_record.get("success"):
            selected_label = "auto_optimizer"
            selected_path = Path(autoopt_record["path"]).resolve()
    summary["candidates"]["auto_optimizer"] = compact_candidate_record(autoopt_record)

    print("Stage: optimization summary")
    summary["candidate_priority"] = ["auto_optimizer", "onnxslim", "original"]
    summary["selection_basis"] = "optimizer_preference_after_validation"
    summary["preferred_candidate"] = selected_label
    summary["preferred_model"] = str(selected_path.resolve())
    summary["preferred_inspection"] = inspect_candidate(
        selected_path,
        include_onnxruntime=True,
    )
    summary["preferred_inspection"] = pick_fields(
        summary["preferred_inspection"],
        ("path", "exists", "onnx_checker", "onnxruntime_load", "error"),
    )

    # Stage: final model alias
    final_path = output_dir / f"{model_name}.onnx"
    print(f"Stage: final model location {final_path}")
    clone_onnx_model(selected_path, final_path)
    summary["final_model"] = str(final_path.resolve())

    write_json(output_dir / "optimization_summary.json", summary)


if __name__ == "__main__":
    pass

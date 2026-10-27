"""Compiled artifact construction, validation and standalone native execution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .backend import Hardware, Program, operation_timings, simulate
from .codegen import annotations, blank, decode, encode, materialize
from .ir import Graph
from .memory import MAX_MEMORY, align, plan_memory, schedule_operations


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@dataclass
class CompiledImage:
    files: dict[str, bytes]
    manifest: dict
    memory: dict
    schedule: dict

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            (directory / name).write_bytes(data)


def build_image(
    graph: Graph,
    program: Program,
    *,
    strategy: str = "memory-aware",
    reuse: bool = True,
    dram_limit: int = MAX_MEMORY,
) -> CompiledImage:
    order = schedule_operations(graph, strategy)
    emitted = list(dict.fromkeys(i["node"] for i in program.instructions))
    if emitted != [o.id for o in order]:
        raise ValueError("lowering order disagrees with compiler schedule")
    memory = plan_memory(graph, order, reuse=reuse, dram_limit=dram_limit)
    records = materialize(program, memory)
    instruction_data = encode(records)
    # One native decoder defines the executable representation.
    canonical = decode(instruction_data)
    for original, loaded in zip(records, canonical, strict=True):
        if {key: original[key] for key in blank()} != {key: loaded[key] for key in blank()}:
            raise ValueError("instruction encoding changed an operand")
    weights = bytearray(memory["weights_bytes"])
    for t in memory["tensors"]:
        if "weight" in t["roles"]:
            raw = graph.tensors[t["id"]].data.astype("<f4", copy=False).tobytes()
            weights[t["address"] : t["address"] + t["size"]] = raw
    metrics = simulate(program)
    events = {e["id"]: e for e in metrics["trace"]}
    schedule = {
        "schema": "npuforge.schedule.v1",
        "strategy": strategy,
        "operator_order": [o.id for o in order],
        "instructions": [
            i | {"start": events[i["id"]]["start"], "end": events[i["id"]]["end"]} for i in records
        ],
    }
    timing = metrics["per_operation"]
    for t in memory["tensors"]:
        t["start_cycle"] = 0 if not order else timing[order[t["first"]].id]["start"]
        t["end_cycle"] = (
            metrics["cycles"] if t["last"] == len(order) else timing[order[t["last"]].id]["end"]
        )
    for arena in memory["scratchpad"]:
        arena["start_cycle"] = events[arena["first_instruction"]]["start"]
        arena["end_cycle"] = events[arena["last_instruction"]]["end"]
    memory["sram_bytes"] = program.hardware.sram_bytes
    memory["instruction_storage"] = {
        "space": "instruction",
        "address": 0,
        "size": len(instruction_data),
        "alignment": 4,
    }
    files = {
        "instructions.bin": instruction_data,
        "weights.bin": bytes(weights),
        "memory_map.json": json_bytes(memory),
        "schedule.json": json_bytes(schedule),
    }
    manifest = {
        "schema": "npuforge.program.v1",
        "format_version": 1,
        "precision": "float32",
        "byte_order": "little",
        "entry_point": 0,
        "model": graph.name,
        "architecture": program.hardware.architecture,
        "hardware": asdict(program.hardware),
        "inputs": graph.inputs,
        "outputs": graph.outputs,
        "instruction_count": len(records),
        "record_bytes": 256,
        "images": {name: digest(data) for name, data in files.items()},
    }
    files["manifest.json"] = json_bytes(manifest)
    return CompiledImage(files, manifest, memory, schedule)


def _integer(value, name: str, maximum: int = MAX_MEMORY) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name}: invalid bounded nonnegative integer")
    return value


def _read(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError(f"{path.name}: image exceeds size limit")
    return path.read_bytes()


@dataclass
class LoadedProgram:
    manifest: dict
    memory: dict
    schedule: dict
    instructions: bytes
    weights: bytes

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
        from . import _native

        if set(inputs) != set(self.manifest["inputs"]):
            raise ValueError("runtime input names must exactly match the manifest")
        raw = {}
        tensors = self.memory["tensors"]
        for index, t in enumerate(tensors):
            if t["id"] in inputs:
                value = inputs[t["id"]]
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float32
                    or value.shape != tuple(t["shape"])
                ):
                    raise ValueError(f"{t['id']}: expected FP32 shape {tuple(t['shape'])}")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{t['id']}: non-finite runtime input")
                raw[index] = value.astype("<f4", copy=False).tobytes(order="C")
        executed = _native.execute_image(
            self.instructions,
            self.weights,
            tensors,
            raw,
            self.memory["dram_bytes"],
            self.manifest["hardware"],
        )
        outputs = {
            tensors[index]["id"]: np.frombuffer(data, dtype="<f4")
            .copy()
            .reshape(tensors[index]["shape"])
            for index, data in executed["outputs"].items()
        }
        outputs = {name: outputs[name] for name in self.manifest["outputs"]}
        metrics = operation_timings(executed["metrics"], self.schedule["instructions"])
        planned = {i["id"]: (i["start"], i["end"]) for i in self.schedule["instructions"]}
        if any(planned[e["id"]] != (e["start"], e["end"]) for e in metrics["trace"]):
            raise ValueError("binary execution disagrees with the planned resource schedule")
        metrics.update(
            {
                "dram_allocation_bytes": self.memory["dram_bytes"],
                "reused_allocation_bytes": self.memory["reused_bytes"],
                "allocation_efficiency": self.memory["dram_bytes"]
                / max(1, self.memory["unreused_bytes"]),
                "instruction_count": self.manifest["instruction_count"],
                "instruction_image_bytes": len(self.instructions),
                "weight_image_bytes": len(self.weights),
            }
        )
        metrics["resource_idle_cycles"] = {
            name: metrics["cycles"] - metrics[name + "_busy"]
            for name in ("compute", "dma", "pim", "dram")
        }
        return outputs, metrics


def load_program(directory: str | Path) -> LoadedProgram:
    directory = Path(directory)
    try:
        manifest = json.loads(_read(directory / "manifest.json", 16 * 1024 * 1024))
        if (
            manifest["schema"] != "npuforge.program.v1"
            or manifest["format_version"] != 1
            or manifest["precision"] != "float32"
            or manifest["byte_order"] != "little"
            or manifest["entry_point"] != 0
            or manifest["record_bytes"] != 256
        ):
            raise ValueError("unsupported program schema, precision or entry point")
        h = Hardware(**manifest["hardware"])
        if manifest["architecture"] != h.architecture or h.sram_bytes > MAX_MEMORY:
            raise ValueError("invalid target architecture or SRAM capacity")
        required = {"instructions.bin", "weights.bin", "memory_map.json", "schedule.json"}
        if set(manifest["images"]) != required:
            raise ValueError("manifest must identify the four required artifact files")
        files = {}
        for name in sorted(required):
            data = _read(
                directory / name, 256 * 1024 * 1024 if name.endswith(".bin") else 32 * 1024 * 1024
            )
            if manifest["images"][name] != digest(data):
                raise ValueError(f"{name}: artifact length/checksum mismatch")
            files[name] = data
        memory = json.loads(files["memory_map.json"])
        schedule = json.loads(files["schedule.json"])
        if memory["schema"] != "npuforge.memory.v1" or schedule["schema"] != "npuforge.schedule.v1":
            raise ValueError("unsupported memory map or schedule schema")
        dram = _integer(memory["dram_bytes"], "dram_bytes")
        limit = _integer(memory["dram_limit"], "dram_limit")
        if dram > limit or memory["sram_bytes"] != h.sram_bytes:
            raise ValueError("memory map exceeds target capacities")
        if len(files["weights.bin"]) != _integer(memory["weights_bytes"], "weights_bytes"):
            raise ValueError("weight image length disagrees with memory map")
        tensors = memory["tensors"]
        if not isinstance(tensors, list) or len(tensors) > 16384:
            raise ValueError("invalid tensor descriptor count")
        names = []
        steps = len(memory["operator_order"])
        for t in tensors:
            if not isinstance(t["id"], str) or not t["id"] or t["id"] in names:
                raise ValueError("duplicate or invalid tensor identity")
            names.append(t["id"])
            shape = t["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) > 64
                or any(type(d) is not int or d <= 0 for d in shape)
            ):
                raise ValueError("invalid static tensor shape")
            if math.prod(shape) * 4 != t["size"] or t["dtype"] != "float32" or t["space"] != "dram":
                raise ValueError("invalid tensor size/type/space")
            for key in ("address", "size", "allocation_size"):
                _integer(t[key], key)
            if (
                t["alignment"] != 64
                or t["address"] % 64
                or t["address"] + t["allocation_size"] > dram
                or t["allocation_size"] != align(t["size"])
            ):
                raise ValueError("invalid tensor address/alignment")
            for key in ("first", "last"):
                _integer(t[key], key, steps)
            roles = t["roles"]
            if (
                not isinstance(roles, list)
                or not set(roles) <= {"input", "weight", "output", "intermediate"}
                or t["role_bits"]
                != sum(
                    bit
                    for name, bit in [("weight", 1), ("input", 2), ("output", 4)]
                    if name in roles
                )
            ):
                raise ValueError("invalid tensor roles")
            if set(roles) & {"input", "weight", "output"} and t["last"] != steps:
                raise ValueError("public buffers and weights must remain live")
            if set(roles) & {"input", "weight"} and t["first"] != 0:
                raise ValueError("inputs/weights must be live from entry")
            if "weight" in roles and t["address"] + t["size"] > len(files["weights.bin"]):
                raise ValueError("weight tensor exceeds data image")
            if t["weight_offset"] != (t["address"] if "weight" in roles else None):
                raise ValueError("weight offset disagrees with allocated address")
        for index, t in enumerate(tensors):
            for other in tensors[:index]:
                overlap = (
                    t["address"] < other["address"] + other["allocation_size"]
                    and other["address"] < t["address"] + t["allocation_size"]
                )
                if overlap and t["first"] <= other["last"] and other["first"] <= t["last"]:
                    raise ValueError("overlapping live DRAM allocations")
        for key in ("inputs", "outputs"):
            expected = [t["id"] for t in tensors if key[:-1] in t["roles"]]
            if len(set(manifest[key])) != len(manifest[key]) or set(expected) != set(manifest[key]):
                raise ValueError(f"invalid manifest {key}")
        if not manifest["outputs"]:
            raise ValueError("compiled program needs an output")
        decoded = decode(files["instructions.bin"])
        if manifest["instruction_count"] != len(decoded) or len(schedule["instructions"]) != len(
            decoded
        ):
            raise ValueError("instruction count mismatch")
        for raw, scheduled in zip(decoded, schedule["instructions"], strict=True):
            if {key: raw[key] for key in blank()} != {key: scheduled[key] for key in blank()}:
                raise ValueError("schedule operands disagree with binary instructions")
            if any(scheduled[key] != value for key, value in annotations(raw["op"]).items()):
                raise ValueError("schedule resource/precision/space mismatch")
            for key in ("tensor_a", "tensor_b", "tensor_c", "tensor_out"):
                if raw[key] != 0xFFFFFFFF and raw[key] >= len(tensors):
                    raise ValueError("invalid tensor reference in instruction")
        _validate_schedule(decoded, schedule, memory, manifest, h)
        return LoadedProgram(
            manifest, memory, schedule, files["instructions.bin"], files["weights.bin"]
        )
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid compiled program metadata: {exc}") from exc


def _validate_schedule(decoded, schedule, memory, manifest, hardware):
    from . import _native

    if schedule["strategy"] not in {"baseline", "memory-aware"}:
        raise ValueError("unsupported scheduling strategy")
    order = memory["operator_order"]
    if (
        not isinstance(order, list)
        or any(not isinstance(o, str) for o in order)
        or len(set(order)) != len(order)
        or schedule["operator_order"] != order
        or list(dict.fromkeys(i["node"] for i in schedule["instructions"])) != order
    ):
        raise ValueError("inconsistent operator order")
    simulated = _native.simulate(decoded, asdict(hardware))
    events = {e["id"]: e for e in simulated["trace"]}
    for i in schedule["instructions"]:
        e = events[i["id"]]
        if (i["start"], i["end"]) != (e["start"], e["end"]):
            raise ValueError("planned cycles disagree with native resource model")
    storage = memory["instruction_storage"]
    if storage != {
        "space": "instruction",
        "address": 0,
        "alignment": 4,
        "size": manifest["images"]["instructions.bin"]["size"],
    }:
        raise ValueError("instruction memory layout mismatch")
    allocations = {i["buffer"]: i for i in decoded if i["op"] == "ALLOC"}
    releases = {i["buffer"]: i for i in decoded if i["op"] == "FREE"}
    arenas = memory["scratchpad"]
    grouped = {}
    for instruction in decoded:
        grouped.setdefault(instruction["buffer"], []).append(instruction)
    if (
        len(allocations) != len(arenas)
        or set(allocations) != set(releases)
        or {a["buffer"] for a in arenas} != set(allocations)
    ):
        raise ValueError("scratchpad lifetime count mismatch")
    for a in arenas:
        alloc, free = allocations[a["buffer"]], releases[a["buffer"]]
        if (
            a["address"] != alloc["out"]
            or a["size"] != alloc["sram_bytes"]
            or a["alignment"] != 4
            or a["space"] != "sram"
            or a["first_instruction"] != alloc["id"]
            or a["last_instruction"] != free["id"]
            or a["start_cycle"] != events[alloc["id"]]["start"]
            or a["end_cycle"] != events[free["id"]]["end"]
            or a["address"] < 0
            or a["address"] + a["size"] > hardware.sram_bytes
        ):
            raise ValueError("scratchpad map disagrees with allocation instructions")
        parts = a["partitions"]
        by_role = {p["role"]: p for p in parts}
        if (
            len(by_role) != len(parts)
            or not set(by_role) <= {"a", "b", "c", "out"}
            or sum(p["size"] for p in parts) != a["size"]
        ):
            raise ValueError("invalid scratchpad partition roles or coverage")
        for index, part in enumerate(parts):
            if (
                part["address"] % 4
                or part["size"] <= 0
                or part["size"] % 4
                or part["address"] < a["address"]
                or part["address"] + part["size"] > a["address"] + a["size"]
            ):
                raise ValueError("scratchpad partition out of bounds")
            for other in parts[:index]:
                if (
                    part["address"] < other["address"] + other["size"]
                    and other["address"] < part["address"] + part["size"]
                ):
                    raise ValueError("overlapping scratchpad partitions")
        for i in grouped[a["buffer"]]:
            if i["op"] in {"MATMUL", "PIM_MATMUL", "FUSED_MATMUL_ADD_RELU"}:
                accesses = {"a": 4 * i["m"] * i["k"], "out": 4 * i["m"] * i["n"]}
                if i["op"] != "PIM_MATMUL":
                    accesses["b"] = 4 * i["k"] * i["n"]
                if i["tensor_c"] != 0xFFFFFFFF:
                    accesses["c"] = 4 * i["m"] * i["n"]
            elif i["op"] in {"ADD", "RELU"}:
                accesses = dict.fromkeys(
                    ["a", "out", *(["b"] if i["op"] == "ADD" else [])], 4 * i["count"]
                )
            else:
                continue
            for role, size in accesses.items():
                part = by_role.get(role)
                if part is None or part["address"] != i[role] or size > part["size"]:
                    raise ValueError("scratchpad partition disagrees with executable operands")
    tensors = memory["tensors"]
    sources, uses = {}, {i: [] for i in range(len(tensors))}
    positions = {node: i for i, node in enumerate(order)}
    for i in schedule["instructions"]:
        pos = positions[i["node"]]
        if i["op"] == "DMA_LOAD":
            uses[i["tensor_a"]].append(pos)
        if i["op"] == "DMA_STORE":
            if i["tensor_out"] in sources and sources[i["tensor_out"]] != pos:
                raise ValueError("multiple tensor producers in image")
            sources[i["tensor_out"]] = pos
        if i["op"] not in {"MATMUL", "PIM_MATMUL", "FUSED_MATMUL_ADD_RELU", "ADD", "RELU"}:
            continue
        if i["tensor_out"] in sources and sources[i["tensor_out"]] != pos:
            raise ValueError("multiple tensor producers in image")
        sources[i["tensor_out"]] = pos
        for key in ("tensor_a", "tensor_b", "tensor_c"):
            if i[key] != 0xFFFFFFFF:
                uses[i[key]].append(pos)
    node_times = {}
    for instruction in schedule["instructions"]:
        span = node_times.setdefault(
            instruction["node"], [instruction["start"], instruction["end"]]
        )
        span[0] = min(span[0], instruction["start"])
        span[1] = max(span[1], instruction["end"])
    for index, t in enumerate(tensors):
        first = sources.get(index, 0)
        last = len(order) if t["role_bits"] else max([first, *uses[index]])
        if (t["first"], t["last"]) != (first, last):
            raise ValueError("tensor lifetime disagrees with instruction uses")
        start = node_times[order[first]][0] if order else 0
        end = simulated["cycles"] if last == len(order) else node_times[order[last]][1]
        if (t["start_cycle"], t["end_cycle"]) != (start, end):
            raise ValueError("tensor cycle lifetime disagrees with schedule")

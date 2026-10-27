"""Versioned little-endian instruction images and concrete tile operands."""

from __future__ import annotations

import math
import struct

from .backend import Program

MAGIC = b"NPUFIMG\0"
HEADER = struct.Struct("<8sHHIQQ")
RECORD = struct.Struct("<8I4I4Q8Q6I2f80x")
OPCODES = [
    "",
    "ALLOC",
    "FREE",
    "DMA_LOAD",
    "DMA_STORE",
    "MATMUL",
    "ADD",
    "RELU",
    "FUSED_MATMUL_ADD_RELU",
    "PIM_MATMUL",
]
NONE = 0xFFFFFFFF
IDS = ["tensor_a", "tensor_b", "tensor_c", "tensor_out"]
ADDRESSES = ["a", "b", "c", "out"]
COUNTERS = [
    "sram_bytes",
    "working_set_bytes",
    "bytes",
    "macs",
    "elements",
    "local_read_bytes",
    "flat_start",
    "count",
]
DIMS = ["m0", "n0", "k0", "m", "n", "k"]


def annotations(op: str) -> dict:
    resources = (
        []
        if op in {"ALLOC", "FREE"}
        else ["dma", "dram"]
        if op.startswith("DMA")
        else ["pim", "dram"]
        if op == "PIM_MATMUL"
        else ["compute"]
    )
    spaces = (
        {"out": "sram"}
        if op == "ALLOC"
        else {}
        if op == "FREE"
        else {"a": "dram", "out": "sram"}
        if op == "DMA_LOAD"
        else {"a": "sram", "out": "dram"}
        if op == "DMA_STORE"
        else {
            "a": "sram",
            "b": "dram" if op == "PIM_MATMUL" else "sram",
            "c": "sram",
            "out": "sram",
        }
    )
    return {"dtype": "float32", "resources": resources, "spaces": spaces}


def blank() -> dict:
    return {
        "id": 0,
        "op": "MATMUL",
        "deps": [],
        "buffer": 0,
        "flags": 0,
        "mode": 0,
        **dict.fromkeys(IDS, NONE),
        **dict.fromkeys(ADDRESSES + COUNTERS + DIMS, 0),
        "alpha": 1.0,
        "beta": 1.0,
    }


def _uint(value, bits):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(f"binary field requires uint{bits}, got {value!r}")
    return value


def encode(instructions: list[dict]) -> bytes:
    records, deps = [], []
    for position, source in enumerate(instructions):
        if source.keys() - blank().keys() - {"node", "dtype", "resources", "spaces"}:
            raise ValueError("unknown hardware instruction fields")
        i = blank() | source
        if i["id"] != position or i["op"] not in OPCODES[1:]:
            raise ValueError("invalid instruction ID or opcode")
        for key, value in annotations(i["op"]).items():
            if key in source and source[key] != value:
                raise ValueError("instruction type/resource/space disagrees with opcode")
        if any(type(d) is not int or not 0 <= d < len(instructions) for d in i["deps"]):
            raise ValueError("invalid instruction dependency")
        begin = len(deps)
        deps.extend(i["deps"])
        head = [
            OPCODES.index(i["op"]),
            i["id"],
            i["flags"],
            i["mode"],
            i["buffer"],
            begin,
            len(i["deps"]),
            1,
        ]
        values = [_uint(v, 32) for v in head + [i[k] for k in IDS]]
        values += [_uint(i[k], 64) for k in ADDRESSES + COUNTERS]
        values += [_uint(i[k], 32) for k in DIMS]
        for key in ("alpha", "beta"):
            value = i[key]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("binary coefficients must be finite numbers")
            values.append(value)
        try:
            records.append(RECORD.pack(*values))
        except (struct.error, OverflowError) as exc:
            raise ValueError(f"invalid binary coefficient/operand: {exc}") from exc
    return (
        HEADER.pack(MAGIC, 1, RECORD.size, 1, len(records), len(deps))
        + b"".join(records)
        + b"".join(struct.pack("<I", d) for d in deps)
    )


def decode(data: bytes) -> list[dict]:
    from . import _native

    return [dict(i) | annotations(i["op"]) for i in _native.decode_image(data)]


def materialize(program: Program, memory: dict) -> list[dict]:
    tensors = {t["id"]: (index, t) for index, t in enumerate(memory["tensors"])}
    groups = {}
    for i in program.instructions:
        groups.setdefault(i["buffer"], []).append(i)
    areas = {}
    for buffer, group in groups.items():
        alloc = group[0]
        matrix = next((i for i in group if "MATMUL" in i["op"]), None)
        if matrix:
            m = matrix["ranges"]["m"][1] - matrix["ranges"]["m"][0]
            n = matrix["ranges"]["n"][1] - matrix["ranges"]["n"][0]
            k = max(i["ranges"]["k"][1] - i["ranges"]["k"][0] for i in group if "MATMUL" in i["op"])
            sizes = [
                4 * m * k,
                0 if matrix["op"] == "PIM_MATMUL" else 4 * k * n,
                4 * m * n if len(matrix["inputs"]) == 3 else 0,
                4 * m * n,
            ]
        else:
            vector = next(i for i in group if i["op"] in {"ADD", "RELU"})
            size = 4 * vector["elements"]
            sizes = [size, size if vector["op"] == "ADD" else 0, 0, size]
        a, b = 0, sizes[0]
        out = b + sizes[1]
        c = out + sizes[3]
        if sum(sizes) != alloc["sram_bytes"]:
            raise ValueError("tile layout disagrees with scratchpad reservation")
        areas[buffer] = [a, b, c, out]
        memory["scratchpad"].append(
            {
                "buffer": buffer,
                "space": "sram",
                "address": 0,
                "size": alloc["sram_bytes"],
                "alignment": 4,
                "first_instruction": alloc["id"],
                "last_instruction": group[-1]["id"],
                "partitions": [
                    {"role": role, "address": address, "size": size}
                    for role, address, size in zip(
                        ["a", "b", "c", "out"], [a, b, c, out], sizes, strict=True
                    )
                    if size
                ],
            }
        )
    result = []
    for old in program.instructions:
        i = blank()
        for key in ["id", "op", "deps", "buffer", *COUNTERS[:6]]:
            if key in old:
                i[key] = old[key]
        i["node"] = old["node"]
        for key, name in zip(
            IDS, [*old["inputs"], *[None] * (3 - len(old["inputs"])), old["output"]], strict=True
        ):
            i[key] = tensors[name][0] if name else NONE
        i.update(dict(zip(ADDRESSES, areas[old["buffer"]], strict=True)))
        gemm = old.get("gemm", {})
        i["alpha"], i["beta"] = gemm.get("alpha", 1.0), gemm.get("beta", 1.0)
        i["flags"] = (16 if gemm.get("transA") else 0) | (32 if gemm.get("transB") else 0)
        if "ranges" in old:
            for axis in ("m", "n", "k"):
                if axis in old["ranges"]:
                    lo, hi = old["ranges"][axis]
                    i[axis + "0"], i[axis] = lo, hi - lo
        if "flat_range" in old:
            lo, hi = old["flat_range"]
            i["flat_start"], i["count"] = lo, hi - lo
        if old["op"] == "ALLOC":
            i["out"] = 0
        elif old["op"] == "DMA_LOAD":
            slot = old["operand"]
            i["tensor_a"] = tensors[old["tensor"]][0]
            i["a"] = tensors[old["tensor"]][1]["address"]
            i["out"] = areas[old["buffer"]][slot]
            i["mode"] = slot if "ranges" in old else 3
        elif old["op"] == "DMA_STORE":
            i["tensor_a"] = tensors[old["output"]][0]
            i["a"] = areas[old["buffer"]][3]
            i["out"] = tensors[old["output"]][1]["address"]
            i["mode"] = 4 if "ranges" in old else 5
        elif "MATMUL" in old["op"]:
            final = old.get("gemm_epilogue", False) or old.get("fused_epilogue", False)
            relu = old.get("gemm_relu", False) or old.get("fused_epilogue", False)
            i["flags"] |= (
                int(old.get("accumulate", False))
                | (2 if final else 0)
                | (4 if relu else 0)
                | (8 if gemm else 0)
            )
            if old["op"] == "PIM_MATMUL":
                i["b"] = tensors[old["inputs"][1]][1]["address"]
        result.append(i | annotations(i["op"]))
    return result

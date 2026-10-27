"""Optional native MLIR frontend and bridge to the existing NPU backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ir import Graph, Operation, Tensor, gemm_attributes

OPCODES = {
    "MatMul": "matmul",
    "Add": "add",
    "Relu": "relu",
    "Gemm": "gemm",
    "FusedMatMulAddRelu": "fused_matmul_add_relu",
    "FusedGemmRelu": "fused_gemm_relu",
}
MAX_TEXT_BYTES = 64 * 1024 * 1024


def string(value: str) -> str:
    return (
        '"'
        + "".join(
            chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:02X}"
            for b in value.encode("utf-8")
        )
        + '"'
    )


def tensor_type(tensor: Tensor) -> str:
    return "tensor<" + "".join(f"{d}x" for d in tensor.shape) + "f32>"


def emit_graph(graph: Graph) -> str:
    graph.validate()
    if any(t.nbytes > 64 * 1024 * 1024 or len(t.shape) > 64 for t in graph.tensors.values()):
        raise ValueError("MLIR frontend supports tensors up to 64 MiB and rank 64")
    names = {name: f"%arg{i}" for i, name in enumerate(graph.inputs)}
    arguments = ", ".join(f"{names[n]}: {tensor_type(graph.tensors[n])}" for n in graph.inputs)
    result_types = ", ".join(tensor_type(graph.tensors[n]) for n in graph.outputs)
    input_names = ", ".join(map(string, graph.inputs))
    output_names = ", ".join(map(string, graph.outputs))
    lines = [
        "module attributes {npuforge.schema = 1 : i64, "
        f"npuforge.model = {string(graph.name)}}} {{",
        f"  func.func @main({arguments}) -> ({result_types}) attributes {{",
        f"    npuforge.input_names = [{input_names}], npuforge.output_names = [{output_names}]",
        "  } {",
    ]
    for name, tensor in sorted(graph.tensors.items()):
        if tensor.data is None:
            continue
        value = f"%v{len(names)}"
        names[name] = value
        kind = tensor_type(tensor)
        raw = tensor.data.astype("<f4", copy=False).tobytes().hex().upper()
        lines.append(
            f'    {value} = "npuforge.constant"() {{tensor_id = {string(name)}, '
            f'value = dense<"0x{raw}"> : {kind}}} : () -> {kind}'
        )
    for op in graph.ordered():
        attrs = [f"id = {string(op.id)}", f"tensor_id = {string(op.output)}"]
        if op.op in {"Gemm", "FusedGemmRelu"}:
            gemm = gemm_attributes(op)
            attrs += [f"{k} = {gemm[k]:.9e} : f32" for k in ("alpha", "beta")]
            attrs += [f"{k} = {gemm[k]} : i64" for k in ("transA", "transB")]
        if "fused_from" in op.attrs:
            attrs.append("fused_from = [" + ", ".join(map(string, op.attrs["fused_from"])) + "]")
        value = f"%v{len(names)}"
        names[op.output] = value
        inputs = ", ".join(names[n] for n in op.inputs)
        types = ", ".join(tensor_type(graph.tensors[n]) for n in op.inputs)
        out = tensor_type(graph.tensors[op.output])
        lines.append(
            f'    {value} = "npuforge.{OPCODES[op.op]}"({inputs}) '
            f"{{{', '.join(attrs)}}} : ({types}) -> {out}"
        )
    lines += [
        "    return " + ", ".join(names[n] for n in graph.outputs) + " : " + result_types,
        "  }",
        "}",
    ]
    text = "\n".join(lines) + "\n"
    if len(text.encode()) > MAX_TEXT_BYTES:
        raise ValueError("MLIR source exceeds the 64 MiB text limit")
    return text


def find_tool(path: str | Path | None = None) -> Path:
    candidate = (
        str(path) if path is not None else os.environ.get("NPUFORGE_MLIR_OPT", "npuforge-opt")
    )
    found = shutil.which(candidate)
    if found is None:
        raise ValueError(
            "npuforge-opt was not found; build mlir/ with LLVM/MLIR 19 and set "
            "NPUFORGE_MLIR_OPT, put it on PATH, or pass --mlir-opt"
        )
    return Path(found).resolve()


def invoke(arguments: list[str | Path], *, timeout: int = 180) -> str:
    try:
        completed = subprocess.run(
            [str(a) for a in arguments], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"MLIR tool exceeded {timeout} seconds: {arguments[0]}") from exc
    if completed.returncode:
        raise ValueError(f"MLIR tool failed: {completed.stderr.strip()}")
    return completed.stdout


def import_bridge(data: str) -> Graph:
    try:
        document = json.loads(data)
        if document["schema"] != "npuforge.mlir.bridge.v1":
            raise ValueError("unsupported MLIR graph bridge schema")
        tensors = {}
        for t in document["tensors"]:
            if t["id"] in tensors:
                raise ValueError("duplicate MLIR tensor identity")
            raw = t["data_hex"]
            value = None
            if raw is not None:
                value = np.frombuffer(bytes.fromhex(raw), dtype="<f4").copy().reshape(t["shape"])
            tensors[t["id"]] = Tensor(t["id"], tuple(t["shape"]), t["dtype"], value)
        graph = Graph(
            tensors,
            [Operation(**op) for op in document["operations"]],
            document["inputs"],
            document["outputs"],
            document["name"],
        )
        graph.validate()
        return graph
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MLIR graph bridge: {exc}") from exc


@dataclass
class MLIRCompilation:
    graph: Graph
    changes: dict
    artifacts: dict[str, bytes]
    metadata: dict


def compile_graph(
    graph: Graph, enabled: bool = True, tool: str | Path | None = None
) -> MLIRCompilation:
    binary = find_tool(tool)
    version = invoke([binary, "--version"]).strip()
    source = emit_graph(graph)
    with tempfile.TemporaryDirectory(prefix="npuforge-mlir-") as temporary:
        root = Path(temporary)
        original, optimized = root / "original.mlir", root / "optimized.mlir"
        bridge, lowered = root / "frontend.graph.json", root / "lowered.linalg.mlir"
        original.write_text(source)
        pipeline = ["--npuforge-verify"]
        if enabled:
            pipeline.append("--npuforge-optimize")
        invoke(
            [
                binary,
                original,
                *pipeline,
                f"--npuforge-export-graph=output={bridge}",
                "-o",
                optimized,
            ]
        )
        result = import_bridge(bridge.read_text())
        invoke([binary, optimized, "--convert-npuforge-to-linalg", "-o", lowered])
        artifacts = {p.name: p.read_bytes() for p in [original, optimized, bridge, lowered]}
    if result.inputs != graph.inputs or result.outputs != graph.outputs:
        raise ValueError("MLIR changed the public input/output contract")
    for name in graph.inputs + graph.outputs:
        if result.tensors[name].shape != graph.tensors[name].shape:
            raise ValueError("MLIR changed a public tensor shape")
    for name, tensor in result.tensors.items():
        if tensor.data is not None:
            original = graph.tensors.get(name)
            if (
                original is None
                or original.data is None
                or tensor.data.tobytes() != original.data.tobytes()
            ):
                raise ValueError("MLIR changed initializer values")
    originals = {op.id: op for op in graph.operations}
    folded = sum(
        len(op.attrs["fused_from"]) - 1
        for op in result.operations
        if "fused_from" in op.attrs and originals[op.id].op != op.op
    )
    dead = len(graph.operations) - len(result.operations) - folded
    if dead < 0 or (not enabled and (dead or folded)):
        raise ValueError("inconsistent MLIR transformation counts")
    changes = {
        "enabled": enabled,
        "original_operations": len(graph.operations),
        "dead_operations_removed": dead,
        "fusion_patterns": sum(
            op.op in {"FusedMatMulAddRelu", "FusedGemmRelu"} for op in result.operations
        ),
        "final_operations": len(result.operations),
    }
    metadata = {
        "schema": "npuforge.mlir.v1",
        "toolchain": version,
        "passes": [*pipeline, "npuforge-export-graph", "convert-npuforge-to-linalg"],
        "binary_backend_input": "frontend.graph.json",
        "standard_lowering": "lowered.linalg.mlir",
    }
    artifacts["mlir.json"] = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    return MLIRCompilation(result, changes, artifacts, metadata)


def lower_to_llvm(source: bytes, llvm_bin: str | Path | None = None) -> dict[str, bytes]:
    """Run upstream MLIR bufferization/conversion and LLVM IR verification."""
    tools = {}
    for name in ("mlir-opt", "mlir-translate", "opt"):
        candidate = str(Path(llvm_bin) / name) if llvm_bin else shutil.which(name + "-19") or name
        found = shutil.which(candidate)
        if found is None or "version 19." not in invoke([found, "--version"]):
            raise ValueError(f"LLVM/MLIR 19 {name} is required; provide --llvm-bin")
        tools[name] = found
    with tempfile.TemporaryDirectory(prefix="npuforge-llvm-") as temporary:
        root = Path(temporary)
        original, lowered, ir = root / "linalg.mlir", root / "lowered.llvm.mlir", root / "model.ll"
        original.write_bytes(source)
        invoke(
            [
                tools["mlir-opt"],
                original,
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map",
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--convert-scf-to-cf",
                "--expand-strided-metadata",
                "--convert-arith-to-llvm",
                "--finalize-memref-to-llvm",
                "--llvm-request-c-wrappers",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
                "-o",
                lowered,
            ]
        )
        invoke([tools["mlir-translate"], "--mlir-to-llvmir", lowered, "-o", ir])
        invoke([tools["opt"], "-passes=verify", "-disable-output", ir])
        return {p.name: p.read_bytes() for p in [lowered, ir]}

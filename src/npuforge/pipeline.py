"""The public end-to-end compilation API and inspectable run artifacts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backend import Hardware, Program, lower
from .compiler import optimize
from .functional import make_inputs, verify_outputs
from .image import CompiledImage, build_image, json_bytes, load_program
from .importer import import_onnx
from .ir import Graph
from .memory import MAX_MEMORY, schedule_operations


@dataclass
class Run:
    model: str
    original: Graph
    graph: Graph
    changes: dict
    program: Program
    metrics: dict
    verification: dict
    compiled: CompiledImage | None = None
    inputs: dict | None = None
    outputs: dict | None = None
    frontend_artifacts: dict[str, bytes] | None = None

    def summary(self) -> dict:
        return {
            "model": self.model,
            "architecture": self.program.hardware.architecture,
            "transformations": self.changes,
            "verification": self.verification,
            "metrics": {k: v for k, v in self.metrics.items() if k != "trace"},
        }

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "original.ir.json").write_text(self.original.to_json())
        (directory / "optimized.ir.json").write_text(self.graph.to_json())
        (directory / "instructions.json").write_text(self.program.to_json())
        if self.compiled is not None:
            self.compiled.write(directory)
            np.savez(directory / "inputs.npz", **self.inputs)
            np.savez(directory / "outputs.npz", **self.outputs)
        if self.frontend_artifacts:
            for name, data in self.frontend_artifacts.items():
                (directory / name).write_bytes(data)
        for name, data in [("simulation", self.metrics), ("summary", self.summary())]:
            (directory / f"{name}.json").write_text(
                json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )


def compile_model(
    path: str | Path,
    hardware: Hardware | None = None,
    optimizations: bool = True,
    *,
    scheduler: str = "memory-aware",
    reuse: bool = True,
    dram_limit: int = MAX_MEMORY,
    compiler: str = "python",
    mlir_opt: str | Path | None = None,
    emit_llvm: bool = False,
    llvm_bin: str | Path | None = None,
) -> Run:
    original = import_onnx(path)
    frontend = None
    if compiler == "mlir":
        from .mlir import compile_graph, lower_to_llvm

        frontend = compile_graph(original, optimizations, mlir_opt)
        if emit_llvm:
            frontend.artifacts.update(
                lower_to_llvm(frontend.artifacts["lowered.linalg.mlir"], llvm_bin)
            )
            frontend.metadata["llvm_lowering"] = "model.ll (verified LLVM 19 IR with C wrappers)"
            frontend.artifacts["mlir.json"] = json_bytes(frontend.metadata)
        graph, changes = frontend.graph, frontend.changes
    elif compiler == "python" and mlir_opt is None and not emit_llvm and llvm_bin is None:
        graph, changes = optimize(original, optimizations)
    else:
        raise ValueError("compiler must be python or mlir; MLIR/LLVM options require compiler=mlir")
    order = schedule_operations(graph, scheduler)
    program = lower(graph, hardware, operation_order=order)
    compiled = build_image(graph, program, strategy=scheduler, reuse=reuse, dram_limit=dram_limit)
    if frontend is not None:
        compiled.manifest["compiler_frontend"] = {"name": "mlir", **frontend.metadata}
        compiled.files["manifest.json"] = json_bytes(compiled.manifest)
    inputs = make_inputs(graph)
    with tempfile.TemporaryDirectory(prefix="npuforge-") as directory:
        compiled.write(directory)
        outputs, metrics = load_program(directory).run(inputs)
    verification = verify_outputs(path, inputs, outputs)
    return Run(
        Path(path).name,
        original,
        graph,
        changes,
        program,
        metrics,
        verification,
        compiled,
        inputs,
        outputs,
        frontend.artifacts if frontend else None,
    )

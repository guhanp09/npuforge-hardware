"""Accessible CLI with real execution, checked errors, and portable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .backend import Hardware
from .benchmark import run_backend_benchmarks, run_benchmarks
from .functional import verify_outputs
from .image import load_program
from .importer import import_onnx
from .pipeline import Run, compile_model
from .reports import render_report


def program_directory(path: Path) -> Path:
    if (path / "manifest.json").is_file():
        return path
    candidates = [
        path / a for a in ("conventional", "pim") if (path / a / "manifest.json").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "select a directory containing manifest.json "
        "(choose an architecture subdirectory for comparisons)"
    )


def runtime_inputs(path: Path | None, loaded, *, generate: bool = False) -> dict:
    if path is None:
        if not loaded.manifest["inputs"]:
            return {}
        if not generate:
            raise ValueError("provide --inputs with an NPZ file, or NPY for a single input")
        rng = np.random.default_rng(7)
        tensors = {t["id"]: t for t in loaded.memory["tensors"]}
        return {
            name: rng.normal(size=tensors[name]["shape"]).astype(np.float32)
            for name in loaded.manifest["inputs"]
        }
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        if len(loaded.manifest["inputs"]) != 1:
            raise ValueError(
                "NPY input requires exactly one declared input; use NPZ for multiple inputs"
            )
        return {loaded.manifest["inputs"][0]: data}
    with data:
        return {name: data[name] for name in data.files}


def print_run(label: str, run: Run) -> None:
    m = run.metrics
    c = run.changes
    print(f"\n{label} · {run.model} · {run.program.hardware.architecture}")
    print("  [PASS] ONNX validation / graph import / IR verification")
    print(
        f"  [PASS] Optimization: {c['original_operations']} → {c['final_operations']} ops; "
        f"{c['dead_operations_removed']} dead; {c['fusion_patterns']} fused"
    )
    print(f"  [PASS] Hardware lowering: {len(run.program.instructions):,} instructions")
    print("  [PASS] Native C++ simulation / numerical verification against ONNX Runtime")
    if run.compiled is not None:
        if run.frontend_artifacts:
            print(
                "  [PASS] Native MLIR verification / SSA rewrites / Linalg lowering / graph bridge"
            )
        print(
            f"  [PASS] Address allocation / {run.compiled.schedule['strategy']} scheduling / "
            "binary encoding and loading"
        )
        print(
            f"  Images {m['instruction_image_bytes']:,} B instructions, "
            f"{m['weight_image_bytes']:,} B weights; "
            f"DRAM allocation {m['dram_allocation_bytes']:,} B"
        )
    print(
        f"  Cycles {m['cycles']:,}  |  Latency {m['latency_ns']:,.0f} ns  |  "
        f"Off-chip {m['offchip_bytes']:,} B  |  SRAM peak {m['peak_sram_bytes']:,} B"
    )
    print(
        f"  Compute {m['compute_utilization']:.1%}  |  DMA {m['dma_utilization']:.1%}  |  "
        f"PIM {m['pim_utilization']:.1%}  |  Pressure: {m['bottleneck']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NPUForge — ONNX-to-NPU compiler & hardware-aware simulator"
    )
    parser.add_argument("--version", action="version", version=f"NPUForge {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "benchmark"):
        command = commands.add_parser(name, help="run both demos and all six comparisons")
        command.add_argument("--output", type=Path, default=Path("output"))
        command.add_argument("--config", type=Path, help="JSON hardware configuration")
        if name == "benchmark":
            command.add_argument("model", nargs="?", type=Path)
            command.add_argument("--compare-schedulers", action="store_true")
    compile_parser = commands.add_parser("compile", help="compile, simulate, verify and report")
    compile_parser.add_argument("model", type=Path)
    compile_parser.add_argument("--output", type=Path, default=Path("output"))
    compile_parser.add_argument("--config", type=Path)
    compile_parser.add_argument("--arch", "--target", choices=["conventional", "pim"])
    compile_parser.add_argument("--no-opt", action="store_true")
    compile_parser.add_argument(
        "--schedule", choices=["baseline", "memory-aware"], default="memory-aware"
    )
    compile_parser.add_argument("--no-reuse", action="store_true")
    compile_parser.add_argument("--dram-limit", type=int, default=64 * 1024 * 1024)
    compile_parser.add_argument(
        "--compare", action="store_true", help="also run the other architecture"
    )
    inspect_parser = commands.add_parser("inspect", help="validate and print imported IR")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    for command in (compile_parser, commands.choices["demo"], commands.choices["benchmark"]):
        command.add_argument("--compiler", choices=["python", "mlir"], default="python")
        command.add_argument(
            "--mlir-opt", type=Path, help="path to the optional native npuforge-opt"
        )
        command.add_argument(
            "--emit-llvm", action="store_true", help="also lower MLIR to verified LLVM IR"
        )
        command.add_argument("--llvm-bin", type=Path, help="LLVM/MLIR 19 bin directory")
    for name in ("run", "verify", "inspect-program"):
        command = commands.add_parser(name)
        if name == "verify":
            command.add_argument("model", type=Path)
            command.add_argument("--compiled", type=Path, required=True)
        else:
            command.add_argument("compiled", type=Path)
        if name in ("run", "verify"):
            command.add_argument("--inputs", "--input", type=Path)
        if name == "run":
            command.add_argument("--output", type=Path, default=Path("output/outputs.npz"))
        if name == "inspect-program":
            command.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "verify", "inspect-program"}:
            loaded = load_program(program_directory(args.compiled))
            if args.command == "inspect-program":
                value = {
                    "manifest": loaded.manifest,
                    "memory_map": loaded.memory,
                    "schedule": loaded.schedule,
                }
                print(
                    json.dumps(value, indent=2, sort_keys=True)
                    if args.json
                    else f"{loaded.manifest['model']}: "
                    f"{loaded.manifest['instruction_count']} instructions, "
                    f"{loaded.memory['dram_bytes']} DRAM bytes; {loaded.schedule['strategy']}"
                )
                return 0
            inputs = runtime_inputs(args.inputs, loaded, generate=args.command == "verify")
            outputs, metrics = loaded.run(inputs)
            if args.command == "verify":
                print(json.dumps(verify_outputs(args.model, inputs, outputs), indent=2))
            else:
                output = (
                    args.output if args.output.suffix == ".npz" else args.output.with_suffix(".npz")
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez(output, **outputs)
                output.with_suffix(".metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"Executed {loaded.manifest['instruction_count']} decoded instructions; "
                    f"{metrics['cycles']} simulated cycles.\nOutputs: {output}"
                )
            return 0
        if args.command == "inspect":
            graph = import_onnx(args.model)
            print(graph.to_json() if args.json else str(graph))
            return 0
        config = Hardware.load(args.config) if args.config else Hardware()
        compiler_options = {
            "compiler": args.compiler,
            "mlir_opt": args.mlir_opt,
            "emit_llvm": args.emit_llvm,
            "llvm_bin": args.llvm_bin,
        }
        print("NPUForge\nONNX-to-NPU Compiler & Hardware Simulator")
        if args.command in {"demo", "benchmark"}:
            if args.command == "benchmark" and (args.compare_schedulers or args.model):
                runs = run_backend_benchmarks(args.output, args.model, config, **compiler_options)
            else:
                runs = run_benchmarks(args.output, config, **compiler_options)
        else:
            if args.arch:
                config = config.alternative(args.arch)
            options = {
                "scheduler": args.schedule,
                "reuse": not args.no_reuse,
                "dram_limit": args.dram_limit,
                **compiler_options,
            }
            run = compile_model(args.model, config, not args.no_opt, **options)
            run.write(args.output / config.architecture)
            runs = [(config.architecture, run)]
            if args.compare:
                other = "pim" if config.architecture == "conventional" else "conventional"
                alternative = compile_model(
                    args.model, config.alternative(other), not args.no_opt, **options
                )
                alternative.write(args.output / other)
                runs.append((other, alternative))
        for label, run in runs:
            print_run(label, run)
        report = render_report(runs, args.output / "report.html")
        print(f"\nReport: {report}\nArtifacts: {args.output}")
        return 0
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(
            f"npuforge: {exc}\nCheck model support, file paths and hardware capacities. "
            "See README.md and docs/architecture.md.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

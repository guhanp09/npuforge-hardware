"""Deterministic comparisons; all rows are produced by compiler/native execution."""

import json
from dataclasses import replace
from pathlib import Path

from .backend import Hardware
from .models import generate_model
from .pipeline import Run, compile_model


def run_benchmarks(
    directory: str | Path, hardware: Hardware | None = None, **compiler_options
) -> list[tuple[str, Run]]:
    directory = Path(directory)
    h = hardware or Hardware()
    matmul = generate_model(directory / "models" / "demo_matmul.onnx", "matmul")
    mlp = generate_model(directory / "models" / "demo_mlp.onnx", "mlp")
    specs = [
        ("matmul_conventional", matmul, h.alternative("conventional"), True),
        ("matmul_pim", matmul, h.alternative("pim"), True),
        (
            "matmul_tile8",
            matmul,
            replace(h, architecture="conventional", tile_m=8, tile_n=8, tile_k=8),
            True,
        ),
        ("mlp_unoptimized", mlp, h.alternative("conventional"), False),
        ("mlp_optimized", mlp, h.alternative("conventional"), True),
        ("mlp_pim", mlp, h.alternative("pim"), True),
    ]
    runs = []
    rows = []
    for label, path, config, enabled in specs:
        run = compile_model(path, config, enabled, **compiler_options)
        run.write(directory / label)
        runs.append((label, run))
        rows.append(
            {
                "case": label,
                "hardware": run.program.to_dict()["hardware"],
                "model_dimensions": [16, 32, 24],
                "seed": 11,
                **run.summary(),
            }
        )
    (directory / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (directory / "results.md").write_text(benchmark_table(runs))
    return runs


def run_fusion_benchmarks(directory: str | Path) -> list[tuple[str, Run]]:
    """Compare one Gemm/Relu workload with both passes and both architectures."""
    directory = Path(directory)
    model = generate_model(directory / "models" / "demo_gemm_relu.onnx", "gemm_relu")
    runs, rows = [], []
    for architecture in ("conventional", "pim"):
        for enabled in (False, True):
            label = architecture + ("_fused" if enabled else "_unoptimized")
            run = compile_model(model, Hardware(architecture=architecture), enabled)
            run.write(directory / label)
            runs.append((label, run))
            rows.append(
                {
                    "case": label,
                    "hardware": run.program.to_dict()["hardware"],
                    "model_dimensions": [16, 32, 24],
                    "seed": 11,
                    **run.summary(),
                }
            )
    (directory / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (directory / "results.md").write_text(benchmark_table(runs))
    return runs


def run_backend_benchmarks(
    directory: str | Path,
    model: str | Path | None = None,
    hardware: Hardware | None = None,
    **compiler_options,
) -> list[tuple[str, Run]]:
    directory = Path(directory)
    if model is None:
        model = generate_model(directory / "models" / "branches.onnx", "branches", 8, 8, 64)
    runs, rows = [], []
    for label, scheduler, reuse in [
        ("baseline", "baseline", True),
        ("memory_aware", "memory-aware", True),
        ("memory_aware_no_reuse", "memory-aware", False),
    ]:
        run = compile_model(model, hardware, scheduler=scheduler, reuse=reuse, **compiler_options)
        run.write(directory / label)
        runs.append((label, run))
        rows.append({"case": label, "scheduler": scheduler, "reuse": reuse, **run.summary()})
    (directory / "results.json").write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n")
    (directory / "results.md").write_text(benchmark_table(runs))
    return runs


def run_mlir_benchmarks(
    directory: str | Path, mlir_opt: str | Path | None = None
) -> list[tuple[str, Run]]:
    """Compare both compiler paths on identical MLPs and hardware settings."""
    directory = Path(directory)
    model = generate_model(directory / "models" / "mlp.onnx", "mlp")
    runs, rows = [], []
    for architecture in ("conventional", "pim"):
        for compiler in ("python", "mlir"):
            for enabled in (False, True):
                label = f"{architecture}_{compiler}_" + ("optimized" if enabled else "unoptimized")
                run = compile_model(
                    model,
                    Hardware(architecture=architecture),
                    enabled,
                    compiler=compiler,
                    mlir_opt=mlir_opt if compiler == "mlir" else None,
                )
                run.write(directory / label)
                runs.append((label, run))
                rows.append({"case": label, "compiler": compiler, **run.summary()})
    (directory / "results.json").write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n")
    (directory / "results.md").write_text(benchmark_table(runs))
    return runs


def benchmark_table(runs: list[tuple[str, Run]]) -> str:
    lines = [
        "| Case | Cycles | Latency (ns) | Off-chip bytes | PIM-local bytes | "
        "Peak SRAM | DRAM allocation |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, run in runs:
        m = run.metrics
        lines.append(
            f"| {label} | {m['cycles']} | {m['latency_ns']:.0f} | "
            f"{m['offchip_bytes']} | {m['pim_local_read_bytes']} | {m['peak_sram_bytes']} | "
            f"{m['dram_allocation_bytes']} |"
        )
    return "\n".join(lines) + "\n"

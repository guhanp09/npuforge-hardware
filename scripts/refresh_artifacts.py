"""Regenerate checked-in benchmarks and real execution figures after installation."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from npuforge.benchmark import (
    run_backend_benchmarks,
    run_benchmarks,
    run_fusion_benchmarks,
    run_mlir_benchmarks,
)
from npuforge.reports import graph_svg, render_report, timeline_svg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mlir", action="store_true", help="also refresh the optional MLIR comparison")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
runs = run_benchmarks(root / "output" / "reference")
render_report(runs, root / "output" / "report.html")
figures = root / "docs" / "figures"
figures.mkdir(parents=True, exist_ok=True)
mlp = dict(runs)["mlp_optimized"]
(figures / "optimized_graph.svg").write_text(graph_svg(mlp.graph))
(figures / "timeline.svg").write_text(timeline_svg(mlp))
for name in ("results.json", "results.md"):
    (root / "benchmarks" / name).write_bytes((root / "output" / "reference" / name).read_bytes())
fusion_dir = root / "output" / "gemm-fusion"
fusion_runs = run_fusion_benchmarks(fusion_dir)
render_report(fusion_runs, fusion_dir / "report.html")
for name in ("results.json", "results.md"):
    (root / "benchmarks" / f"gemm_fusion_{name}").write_bytes((fusion_dir / name).read_bytes())
backend_dir = root / "output" / "schedulers"
backend_runs = run_backend_benchmarks(backend_dir)
render_report(backend_runs, backend_dir / "report.html")
for name in ("results.json", "results.md"):
    (root / "benchmarks" / f"backend_{name}").write_bytes((backend_dir / name).read_bytes())
if args.mlir:
    mlir_dir = root / "output" / "mlir-comparison"
    mlir_runs = run_mlir_benchmarks(mlir_dir)
    render_report(mlir_runs, mlir_dir / "report.html")
    for name in ("results.json", "results.md"):
        (root / "benchmarks" / f"mlir_{name}").write_bytes((mlir_dir / name).read_bytes())
packages = {}
for name in (
    "npuforge",
    "numpy",
    "onnx",
    "onnxruntime",
    "pybind11",
    "scikit-build-core",
    "pytest",
    "ruff",
    "cmake",
    "clang-format",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # pip's isolated native build need not install build dependencies into the host.
        packages[name] = "not installed in host environment"
environment = {
    "python": platform.python_version(),
    "platform": platform.system(),
    "machine": platform.machine(),
    "packages": packages,
}
(root / "benchmarks" / "environment.json").write_text(
    json.dumps(environment, sort_keys=True, indent=2) + "\n"
)
print("Refreshed benchmarks, graph, timeline, and output/report.html from actual native runs.")

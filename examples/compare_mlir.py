"""Compare native MLIR and Python optimizers using the same binary NPU backend."""

import argparse
from pathlib import Path

from npuforge.benchmark import benchmark_table, run_mlir_benchmarks
from npuforge.reports import render_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/mlir-comparison"))
parser.add_argument("--mlir-opt", type=Path)
args = parser.parse_args()
runs = run_mlir_benchmarks(args.output, args.mlir_opt)
report = render_report(runs, args.output / "report.html")
print(benchmark_table(runs))
print(f"All eight binary programs verified against ONNX Runtime.\nReport: {report}")

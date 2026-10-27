#!/bin/sh
# Disposable Debian/Python container with the source mounted read-only at /source.
set -eu
apt-get update -qq
apt-get install -y --no-install-recommends build-essential git libmlir-19-dev \
  mlir-19-tools llvm-19-dev clang-19 zlib1g-dev libzstd-dev >/dev/null
mkdir -p /tmp/npuforge
tar -C /source --exclude='.venv*' --exclude=.git --exclude=output --exclude=build \
  --exclude=dist --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache \
  -cf - . | tar -C /tmp/npuforge -xf -
cd /tmp/npuforge
python -m pip install --quiet '.[dev]'
python scripts/build_mlir.py --llvm-prefix /usr/lib/llvm-19
export NPUFORGE_MLIR_OPT=/tmp/npuforge/build/mlir/npuforge-opt
export NPUFORGE_LLVM_BIN=/usr/lib/llvm-19/bin
ruff check .
ruff format --check .
clang-format --dry-run --Werror cpp/include/*.hpp cpp/src/*.cpp cpp/tests/*.cpp \
  mlir/include/*.h mlir/lib/*.cpp mlir/*.cpp
python -m pytest -q
npuforge demo --compiler mlir --output output/mlir-demo
npuforge compile output/mlir-demo/models/demo_mlp.onnx --compiler mlir \
  --emit-llvm --llvm-bin "$NPUFORGE_LLVM_BIN" --output output/mlir-llvm
npuforge run output/mlir-demo/mlp_optimized \
  --inputs output/mlir-demo/mlp_optimized/inputs.npz --output output/replayed.npz
python examples/compare_mlir.py
python -m build
python - <<'PY'
import json
from pathlib import Path
actual = json.loads(Path('output/mlir-comparison/results.json').read_text())
reference = json.loads(Path('benchmarks/mlir_results.json').read_text())
for a, b in zip(actual, reference, strict=True):
    assert a['case'] == b['case']
    assert a['metrics'] == b['metrics'], a['case']
    assert a['verification']['passed']
print('LINUX MLIR VERIFIED: all eight compiler comparison metric sets match macOS.')
PY

"""Build the optional MLIR 19 tool without changing the core package installation."""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--llvm-prefix", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("build/mlir"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
venv_bin = Path(sys.executable).parent
cmake = shutil.which("cmake") or str(venv_bin / "cmake")
ninja = shutil.which("ninja") or str(venv_bin / "ninja")
arguments = [
    cmake,
    "-S",
    str(root / "mlir"),
    "-B",
    str(args.output),
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DCMAKE_MAKE_PROGRAM={ninja}",
    f"-DMLIR_DIR={args.llvm_prefix.resolve() / 'lib/cmake/mlir'}",
]
if platform.system() == "Darwin":
    arguments.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={platform.mac_ver()[0]}")
subprocess.run(arguments, check=True)
subprocess.run([cmake, "--build", str(args.output), "--parallel", "2"], check=True)
print(f"Native MLIR tool: {(args.output / 'npuforge-opt').resolve()}")

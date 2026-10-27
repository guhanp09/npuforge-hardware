# MLIR compiler integration

The optional compiler path is an out-of-tree C++17 tool built against LLVM/MLIR
**19.x**. Its TableGen operations are registered with MLIR: parsing, SSA dominance,
type checking, dialect verifiers, rewrite patterns, the pass manager and dialect
conversion all run in the actual MLIR library. No unregistered-dialect parsing or
text substitution implements the compiler transformations.

## Two paths, one NPU backend

```
ONNX → existing graph IR → npuforge tensor dialect
                            │
                     native SSA rewrites
                            │
                    verified optimized MLIR
                     ┌──────┴──────────┐
             graph bridge JSON     Linalg / Tensor / Arith
                     │                 │
             existing NPU backend  upstream bufferization and conversion
                     │                 │
             binary images         LLVM dialect → LLVM IR
                     │                 │
             native NPU execution  Clang → CPU execution in tests
```

The bridge reconstructs the existing `Graph` class from the native tool's output;
it does not substitute the original Python graph. The compiler's normal binary
load/execute/ORT verification then runs unchanged. The optimized graph is the input
to NPU code generation; Linalg/LLVM is a separate CPU lowering, not a second NPU
instruction encoder or a commercial accelerator target. Timing metrics always come
from the existing NPU resource model, not CPU execution time.

## Dialect and passes

`npuforge.constant`, `matmul`, `add`, `relu`, `gemm`, `fused_matmul_add_relu` and
`fused_gemm_relu` produce static ranked FP32 tensors. Constants contain actual
DenseElementsAttr initializer bytes. Operations are pure; `func.return` keeps public
values observable. A single `func.func @main` carries input/output names, and SSA
values carry original tensor identities. Shapes, broadcasting, matrix dimensions,
finite coefficients/weights and identity uniqueness are checked. Bounds are rank
64, 64 MiB per tensor and 64 MiB emitted source text. Other operations/types,
dynamic shapes and unknown operation attributes are rejected.

- `--npuforge-verify` checks the closed graph/public SSA contract.
- `--npuforge-optimize` runs greedy C++ rewrite patterns and MLIR dead pure-operation
  elimination. It fuses MatMul/Add/Relu and Gemm/Relu only with single-use,
  non-expanding epilogues. A returned/shared intermediate blocks fusion. The final
  Relu's ID is retained, preventing new identity collisions; provenance records the
  replaced IDs. `--no-opt` bypasses this pass, not verification or lowering.
- `--npuforge-export-graph='output=PATH'` writes the graph consumed by the NPU backend.
  The bridge uses a versioned transport schema and exact little-endian FP32 hex
  values, including signed zero. Python revalidates the graph, I/O shapes and all
  surviving initializer bytes before code generation.
- `--convert-npuforge-to-linalg` uses full dialect conversion. Matrix products become
  zero-filled `linalg.matmul` or transpose-aware reduction `linalg.generic`; Add,
  broadcasting, scaling, bias and Relu become parallel generics with Arith scalar
  operations. Constants become `arith.constant`. No custom tensor operation may
  remain after successful conversion.

`--emit-llvm` runs upstream one-shot bufferization, Linalg-to-loops, Affine/SCF/Arith/
MemRef/Func conversions, cast reconciliation, LLVM translation and `opt` verification.
It adds C ABI wrappers for independently executing the LLVM result. There is no
MLIR-to-native-NPU timing shortcut and no claim that MLIR alone improves latency.

## Installation

First install the core package and development build tools:

```bash
python -m pip install -e '.[dev]'
```

macOS:

```bash
brew install llvm@19
python scripts/build_mlir.py --llvm-prefix "$(brew --prefix llvm@19)"
export NPUFORGE_LLVM_BIN="$(brew --prefix llvm@19)/bin"
```

Debian Bookworm or Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install build-essential libmlir-19-dev mlir-19-tools llvm-19-dev \
  clang-19 zlib1g-dev libzstd-dev
python scripts/build_mlir.py --llvm-prefix /usr/lib/llvm-19
export NPUFORGE_LLVM_BIN=/usr/lib/llvm-19/bin
```

Both platforms:

```bash
export NPUFORGE_MLIR_OPT="$PWD/build/mlir/npuforge-opt"
npuforge demo --compiler mlir --output output/mlir-demo
python examples/compare_mlir.py
npuforge compile output/mlir-demo/models/demo_mlp.onnx --compiler mlir \
  --emit-llvm --llvm-bin "$NPUFORGE_LLVM_BIN" --output output/mlir-llvm
python -m pytest -q
```

The builder selects the current macOS deployment target to match the locally
installed toolchain. CMake rejects other LLVM major versions. The core package
build does not download, link or require MLIR. Set `NPUFORGE_MLIR_OPT`, put
`npuforge-opt` on PATH, or use `--mlir-opt PATH`; missing tools fail with instructions
instead of silently selecting the Python compiler. Upstream LLVM emission tools
must also be version 19. `cmake --install build/mlir --prefix PATH` installs the
optional tool separately.

For direct pass experimentation:

```bash
build/mlir/npuforge-opt output/mlir-demo/mlp_unoptimized/original.mlir \
  --npuforge-optimize --npuforge-export-graph='output=output/graph.json' \
  -o output/optimized.mlir
build/mlir/npuforge-opt output/optimized.mlir \
  --convert-npuforge-to-linalg -o output/linalg.mlir
"$NPUFORGE_LLVM_BIN/mlir-opt" output/linalg.mlir -o /dev/null
```

## Verification and reproducibility

Tests cover optimization enabled/disabled, both NPU mappings, constrained SRAM,
byte-preserving initializers, Unicode names, public/shared intermediate safeguards,
idempotence, invalid types/attributes/graphs, scalar and multidirectional broadcasts,
Gemm transposes/scaling/bias and bounded deep SSA graphs. Binary loading also runs
after deletion of the ONNX file. Python's optimizer is replaced with a failing stub
in integration tests to prove the MLIR path performs its own transformations.

LLVM tests compile actual `model.ll` with Clang, bind the generated memref C ABI in
a separate bounded process, extract output tensors and compare them with ONNX
Runtime. Process exit reclaims bufferized temporaries; this harness is a correctness
check, not a production CPU runtime or performance benchmark. The standalone NPU
runtime remains the execution interface for users.

Without the optional tools, native MLIR tests are explicitly skipped. With the two
environment variables above, the complete MLIR/LLVM suite runs; configured missing
or wrong-version tools fail. The dedicated CI job builds this path separately.
See [validation evidence](validation.md) and the [compiler comparison](../benchmarks/mlir_results.md).

Implementation references: LLVM's [dialect tutorial](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/),
[pattern rewriting](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/) and
[dialect conversion](https://mlir.llvm.org/docs/DialectConversion/).
LLVM/MLIR is separately distributed under Apache-2.0 with LLVM exceptions; the
NPUForge source retains its MIT license.

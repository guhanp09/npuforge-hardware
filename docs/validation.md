# Validation

The executable backend's v0.4 suite has **245 Python tests** and two native CTest
suites. The MLIR extension brings the complete configuration to **291 tests**.
The [GitHub Actions workflow](https://github.com/guhanp09/npuforge-hardware/actions/workflows/ci.yml)
builds the core package on macOS/Linux and the optional MLIR compiler on Linux.
The evidence below records local macOS builds and clean Linux Docker runs; hosted
results and downloadable reports are available from the workflow page.

## MLIR extension (v0.5)

The complete optional-tool configuration passes **291 Python tests** on macOS
arm64/Python 3.11 and 3.12, and Debian Bookworm x86_64/Python 3.12 in Docker. This includes the original 245 regressions and 46 new MLIR tests.
All 46 MLIR tests also pass with the native tool built under AddressSanitizer and
UndefinedBehaviorSanitizer. LLVM/MLIR 19.1.7 is pinned for these runs.

New coverage includes registered-dialect parsing and verifiers, genuine SSA
fusion/dead-code rewrites, public-value safety, initializer bits, Unicode names,
idempotence, invalid IR, constrained SRAM, scalar/multidirectional broadcasts,
Gemm transposes/epilogues, deep SSA graphs and compiler comparison reports.
Generated LLVM IR is compiled with Clang and executed in separate bounded test
processes; its actual outputs are checked against ONNX Runtime.

The clean Linux run also passed formatting, the MLIR demo, LLVM emission,
standalone binary execution and source/wheel builds. All eight compiler comparison
metric dictionaries matched macOS exactly. The installed macOS wheel compiled a
model through MLIR/LLVM outside the source directory, then ran its binary after
removing the ONNX/MLIR files and disabling tool discovery.

Without optional tools, the core configuration passes 247 tests and explicitly
skips the 44 native-tool cases. Both native core CTest suites still pass in Release
and under ASan/UBSan. The separate MLIR CI job builds the optional compiler and exercises its full test
suite. The v0.4 evidence below remains the baseline release record.

Reproduce the full optional Linux check:

```bash
docker run --rm --platform linux/amd64 --cpus=2 --memory=3g \
  --mount "type=bind,src=$PWD,dst=/source,readonly" \
  python:3.12-slim-bookworm sh /source/scripts/validate_mlir_linux.sh
```

## Tested configurations

| Platform | Python | Checks |
| :--- | :--- | :--- |
| macOS arm64 | 3.12 | 245 tests, both native CTests, Ruff, clang-format, examples, source/wheel builds |
| macOS arm64 | 3.11 | Package installation, 245 tests and dependency checks |
| Linux x86_64, Debian Bookworm container | 3.12 | Clean installation, 245 tests, both native CTests, formatting, examples and packaging |

The v0.4 wheel was installed in a separate environment and exercised outside the
source directory with only the five compiled files and input NPZ. No ONNX model
was present; output arrays exactly matched the compiler run.
Dependency checks passed. The native tests passed in a Debug build with
AddressSanitizer and UndefinedBehaviorSanitizer after the final interpreter fix.

## Coverage

- ONNX import, unsupported operators, malformed files, static shapes and FP32 types.
- IR producer/consumer invariants, ordering, cycles and deterministic JSON round trips.
- Dead computation elimination and fusion safety for shared or public intermediates.
- MatMul and Gemm tile coverage, K accumulation, transpose gathers, broadcast bias
  and SRAM capacity. Instruction replay reconstructs outputs from emitted tile metadata.
- Gemm defaults, finite coefficients, optional C, transpose flags and final-tile activation.
- Native dependency scheduling, compute/DMA overlap, resource contention, allocation
  lifetimes, working-set bounds, invalid frees and allocation deadlock.
- Binary header, opcodes, lengths, reserved fields, endianness, version, integer
  boundaries, dependency cycles, deterministic round trips and artifact checksums.
- Aligned DRAM lifetimes/reuse, SRAM partition/address consistency, capacity limits,
  uninitialized/stale reads, DMA readiness, staged operand provenance and incomplete
  reductions. Modified images are tested through the C++ interpreter.
- Exact public-output contract verification, including rejection of missing outputs.
- Standalone loading with the original ONNX file deleted, changed runtime inputs,
  output aliases, constrained 128-byte scratchpads, random graphs and a 256-op chain.
- CLI errors, report metrics, HTML escaping, SVG structure and repeatable benchmark output.

Decoded native outputs are compared with ONNX Runtime CPU using `rtol=1e-4, atol=1e-5`.
Tests use fixed random seeds. Non-finite initializers and outputs are rejected.

The C++ tests use hand-calculated cases for single transfers and computations,
sequential dependencies, independent resource overlap, compute contention and the
shared PIM/DMA memory port. The binary suite independently hand-encodes scalar
conventional/PIM programs, checks actual 2×3=6 output, and rejects malformed images
and illegal allocations. Python integration tests check Gemm timing with a
`(2,3) @ (3,4)` workload, alpha=2, beta=-1 and vector bias:

| Configuration | Conventional cycles | PIM cycles |
| :--- | ---: | ---: |
| Gemm | 41 | 38 |
| Gemm followed by Relu | 59 | 56 |
| Fused Gemm/Relu | 43 | 40 |

A cancellation test catches activation applied before the final reduction. Other
regressions cover colliding fusion IDs, published intermediate values and multiple
consumers of the activated output.

## Reproducing Linux validation

From the repository root, with Docker running:

```bash
docker run --rm --platform linux/amd64 --cpus=2 --memory=2g \
  --mount "type=bind,src=$PWD,dst=/source,readonly" \
  python:3.12-slim-bookworm sh /source/scripts/validate_linux.sh
```

The x86_64 container was run under Docker on Apple Silicon. The script copies the
source into the container, installs the package, runs the
checks and compares all thirteen native metric dictionaries with the reference JSON in
`benchmarks/`. The macOS and Linux results matched exactly for those workloads.
This covers the original six cases, four fusion cases and three allocation/schedule
cases. FP32 verification errors may vary slightly with floating-point evaluation order.

Source archives are checked for required source/test files and exclude bytecode,
virtual environments, generated reports and native build output. The timing model's
assumptions and limitations are described in [architecture.md](architecture.md).

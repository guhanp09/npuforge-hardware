# Compiler validation status — v0.5.0

The addressed binary backend and optional MLIR integration are complete.
The existing ONNX frontend, graph IR, allocator, scheduler, binary codec and native
interpreter remain the NPU execution path.

MLIR components: registered ODS FP32 tensor dialect; shape/type/identity verifiers;
SSA fusion and dead-code rewrites; byte-preserving graph bridge consumed by the
existing backend; full Linalg/Arith/Tensor conversion; optional upstream bufferization,
LLVM conversion/translation and IR verification; CLI options, reports and comparison.
Generated LLVM CPU code is executed and compared with ONNX Runtime in tests.

Verified: 291 tests on macOS arm64/Python 3.11 and 3.12 and Linux x86_64/Python 3.12;
all 46 MLIR tests with an ASan/UBSan native tool; both original CTest suites in
Release and under sanitizers; formatting; clean source/wheel builds; dependency
checks; installed-wheel compilation and binary execution outside the repository.
Core-only installation passes 247 tests and explicitly skips 44 optional-tool cases.

Latest commands:
- `python scripts/build_mlir.py --llvm-prefix /path/to/llvm19`
- `npuforge demo --compiler mlir --mlir-opt build/mlir/npuforge-opt --output output/mlir-demo`
- `npuforge compile output/mlir-demo/models/demo_mlp.onnx --compiler mlir --mlir-opt build/mlir/npuforge-opt --emit-llvm --llvm-bin /path/to/llvm19/bin --output output/mlir-llvm`
- `python examples/compare_mlir.py --mlir-opt build/mlir/npuforge-opt`

All eight compiler comparison metric sets match between macOS and Linux. Both
compiler paths take 1,169/945 unoptimized/optimized conventional cycles and
1,022/798 PIM cycles. No additional latency improvement is attributed to MLIR.
The existing allocation comparison remains 10,624/8,704/10,752 DRAM bytes for
baseline/memory-aware/no-reuse, each at 1,112 simulated cycles.

No required implementation or verification work is pending. No known critical
correctness issue remains. Scope: static FP32 subset, single tile workspace,
hypothetical NPU timing and LLVM/MLIR 19. Hosted CI for v0.5 awaits publication.

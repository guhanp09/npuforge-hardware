# Backend validation status — v0.4.0

The existing ONNX frontend, typed IR, optimizations, tiling and native resource
scheduler are retained. The original 182-test regression suite remains passing.

Completed: aligned DRAM allocation and lifetime reuse; concrete SRAM partitions;
baseline/memory-aware ordering; addressed instruction IR; versioned binary codec;
real FP32 weight image; checked manifest/map/schedule loader; native instruction
arithmetic and timing; standalone run/verify/inspect-program; binary-first demo;
SRAM lifetime/report views; allocation benchmarks and focused format/design docs.

Verified on macOS arm64: 245 Python tests on Python 3.11 and 3.12; both Release
CTest suites; both suites under ASan/UBSan; Ruff and clang-format; source/wheel
builds; dependency checks; installed-wheel execution outside the repository without
an ONNX model. Compiled outputs are compared with ONNX Runtime at rtol=1e-4,
atol=1e-5. Partial-reduction stores and incomplete public-output verification have regressions.

Latest commands: `npuforge demo`; `npuforge inspect-program output/mlp_optimized`;
`npuforge run output/mlp_optimized --inputs output/mlp_optimized/inputs.npz`;
`npuforge verify output/models/demo_mlp.onnx --compiled output/mlp_optimized`;
`npuforge benchmark --compare-schedulers --output output/schedulers`.

Allocation benchmark: baseline 10,624 B, memory-aware 8,704 B, memory-aware without
reuse 10,752 B. All take 1,112 simulated cycles and move 16,640 off-chip bytes.

Linux x86_64 validation in a clean Debian/Python 3.12 container passed all 245
Python tests, both native suites, formatting, demos, standalone binary execution,
verification, scheduling comparison and source/wheel builds. All 13 benchmark
metric dictionaries matched macOS exactly. Source/wheel archive checks passed.

Stage A is complete. No known critical correctness defect remains. No required
implementation or verification work is pending.
Limits: static FP32 subset, conservative single tile workspace, hypothetical timing
model, 64 MiB per data space. Advanced features have not been implemented.

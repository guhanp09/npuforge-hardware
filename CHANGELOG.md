# Changelog

## 0.5.0

- Add an optional LLVM/MLIR 19 tool with a registered FP32 tensor dialect and verifiers.
- Implement native SSA fusion and dead-code rewrites, preserving observable values.
- Bridge the transformed MLIR graph to the existing addressed binary backend.
- Lower every supported operation to Linalg/Arith/Tensor and optionally verified LLVM IR.
- Execute generated LLVM code in differential tests; compare both compiler paths on
  the same conventional/PIM workloads and expose MLIR artifacts in HTML reports.

## 0.4.0

- Emit versioned executable instruction and FP32 weight images with checked manifests.
- Allocate aligned DRAM tensors from lifetimes, reuse dead storage, and map tile SRAM.
- Compare baseline and memory-aware operation schedules using the native resource model.
- Decode and execute addressed instructions in C++, validating ownership, operand
  provenance, reduction order and transfer-completion hazards.
- Add standalone run, verify and inspect-program commands, memory/report views,
  scheduling benchmarks and malformed-image/native execution tests.

## 0.3.0

- Fuse Gemm followed by Relu when the intermediate result is neither shared nor public.
- Keep activation in the final matrix tile and include its vector work in simulation.
- Add a four-case conventional/PIM comparison with instruction traces and an HTML report.
- Avoid operation ID collisions during fusion.

## 0.2.0

- Support static FP32 Gemm with transpose flags, alpha/beta scaling and optional bias.
- Lower transposed matrix tiles to DMA gathers or memory-local PIM reads.
- Add numerical, tiling and timing tests for Gemm and a dense-layer example.

## 0.1.0

- Import static FP32 MatMul, Add and Relu graphs from ONNX.
- Add tensor IR, dead computation elimination and MatMul/Add/Relu fusion.
- Tile matrix operations to fit SRAM and emit explicit DMA and allocation instructions.
- Schedule instructions in a C++ resource simulator with conventional and PIM mappings.
- Add ONNX Runtime verification, benchmarks, reports and macOS/Linux CI.

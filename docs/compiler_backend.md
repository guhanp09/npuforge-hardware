# Compiled-program backend

The existing ONNX importer, typed tensor IR, dead computation elimination, fusion,
tiler and native resource scheduler form one pipeline:

```
ONNX → graph IR → optimize → order operators → tile → allocate DRAM/SRAM
     → addressed instruction DAG → binary images → loader → native FP32 execution
     → ONNX Runtime comparison → metrics and HTML report
```

`compile_model` writes its compiled images to disk, reloads them through the public
loader and verifies native outputs before returning a successful run. It retains
those bytes for export. No graph executor supplies the compiled outputs. The
standalone loader has no ONNX dependency in its execution path; ONNX Runtime is
called separately for reference verification. Python binds inputs and reconstructs
output arrays; C++ performs tile arithmetic.

## Artifact contract

| File | Purpose |
| :--- | :--- |
| `instructions.bin` | Versioned fixed-width instructions and dependency pool |
| `weights.bin` | Actual C-order, little-endian FP32 initializer values and zero padding |
| `manifest.json` | Version, architecture, hardware, entry point, I/O names, image sizes and SHA-256 |
| `memory_map.json` | Tensor shapes, types, roles, weight offsets, DRAM addresses/lifetimes, SRAM partitions and instruction storage |
| `schedule.json` | Addressed instructions, node identities, dependencies, resources and native planned cycles |

The manifest identifies the four other files by fixed names and checksums. Tensor
metadata lives in the referenced memory map, avoiding a duplicate source of truth.
Weight file offsets equal the corresponding allocated DRAM addresses. Runtime
inputs are bound by original graph name, exact shape and FP32 type. The compiler
exports example `inputs.npz` and actual `outputs.npz` separately; neither is needed
to decode the program. Old graph IR, readable instructions and trace JSON remain
available for inspection, but the runtime does not use them to execute arithmetic.

## Native execution

The sole binary decoder is implemented in C++. Python's encoder is checked against
its decoded representation. Loading validates image hashes, shapes, live allocation
ranges, schedule/binary agreement, resource annotations, planned timings, lifetimes
and SRAM partitions. The C++ interpreter shares the original deterministic timing
scheduler, then executes its trace in issue order. A read samples at instruction
start; a write becomes available at completion. Per-word ownership and timestamps
reject uninitialized reads, stale aliases, outstanding-transfer hazards and early
reuse. Staged operands also retain source tensor/index provenance, and matrix
accumulators track consecutive K slices. Output stores must match their producer.

Matrix instructions perform FP32 multiply/accumulate, final Gemm scaling/bias and
optional Relu. DMA gathers implement transpose/broadcast addressing and output
stores scatter boundary tiles. PIM reads B directly from its allocated DRAM tensor;
A, bias and the accumulator use SRAM. Timing still follows the documented abstract
rates, without host execution or initialization overhead.

The loader bounds DRAM and SRAM to 64 MiB each, instruction/weight files to 256 MiB,
metadata files to 32 MiB, the manifest to 16 MiB, tensor count to 16,384 and rank to
64. Shapes must be positive and static; scalars have shape `[]`. Binary decoding
accepts at most one million instructions. These are implementation limits, not
claims about a physical device. Hashes detect artifact corruption; they do not
provide authentication.

See [instruction format](instruction_format.md), [memory planning](memory_planning.md)
and [scheduling](scheduling.md). The supported ONNX contract and timing formulas are
specified in [architecture.md](architecture.md).

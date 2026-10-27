# Instruction image, version 1

All integers use unsigned little-endian encoding; arithmetic values are IEEE-754
binary32. Addresses are 64-bit byte offsets into separate DRAM, SRAM or instruction
spaces, never host pointers. The image starts with this 32-byte header:

| Offset | Bytes | Value |
| ---: | ---: | :--- |
| 0 | 8 | Magic `NPUFIMG\0` |
| 8 | 2 | Version = 1 |
| 10 | 2 | Record bytes = 256 |
| 12 | 4 | Precision = 1 (FP32) |
| 16 | 8 | Instruction count |
| 24 | 8 | Dependency-pool entry count |

Exactly `count` records follow, then a packed pool of little-endian uint32
instruction IDs. Record and dependency addresses are 4-byte aligned. File length
must match exactly; trailing bytes, unknown versions, nonzero reserved bytes,
invalid fields and cyclic dependencies fail decoding. Entry point is instruction 0;
empty images are legal for programs whose output aliases an input or initializer.

## Record layout

| Offset | Type / count | Field |
| ---: | :--- | :--- |
| 0 | uint32 | Opcode |
| 4 | uint32 | Dense instruction ID (storage ordinal) |
| 8 | uint32 | Flags |
| 12 | uint32 | DMA addressing mode |
| 16 | uint32 | Workspace buffer identity (0..INT32_MAX) |
| 20 | uint32 | First dependency-pool entry |
| 24 | uint32 | Dependency count |
| 28 | uint32 | Precision = 1 |
| 32 | uint32 × 4 | Tensor identities A, B, C, output; `0xffffffff` means absent |
| 48 | uint64 × 4 | Byte addresses A, B, C, output |
| 80 | uint64 × 8 | SRAM reservation, working set, transfer bytes, MACs, scalar elements, local read bytes, flat start, flat count |
| 144 | uint32 × 6 | Tile origin m0,n0,k0 and extents m,n,k |
| 168 | float32 × 2 | Gemm alpha, beta (finite) |
| 176 | byte × 80 | Reserved, zero |

Tensor identities index the memory map's ordered `tensors` array. Dependencies may
reference forward IDs but must be distinct, valid and acyclic. All data addresses
are 4-byte aligned; allocated DRAM tensor bases are additionally 64-byte aligned.
The decoder checks encoding structure. The loader checks metadata consistency; the
interpreter validates opcode-specific accesses, shapes and arithmetic counters.

## Opcodes and operands

| Value | Opcode | Address spaces / resources |
| ---: | :--- | :--- |
| 1 | ALLOC | output = SRAM arena base; reservation bytes; zero cycles |
| 2 | FREE | buffer identity; zero cycles |
| 3 | DMA_LOAD | A = DRAM tensor base, output = SRAM staging; DMA + DRAM |
| 4 | DMA_STORE | A = SRAM accumulator, output = DRAM tensor base; DMA + DRAM |
| 5 | MATMUL | A/B/C/output in SRAM; compute |
| 6 | ADD | A/B/output in SRAM; compute |
| 7 | RELU | A/output in SRAM; compute |
| 8 | FUSED_MATMUL_ADD_RELU | SRAM operands, final bias/activation; compute |
| 9 | PIM_MATMUL | A/C/output SRAM, B DRAM; PIM + DRAM |

Flags: bit 0 accumulates an existing K prefix; bit 1 executes the final epilogue;
bit 2 includes Relu; bit 3 denotes Gemm; bits 4/5 transpose A/B. Other bits are
invalid. Accumulation is required exactly when k0 is nonzero; epilogue instructions
must mark the final K slice. Plain MatMul needs no epilogue flag. Stores reject an
incomplete matrix reduction. Alpha/beta apply only in the final Gemm epilogue.

DMA modes: 0 gathers matrix A, 1 gathers B, 2 gathers broadcast bias, 3 gathers a
flat broadcast chunk, 4 stores a matrix tile, 5 stores a flat chunk. Matrix loads
use tile origin/extents plus tensor shape; flat loads/stores use flat start/count.
Loads gather into contiguous staging, expanding broadcasts locally. Transfer-byte
counters count unique source FP32 words, not expanded staging size. Instructions
retain tensor identities even when compute addresses point to SRAM, allowing
validation of staged data provenance. JSON-only node/resource annotations are
reconstructed or checked by the loader, not executable payload hidden in the image.

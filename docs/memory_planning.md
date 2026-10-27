# Memory planning

DRAM holds weights, runtime inputs, intermediate results and public outputs. SRAM
is an explicitly allocated tile scratchpad; instruction storage is a separate
read-only space starting at byte zero. All tensors are C-order FP32. DRAM tensor
bases and padded allocation sizes use 64-byte alignment. SRAM partitions use
4-byte alignment. The default SRAM is 65,536 bytes; each data space is bounded to
64 MiB by the interpreter. `compile --dram-limit BYTES` can impose a smaller DRAM
limit. Impossible placements fail before execution.

## DRAM lifetimes and reuse

The selected topological operator order defines each tensor's first producer and
last consumer. Inputs/weights begin at entry and are pinned through program end;
public outputs also remain live until extraction. An intermediate stays live for
the complete last consuming operation, including all DMA and compute instructions.
The lowerer serializes operations and tile workspaces with explicit dependencies,
so operator intervals safely bound outstanding transfers.

A deterministic linear scan allocates constants first (file offset = DRAM address),
then tensors by first use and name. It releases only intervals with `last < first`
and chooses the first address-sorted hole large enough for the aligned request.
It splits larger holes but does not coalesce adjacent holes. Otherwise it appends
to the address high-water mark. No output overwrites an input used by its own
operator. `--no-reuse` allocates every tensor separately for comparison.

The map records logical and padded sizes, roles, byte addresses, operator/cycle
lifetimes and `reused_from`. `reused_bytes` sums requests served from free holes;
`dram_bytes` is the actual address-space high-water mark. `unreused_bytes` sums
aligned tensor sizes. Allocation efficiency is `dram_bytes / unreused_bytes`:
a lower ratio means less storage, not greater compute utilization.

## SRAM tiling

Each tile receives one arena at SRAM address zero, released before the next arena.
The map lists ALLOC/FREE instruction IDs, actual native cycle boundaries and
non-overlapping A/B/accumulator/optional-bias partitions. Conventional matrix tiles
reserve `4*(m*k + k*n + m*n + optional m*n)` bytes. PIM omits B staging and reads
its weights from DRAM. Vector chunks reserve one expanded staging region per
operand and one result region. The tiler shrinks requested extents to fit capacity;
K chunks reuse staging and preserve the accumulator. Boundary tiles use actual
sizes. There is no persistent cache or advanced double buffering.

The loader checks map/binary/schedule consistency, aligned ranges, live overlap,
partition coverage and executable operand addresses. Native execution tracks
per-word ownership, initialization, source provenance and read/write completion.
Thus a stale tensor at a reused address cannot silently satisfy a later read.
These checks cover both compiler output and modified images; checksums alone
cannot establish a legal allocation.

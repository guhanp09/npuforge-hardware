# Scheduling

There are two compiler operation orders and one shared native resource scheduler.
Both orders preserve dependencies and use identical tiling, hardware rates and
FP32 semantics. `compile --schedule baseline|memory-aware` selects the order;
`benchmark [MODEL] --compare-schedulers` compares baseline, memory-aware and
memory-aware without DRAM reuse.

Baseline retains the IR's stable topological order. Memory-aware chooses among
ready operations by the smallest aligned output bytes minus last-use, non-pinned
input bytes. Ties use the original ordinal. This favors finishing branches and
releasing temporary storage. Public inputs, outputs and weights are pinned and
receive no release credit. It is a simple storage heuristic, not a global optimum.

The existing lowerer emits transfer dependencies, tile barriers and ALLOC/FREE
lifetimes. The same C++ event scheduler assigns planned cycles and runtime timing:
compute, DMA and PIM each have one exclusive engine; DMA and PIM share the DRAM
port. It issues ready IDs deterministically when resources and SRAM are available.
The compiler reuses a single workspace, so generated tiles and operations remain
conservative and largely serialized. No second timing estimator invents timestamps.

`schedule.json` contains the actual addressed records, dependencies, resource and
space annotations, and native start/end cycles. Loading re-simulates the binary
and rejects mismatching timestamps. Execution checks read/write readiness against
those cycles and compares its trace with the plan. Total cycles are the last
completion time. `resource_idle_cycles` is makespan minus busy cycles per resource;
it includes idle periods from all causes and is not a separate stall taxonomy.

On the checked-in two-branch workload, memory-aware order reduces allocated DRAM
from 10,624 to 8,704 bytes (18.1%). Both take 1,112 simulated cycles with identical
traffic. Disabling reuse requires 10,752 bytes. The result demonstrates storage
savings; no latency improvement is claimed. See [benchmark details](../benchmarks/README.md)
for the workload, assumptions and generated reference data.

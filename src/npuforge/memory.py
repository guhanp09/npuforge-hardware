"""Topological operator scheduling and aligned interval allocation."""

from __future__ import annotations

from collections import Counter

from .ir import Graph, Operation

MAX_MEMORY = 64 * 1024 * 1024
ALIGNMENT = 64


def align(value: int, alignment: int = ALIGNMENT) -> int:
    if type(value) is not int or value < 0 or alignment <= 0:
        raise ValueError("invalid allocation size or alignment")
    return (value + alignment - 1) // alignment * alignment


def schedule_operations(graph: Graph, strategy: str = "memory-aware") -> list[Operation]:
    graph.validate()
    ordered = graph.ordered()
    if strategy == "baseline":
        return ordered
    if strategy != "memory-aware":
        raise ValueError("scheduler must be baseline or memory-aware")
    available = set(graph.inputs) | {k for k, t in graph.tensors.items() if t.data is not None}
    pinned = available | set(graph.outputs)
    uses = Counter(t for o in ordered for t in set(o.inputs))
    positions = {o.id: i for i, o in enumerate(ordered)}
    pending, result = list(ordered), []
    while pending:
        ready = [o for o in pending if set(o.inputs) <= available]
        if not ready:
            raise ValueError("unschedulable graph")

        def cost(op):
            freed = sum(
                align(graph.tensors[t].nbytes)
                for t in set(op.inputs)
                if uses[t] == 1 and t not in pinned
            )
            return align(graph.tensors[op.output].nbytes) - freed, positions[op.id]

        op = min(ready, key=cost)
        result.append(op)
        pending.remove(op)
        available.add(op.output)
        uses.subtract(set(op.inputs))
    return result


def plan_memory(
    graph: Graph, order: list[Operation], *, reuse: bool = True, dram_limit: int = MAX_MEMORY
) -> dict:
    graph.validate()
    actual = {o.id: o for o in graph.operations}
    available = set(graph.inputs) | {k for k, t in graph.tensors.items() if t.data is not None}
    if len(order) != len(actual) or {o.id for o in order} != set(actual):
        raise ValueError("invalid allocation operator order")
    for op in order:
        if op != actual[op.id] or not set(op.inputs) <= available:
            raise ValueError("allocation order violates graph dependencies/semantics")
        available.add(op.output)
    if type(dram_limit) is not int or not 0 < dram_limit <= MAX_MEMORY:
        raise ValueError("DRAM limit must be between 1 and 64 MiB")
    positions = {o.id: i for i, o in enumerate(order)}
    producers, consumers = graph.producers(), graph.consumers()
    steps = len(order)
    descriptors = []
    for name, t in sorted(graph.tensors.items()):
        roles = []
        if t.data is not None:
            roles.append("weight")
        if name in graph.inputs:
            roles.append("input")
        if name in graph.outputs:
            roles.append("output")
        if not roles:
            roles.append("intermediate")
        first = positions[producers[name].id] if name in producers else 0
        last = max([first, *[positions[o.id] for o in consumers[name]]])
        if set(roles) & {"weight", "input", "output"}:
            last = steps
        size = t.nbytes
        if size <= 0 or size > dram_limit:
            raise ValueError(f"{name}: tensor cannot fit in DRAM")
        descriptors.append(
            {
                "id": name,
                "shape": list(t.shape),
                "dtype": t.dtype,
                "space": "dram",
                "size": size,
                "alignment": ALIGNMENT,
                "allocation_size": align(size),
                "roles": roles,
                "role_bits": (1 if "weight" in roles else 0)
                | (2 if "input" in roles else 0)
                | (4 if "output" in roles else 0),
                "first": first,
                "last": last,
            }
        )
    # Put the weight image first. Its file offsets are also its DRAM offsets.
    descriptors.sort(key=lambda t: ("weight" not in t["roles"], t["first"], t["id"]))
    active, holes, peak, reused = [], [], 0, 0
    for t in descriptors:
        for old in active[:]:
            if old["last"] < t["first"]:
                active.remove(old)
                holes.append((old["address"], old["allocation_size"], old["id"]))
        holes.sort()
        choice = next((h for h in holes if h[1] >= t["allocation_size"]), None) if reuse else None
        if choice:
            holes.remove(choice)
            address, length, previous = choice
            t["reused_from"] = previous
            reused += t["allocation_size"]
            if length > t["allocation_size"]:
                holes.append(
                    (address + t["allocation_size"], length - t["allocation_size"], previous)
                )
        else:
            address = peak
            peak += t["allocation_size"]
            t["reused_from"] = None
        t["address"] = address
        t["weight_offset"] = address if "weight" in t["roles"] else None
        if peak > dram_limit:
            raise ValueError(
                "DRAM capacity exceeded; use memory-aware scheduling or a larger limit"
            )
        active.append(t)
    weights_size = max(
        (t["address"] + t["allocation_size"] for t in descriptors if "weight" in t["roles"]),
        default=0,
    )
    return {
        "schema": "npuforge.memory.v1",
        "alignment": ALIGNMENT,
        "dram_bytes": peak,
        "dram_limit": dram_limit,
        "weights_bytes": weights_size,
        "reuse": reuse,
        "reused_bytes": reused,
        "unreused_bytes": sum(t["allocation_size"] for t in descriptors),
        "operator_order": [o.id for o in order],
        "tensors": descriptors,
        "scratchpad": [],
    }

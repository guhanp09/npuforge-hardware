"""Compiled artifact, concrete allocation and native interpreter regression coverage."""

import copy
import json
import struct
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
from onnx import helper

from npuforge import _native
from npuforge.backend import Hardware
from npuforge.codegen import HEADER, RECORD, blank, decode, encode
from npuforge.functional import verify_outputs
from npuforge.image import digest, json_bytes, load_program
from npuforge.memory import plan_memory, schedule_operations
from npuforge.models import generate_model
from npuforge.pipeline import compile_model


@pytest.mark.parametrize("kind", ["matmul", "mlp", "gemm", "gemm_relu"])
@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("optimized", [False, True])
def test_binary_pipeline_without_original_model(
    tmp_path, kind, architecture, optimized, monkeypatch
):
    model = generate_model(tmp_path / "m.onnx", kind, 5, 7, 9)
    run = compile_model(model, Hardware(architecture=architecture, sram_bytes=128), optimized)
    run.write(tmp_path / "compiled")
    loaded = load_program(tmp_path / "compiled")
    outputs, metrics = loaded.run(run.inputs)
    assert verify_outputs(model, run.inputs, outputs)["passed"]
    assert metrics == run.metrics
    model.unlink()
    monkeypatch.setattr("npuforge.functional.execute", lambda *a: pytest.fail("graph bypass"))
    other = {k: np.ones_like(v) for k, v in run.inputs.items()}
    again, _ = loaded.run(other)
    assert set(again) == set(outputs)
    assert all(np.all(np.isfinite(v)) for v in again.values())
    assert (
        encode(decode(run.compiled.files["instructions.bin"]))
        == run.compiled.files["instructions.bin"]
    )
    for t in run.compiled.memory["tensors"]:
        assert t["address"] % t["alignment"] == 0
        if "weight" in t["roles"]:
            raw = run.compiled.files["weights.bin"][t["address"] : t["address"] + t["size"]]
            np.testing.assert_array_equal(
                np.frombuffer(raw, dtype="<f4").reshape(t["shape"]), run.graph.tensors[t["id"]].data
            )


@pytest.mark.parametrize(
    "op,shape", [("Add", ()), ("Add", (2, 1, 3)), ("Relu", ()), ("Relu", (3, 4))]
)
@pytest.mark.parametrize("architecture", ["conventional", "pim"])
def test_binary_vectors_and_repeated_operands(model_factory, op, shape, architecture):
    names = ["x", "x"] if op == "Add" else ["x"]
    model = model_factory([helper.make_node(op, names, ["y"])], {"x": shape}, {"y": shape})
    assert compile_model(model, Hardware(architecture=architecture, sram_bytes=48)).verification[
        "passed"
    ]


def test_binary_multidirectional_broadcast(model_factory):
    model = model_factory(
        [helper.make_node("Add", ["a", "b"], ["out"])],
        {"a": (2, 1, 3), "b": (1, 4, 1)},
        {"out": (2, 4, 3)},
    )
    assert compile_model(model, Hardware(sram_bytes=48)).verification["passed"]


def test_repeated_matrix_input(model_factory):
    model = model_factory(
        [helper.make_node("MatMul", ["x", "x"], ["out"])], {"x": (3, 3)}, {"out": (3, 3)}
    )
    for arch in ("conventional", "pim"):
        assert compile_model(model, Hardware(architecture=arch, sram_bytes=48)).verification[
            "passed"
        ]


@pytest.fixture
def image_run(tmp_path):
    return compile_model(generate_model(tmp_path / "m.onnx", "mlp", 3, 7, 5))


def native(run, records=None, memory=None):
    tensors = (memory or run.compiled.memory)["tensors"]
    inputs = {
        i: run.inputs[t["id"]].astype("<f4").tobytes()
        for i, t in enumerate(tensors)
        if t["id"] in run.inputs
    }
    return _native.execute_image(
        encode(records) if records is not None else run.compiled.files["instructions.bin"],
        run.compiled.files["weights.bin"],
        tensors,
        inputs,
        (memory or run.compiled.memory)["dram_bytes"],
        asdict(run.program.hardware),
    )


def test_fixed_binary_layout_and_boundaries(image_run):
    data = image_run.compiled.files["instructions.bin"]
    assert HEADER.size == 32 and RECORD.size == 256
    assert data[:16] == b"NPUFIMG\x00\x01\x00\x00\x01\x01\x00\x00\x00"
    records = decode(data)
    assert struct.unpack_from("<I", data, 32)[0] == 1
    for field, value in [
        ("a", -1),
        ("a", 2**64),
        ("m", 2**32),
        ("buffer", -1),
        ("alpha", float("nan")),
    ]:
        invalid = copy.deepcopy(records)
        invalid[0][field] = value
        with pytest.raises(ValueError):
            encode(invalid)
    with pytest.raises(ValueError):
        encode([blank() | {"op": "NO_SUCH_OPCODE"}])


@pytest.mark.parametrize("offset", [0, 8, 10, 12, 32, 40, 60, 208])
def test_malformed_binary_fields(image_run, offset):
    data = bytearray(image_run.compiled.files["instructions.bin"])
    data[offset] = 255
    with pytest.raises(ValueError):
        decode(bytes(data))


@pytest.mark.parametrize("length", [0, 7, 31, 32, 100])
def test_truncated_binary(image_run, length):
    with pytest.raises(ValueError):
        decode(image_run.compiled.files["instructions.bin"][:length])


def test_cycles_and_duplicate_dependencies_rejected(image_run):
    records = decode(image_run.compiled.files["instructions.bin"])
    records[0]["deps"] = [1]
    with pytest.raises(ValueError, match="cyclic"):
        decode(encode(records))
    records[0]["deps"] = []
    records[1]["deps"] = [0, 0]
    with pytest.raises(ValueError, match="dependency"):
        decode(encode(records))


@pytest.mark.parametrize(
    "problem",
    [
        "sram_address",
        "dram_address",
        "early_compute",
        "epilogue",
        "overflow",
        "bad_tensor",
        "traffic",
    ],
)
def test_native_instruction_operand_validation(image_run, problem):
    records = decode(image_run.compiled.files["instructions.bin"])
    compute = next(i for i in records if i["op"] in ("MATMUL", "FUSED_MATMUL_ADD_RELU"))
    load = next(i for i in records if i["op"] == "DMA_LOAD")
    if problem == "sram_address":
        compute["out"] = image_run.program.hardware.sram_bytes
    elif problem == "dram_address":
        load["a"] += 64
    elif problem == "early_compute":
        compute["deps"] = []
    elif problem == "epilogue":
        compute["flags"] |= 4
        compute["flags"] &= ~2
    elif problem == "overflow":
        compute["out"] = 2**64 - 4
    elif problem == "bad_tensor":
        load["tensor_a"] = 9999
    else:
        load["bytes"] += 4
    with pytest.raises(ValueError):
        native(image_run, records)


def test_partial_matrix_reduction_cannot_be_stored(tmp_path):
    run = compile_model(generate_model(tmp_path / "m.onnx", "matmul", 1, 3, 1))
    records = decode(run.compiled.files["instructions.bin"])
    compute = next(i for i in records if i["op"] == "MATMUL")
    compute["k"] = 2
    compute["macs"] = 2
    with pytest.raises(ValueError, match="reduction is complete"):
        native(run, records)


def test_verifier_rejects_missing_public_output(tmp_path):
    model = generate_model(tmp_path / "branches.onnx", "branches", 2, 3, 5)
    run = compile_model(model)
    incomplete = dict(run.outputs)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="exactly match.*declared outputs"):
        verify_outputs(model, run.inputs, incomplete)


def rewrite_metadata(directory, name, change):
    path = directory / name
    data = json.loads(path.read_bytes())
    change(data)
    raw = json_bytes(data)
    path.write_bytes(raw)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["images"][name] = digest(raw)
    (directory / "manifest.json").write_bytes(json_bytes(manifest))


def test_checksums_and_memory_schedule_cross_validation(image_run, tmp_path):
    directory = tmp_path / "program"
    image_run.write(directory)
    path = directory / "weights.bin"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="checksum"):
        load_program(directory)
    image_run.write(directory)
    rewrite_metadata(directory, "memory_map.json", lambda d: d["tensors"][0].update(address=4))
    with pytest.raises(ValueError, match="alignment"):
        load_program(directory)
    image_run.write(directory)
    rewrite_metadata(
        directory, "schedule.json", lambda d: d["instructions"][1].update(start=999999)
    )
    with pytest.raises(ValueError, match="cycles"):
        load_program(directory)
    image_run.write(directory)
    rewrite_metadata(directory, "memory_map.json", lambda d: d["scratchpad"][0].update(size=4))
    with pytest.raises(ValueError, match="scratchpad"):
        load_program(directory)


def test_scratchpad_partitions_must_describe_executable_operands(image_run, tmp_path):
    image_run.write(tmp_path)

    def swap_roles(memory):
        parts = memory["scratchpad"][0]["partitions"]
        parts[0]["role"], parts[-1]["role"] = parts[-1]["role"], parts[0]["role"]

    rewrite_metadata(tmp_path, "memory_map.json", swap_roles)
    with pytest.raises(ValueError, match="partition.*operands"):
        load_program(tmp_path)


def test_allocation_reuse_and_scheduler_comparison(tmp_path):
    model = generate_model(tmp_path / "b.onnx", "branches", 8, 8, 64)
    baseline = compile_model(model, scheduler="baseline")
    aware = compile_model(model, scheduler="memory-aware")
    fresh = compile_model(model, scheduler="memory-aware", reuse=False)
    assert baseline.metrics["cycles"] == aware.metrics["cycles"] == fresh.metrics["cycles"] == 1112
    assert baseline.metrics["dram_allocation_bytes"] == 10624
    assert aware.metrics["dram_allocation_bytes"] == 8704
    assert fresh.metrics["dram_allocation_bytes"] == 10752
    assert baseline.compiled.memory["operator_order"] != aware.compiled.memory["operator_order"]
    assert any(t["reused_from"] for t in aware.compiled.memory["tensors"])
    for run in (baseline, aware, fresh):
        tensors = run.compiled.memory["tensors"]
        for index, t in enumerate(tensors):
            for other in tensors[:index]:
                overlap = (
                    t["address"] < other["address"] + other["allocation_size"]
                    and other["address"] < t["address"] + t["allocation_size"]
                )
                assert not overlap or t["last"] < other["first"] or other["last"] < t["first"]
        for tid in run.outputs:
            np.testing.assert_allclose(
                run.outputs[tid], baseline.outputs[tid], rtol=1e-4, atol=1e-5
            )
    with pytest.raises(ValueError, match="DRAM"):
        compile_model(model, dram_limit=64)
    with pytest.raises(ValueError, match="scheduler"):
        schedule_operations(aware.graph, "invalid")
    with pytest.raises(ValueError):
        plan_memory(aware.graph, schedule_operations(aware.graph), dram_limit=2**64)


def test_cli_artifacts_and_standalone_execution(tmp_path):
    model = generate_model(tmp_path / "m.onnx", "mlp", 3, 7, 5)
    directory = tmp_path / "build"

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "npuforge.cli", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
        )

    cli("compile", model, "--target", "pim", "--schedule", "baseline", "--output", directory)
    compiled = directory / "pim"
    for name in (
        "instructions.bin",
        "weights.bin",
        "manifest.json",
        "memory_map.json",
        "schedule.json",
    ):
        assert (compiled / name).is_file()
    cli("inspect-program", directory, "--json")
    assert "true" in cli("verify", model, "--compiled", directory).stdout
    model.unlink()
    cli("run", directory, "--inputs", compiled / "inputs.npz", "--output", tmp_path / "result.npz")
    with np.load(tmp_path / "result.npz") as actual, np.load(compiled / "outputs.npz") as expected:
        np.testing.assert_array_equal(actual["y"], expected["y"])
    cli("benchmark", "--compare-schedulers", "--output", tmp_path / "comparison")
    assert (tmp_path / "comparison" / "report.html").exists()


def test_bounded_deep_and_random_graphs(model_factory):
    rng = np.random.default_rng(120)
    for case in range(5):
        names = ["x"]
        nodes = []
        for index in range(40):
            chosen = [names[int(rng.integers(len(names)))]] if case else [names[-1]]
            op = "Relu"
            if case and rng.random() < 0.5:
                op = "Add"
                chosen *= 2
            name = f"t{index}"
            nodes.append(helper.make_node(op, chosen, [name]))
            names.append(name)
        path = model_factory(
            nodes, {"x": (3, 5)}, {names[-1]: (3, 5), names[20]: (3, 5)}, name=f"deep{case}"
        )
        for scheduler in ("baseline", "memory-aware"):
            run = compile_model(path, Hardware(sram_bytes=64), scheduler=scheduler)
            assert run.verification["passed"]
            assert run.metrics["peak_sram_bytes"] <= 64


def test_staging_alias_cannot_change_logical_operand(tmp_path):
    run = compile_model(generate_model(tmp_path / "mm.onnx", "matmul", 2, 3, 4))
    records = decode(run.compiled.files["instructions.bin"])
    loads = [i for i in records if i["op"] == "DMA_LOAD"]
    loads[1]["out"] = loads[0]["out"]
    compute = next(i for i in records if i["op"] == "MATMUL")
    compute["b"] = compute["a"]
    with pytest.raises(ValueError, match="provenance"):
        native(run, records)


@pytest.mark.parametrize("source", ["input", "weight"])
def test_no_operation_program(model_factory, tmp_path, source):
    model = model_factory(
        [],
        {"x": (2, 3)} if source == "input" else {},
        {"x": (2, 3)},
        {"x": np.arange(6, dtype=np.float32).reshape(2, 3)} if source == "weight" else {},
    )
    run = compile_model(model)
    run.write(tmp_path / "alias")
    outputs, metrics = load_program(tmp_path / "alias").run(run.inputs)
    assert metrics["cycles"] == 0
    assert run.verification["passed"]
    assert outputs["x"].shape == (2, 3)


def test_deep_reuse_chain(model_factory):
    nodes = [
        helper.make_node("Relu", ["x" if i == 0 else f"t{i - 1}"], [f"t{i}"]) for i in range(256)
    ]
    model = model_factory(nodes, {"x": (2, 2)}, {"t255": (2, 2)})
    run = compile_model(model, Hardware(sram_bytes=32), False)
    assert run.metrics["dram_allocation_bytes"] == 192
    assert run.metrics["instruction_count"] == 256 * 5
    assert run.verification["passed"]


def test_allocator_rejects_out_of_order_operations(image_run):
    with pytest.raises(ValueError, match="order"):
        plan_memory(image_run.graph, list(reversed(image_run.graph.ordered())))


@pytest.mark.parametrize(
    "key,value", [("format_version", 99), ("entry_point", 1), ("precision", "int8")]
)
def test_manifest_contract_rejection(image_run, tmp_path, key, value):
    image_run.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest[key] = value
    (tmp_path / "manifest.json").write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):
        load_program(tmp_path)

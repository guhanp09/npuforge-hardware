"""Real MLIR parsing, SSA rewrites, dialect conversion and binary/LLVM differential tests."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from npuforge.backend import Hardware
from npuforge.functional import verify_outputs
from npuforge.image import load_program
from npuforge.importer import import_onnx
from npuforge.ir import Graph, Operation, Tensor
from npuforge.mlir import (
    compile_graph,
    emit_graph,
    find_tool,
    import_bridge,
    invoke,
    lower_to_llvm,
    string,
)
from npuforge.models import generate_model
from npuforge.pipeline import compile_model


@pytest.fixture
def mlir_tool():
    configured = os.environ.get("NPUFORGE_MLIR_OPT")
    candidate = configured or shutil.which("npuforge-opt")
    if candidate is None:
        pytest.skip("optional native MLIR tool is not configured")
    assert Path(candidate).is_file(), "configured MLIR tool does not exist"
    return str(Path(candidate).resolve())


@pytest.fixture
def llvm_bin():
    configured = os.environ.get("NPUFORGE_LLVM_BIN")
    candidate = shutil.which("mlir-opt-19") or shutil.which("mlir-opt")
    if configured is None and candidate is None:
        pytest.skip("optional upstream LLVM/MLIR 19 tools are not configured")
    directory = Path(configured) if configured else Path(candidate).resolve().parent
    assert directory.is_dir()
    assert "version 19." in invoke([directory / "mlir-opt", "--version"])
    return directory


def test_mlir_emitter_escapes_utf8_and_string_delimiters():
    assert string('a"\\\nλ') == '"a\\22\\5C\\0A\\CE\\BB"'


def test_mlir_missing_tool_and_bridge_rejection(monkeypatch):
    monkeypatch.setenv("NPUFORGE_MLIR_OPT", "/does-not-exist/npuforge-opt")
    with pytest.raises(ValueError, match="was not found"):
        find_tool()
    with pytest.raises(ValueError, match="schema"):
        import_bridge('{"schema":"unknown"}')


@pytest.mark.parametrize("kind", ["matmul", "mlp", "gemm", "gemm_relu"])
@pytest.mark.parametrize("architecture", ["conventional", "pim"])
@pytest.mark.parametrize("enabled", [False, True])
def test_mlir_pipeline_executes_binary_and_preserves_metrics(
    tmp_path, mlir_tool, kind, architecture, enabled, monkeypatch
):
    model = generate_model(tmp_path / "m.onnx", kind, 3, 7, 5)
    hardware = Hardware(architecture=architecture, sram_bytes=128)
    baseline = compile_model(model, hardware, enabled)
    monkeypatch.setattr(
        "npuforge.pipeline.optimize", lambda *a: pytest.fail("Python optimizer bypass")
    )
    run = compile_model(model, hardware, enabled, compiler="mlir", mlir_opt=mlir_tool)
    assert run.verification["passed"]
    for key in ["cycles", "offchip_bytes", "peak_sram_bytes", "dram_allocation_bytes"]:
        assert run.metrics[key] == baseline.metrics[key], key
    for name in baseline.outputs:
        np.testing.assert_allclose(run.outputs[name], baseline.outputs[name], rtol=1e-4, atol=1e-5)
    assert run.changes == baseline.changes
    run.write(tmp_path / "compiled")
    assert (tmp_path / "compiled" / "lowered.linalg.mlir").exists()
    assert '"npuforge.' not in run.frontend_artifacts["lowered.linalg.mlir"].decode()
    assert run.compiled.manifest["compiler_frontend"]["name"] == "mlir"
    model.unlink()
    outputs, _ = load_program(tmp_path / "compiled").run(run.inputs)
    for name in outputs:
        np.testing.assert_array_equal(outputs[name], run.outputs[name])


def test_mlir_dce_fusion_idempotence_and_initializer_bits(tmp_path, mlir_tool):
    weights = np.array([[0.0, -0.0], [1e-38, -3.25]], dtype=np.float32)
    graph = Graph(
        {
            name: Tensor(name, (2, 2), data=weights.copy() if name == "w" else None)
            for name in ["x", "w", "mm", "add", "out", "dead"]
        },
        [
            Operation("mm", "MatMul", ["x", "w"], "mm"),
            Operation("add", "Add", ["mm", "w"], "add"),
            Operation("relu", "Relu", ["add"], "out"),
            Operation("dead", "Relu", ["x"], "dead"),
        ],
        ["x"],
        ["out"],
        'graph "λ\\name',
    )
    result = compile_graph(graph, tool=mlir_tool)
    assert result.changes["dead_operations_removed"] == 1
    assert result.changes["fusion_patterns"] == 1
    assert result.graph.tensors["w"].data.tobytes() == weights.tobytes()
    assert result.graph.name == graph.name
    source = tmp_path / "optimized.mlir"
    source.write_bytes(result.artifacts["optimized.mlir"])
    again = tmp_path / "again.mlir"
    invoke([mlir_tool, source, "--npuforge-optimize", "-o", again])
    assert again.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("public", ["mm", "add"])
def test_mlir_public_intermediates_prevent_fusion(tmp_path, mlir_tool, public):
    model = generate_model(tmp_path / "m.onnx", "mlp", 2, 3, 4)
    graph = import_onnx(model)
    name = graph.operations[0 if public == "mm" else 1].output
    graph.outputs.append(name)
    result = compile_graph(graph, tool=mlir_tool)
    assert result.changes["fusion_patterns"] == 0
    assert result.graph.outputs == graph.outputs


@pytest.mark.parametrize(
    "problem",
    ["shape", "dtype", "dynamic", "attribute", "identity", "output", "unknown", "coefficient"],
)
def test_mlir_rejects_invalid_ir(tmp_path, mlir_tool, problem):
    model = generate_model(tmp_path / "m.onnx", "gemm", 2, 3, 4)
    source = emit_graph(import_onnx(model))
    if problem == "shape":
        source = source.replace("tensor<2x4xf32>", "tensor<2x5xf32>")
    elif problem == "dtype":
        source = source.replace("f32", "f64")
    elif problem == "dynamic":
        source = source.replace("tensor<2x3xf32>", "tensor<?x3xf32>")
    elif problem == "attribute":
        source = source.replace('id = "op0_', 'extra = 1 : i64, id = "op0_')
    elif problem == "identity":
        source = source.replace('tensor_id = "y"', 'tensor_id = "x"')
    elif problem == "output":
        source = source.replace(
            'npuforge.output_names = ["y"]', 'npuforge.output_names = ["other"]'
        )
    elif problem == "unknown":
        source = source.replace('"npuforge.gemm"', '"npuforge.unknown"')
    else:
        source = source.replace("alpha = 7.500000000e-01", "alpha = 0x7F800000")
    invalid = tmp_path / "invalid.mlir"
    invalid.write_text(source)
    run = subprocess.run(
        [mlir_tool, str(invalid), "--npuforge-verify", "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0, problem
    assert "error:" in run.stderr
    assert run.returncode > 0, "native tool must reject invalid IR without a crash"


def execute_llvm(run, root, llvm_bin):
    artifacts = lower_to_llvm(run.frontend_artifacts["lowered.linalg.mlir"], llvm_bin)
    ir = root / "model.ll"
    ir.write_bytes(artifacts["model.ll"])
    library = root / ("model.dylib" if sys.platform == "darwin" else "model.so")
    command = [llvm_bin / "clang", "-shared", "-fPIC", "-O0", ir, "-o", library]
    if sys.platform == "darwin":
        sdk = invoke(["xcrun", "--show-sdk-path"]).strip()
        command += ["-isysroot", sdk]
    else:
        command += ["-Wl,-Bsymbolic"]
    invoke(command)
    signature = {
        "inputs": run.graph.inputs,
        "outputs": run.graph.outputs,
        "input_shapes": [run.graph.tensors[n].shape for n in run.graph.inputs],
        "output_shapes": [run.graph.tensors[n].shape for n in run.graph.outputs],
    }
    schema = root / "signature.json"
    schema.write_text(json.dumps(signature))
    inputs, outputs = root / "inputs.npz", root / "outputs.npz"
    np.savez(inputs, **run.inputs)
    worker = Path(__file__).with_name("llvm_cpu_worker.py")
    invoke([sys.executable, worker, library, schema, inputs, outputs])
    with np.load(outputs) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.mark.parametrize("kind", ["matmul", "mlp", "gemm", "gemm_relu", "branches"])
def test_standard_linalg_to_llvm_executes_correctly(tmp_path, mlir_tool, llvm_bin, kind):
    model = generate_model(tmp_path / "m.onnx", kind, 3, 7, 5)
    run = compile_model(model, compiler="mlir", mlir_opt=mlir_tool)
    outputs = execute_llvm(run, tmp_path, llvm_bin)
    assert verify_outputs(model, run.inputs, outputs)["passed"]


@pytest.mark.parametrize(
    "op,shapes,out",
    [("Add", [(2, 1, 3), (1, 4, 1)], (2, 4, 3)), ("Add", [(), ()], ()), ("Relu", [()], ())],
)
def test_mlir_broadcast_and_scalar_llvm_semantics(
    tmp_path, model_factory, mlir_tool, llvm_bin, op, shapes, out
):
    names = [f"x{i}" for i in range(len(shapes))]
    model = model_factory(
        [helper.make_node(op, names, ["out"])], dict(zip(names, shapes, strict=True)), {"out": out}
    )
    run = compile_model(model, compiler="mlir", mlir_opt=mlir_tool)
    assert verify_outputs(model, run.inputs, execute_llvm(run, tmp_path, llvm_bin))["passed"]


def test_mlir_cli_llvm_artifacts_and_report(tmp_path, mlir_tool, llvm_bin):
    model = generate_model(tmp_path / "m.onnx", "mlp", 2, 3, 4)
    invoke(
        [
            sys.executable,
            "-m",
            "npuforge.cli",
            "compile",
            model,
            "--compiler",
            "mlir",
            "--mlir-opt",
            mlir_tool,
            "--emit-llvm",
            "--llvm-bin",
            llvm_bin,
            "--output",
            tmp_path / "compiled",
        ]
    )
    directory = tmp_path / "compiled" / "conventional"
    assert (directory / "model.ll").read_text().find("define") >= 0
    assert (directory / "lowered.llvm.mlir").exists()
    assert load_program(directory).manifest["compiler_frontend"]["llvm_lowering"]


@pytest.mark.parametrize("ta,tb,bias", [(0, 0, None), (0, 1, ()), (1, 0, (3, 1)), (1, 1, (3, 4))])
def test_mlir_transpose_and_gemm_epilogue_llvm_semantics(
    tmp_path, gemm_factory, mlir_tool, llvm_bin, ta, tb, bias
):
    model = gemm_factory(
        attrs={"transA": ta, "transB": tb, "alpha": -0.5, "beta": 2.0},
        bias_shape=bias,
        m=3,
        k=5,
        n=4,
        relu=True,
    )
    run = compile_model(model, compiler="mlir", mlir_opt=mlir_tool)
    assert verify_outputs(model, run.inputs, execute_llvm(run, tmp_path, llvm_bin))["passed"]


@pytest.mark.parametrize("constant", [False, True])
def test_mlir_zero_operation_alias_program(tmp_path, model_factory, mlir_tool, llvm_bin, constant):
    weights = {"x": np.array(-0.25, dtype=np.float32)} if constant else None
    model = model_factory([], {} if constant else {"x": ()}, {"x": ()}, weights)
    run = compile_model(model, compiler="mlir", mlir_opt=mlir_tool)
    assert run.metrics["cycles"] == 0
    assert verify_outputs(model, run.inputs, execute_llvm(run, tmp_path, llvm_bin))["passed"]


def test_mlir_bounded_deep_graph(mlir_tool):
    tensors = {n: Tensor(n, (2, 2)) for n in ["x", *[f"t{i}" for i in range(96)]]}
    operations = [
        Operation(f"r{i}", "Relu", ["x" if i == 0 else f"t{i - 1}"], f"t{i}") for i in range(96)
    ]
    graph = Graph(tensors, operations, ["x"], ["t95"])
    result = compile_graph(graph, tool=mlir_tool)
    assert len(result.graph.operations) == 96
    result.graph.validate()


def test_mlir_comparison_report(tmp_path, mlir_tool):
    from npuforge.benchmark import run_mlir_benchmarks
    from npuforge.reports import render_report

    runs = run_mlir_benchmarks(tmp_path, mlir_tool)
    report = render_report(runs, tmp_path / "report.html").read_text()
    assert "MLIR compiler path" in report
    assert "lowered.linalg.mlir" in report
    assert len(runs) == 8
    for i in (0, 1, 4, 5):
        assert runs[i][1].metrics["cycles"] == runs[i + 2][1].metrics["cycles"]

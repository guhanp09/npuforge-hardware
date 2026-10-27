# Examples

After installation, `npuforge demo` generates and runs the MatMul and MLP ONNX models.
The standalone generator also creates a Gemm dense layer with transposed weights,
alpha 0.75, beta 0.5 and a vector bias, plus a Gemm/Relu model for fusion comparison.
To inspect compilation manually:

```bash
python examples/generate_models.py --output output/models
npuforge inspect output/models/demo_mlp.onnx
npuforge compile output/models/demo_mlp.onnx --no-opt --output output/baseline
npuforge compile output/models/demo_mlp.onnx --compare --output output/optimized
npuforge compile output/models/demo_matmul.onnx --config configs/pim.json --output output/pim
npuforge compile output/models/demo_gemm.onnx --compare --output output/gemm
python examples/compare_gemm_fusion.py
```

Generated models and artifacts stay in ignored `output/`. No downloaded network
weights, specialized accelerator hardware, external account or service is required.

The fusion comparison produces one report containing conventional/PIM, each with
optimization disabled and enabled. Every case is compiled, simulated natively and
verified against ONNX Runtime. Override its artifact directory with `--output`.

## Execute a saved binary program

```bash
npuforge demo
npuforge inspect-program output/mlp_optimized
npuforge run output/mlp_optimized --inputs output/mlp_optimized/inputs.npz \
  --output output/replayed.npz
npuforge verify output/models/demo_mlp.onnx --compiled output/mlp_optimized
npuforge benchmark --compare-schedulers --output output/schedulers
```

`run` needs only the five compiled artifact files and runtime inputs. The ONNX file
is used only by `verify`. Outputs are an NPZ keyed by the original graph output
names; a neighboring metrics JSON records native timing and memory use. The demo's
sample inputs are ordinary runtime files and can be replaced without recompiling.

## MLIR and LLVM

After building the optional tool as described in [the MLIR guide](../docs/mlir.md):

```bash
export NPUFORGE_MLIR_OPT="$PWD/build/mlir/npuforge-opt"
npuforge demo --compiler mlir --output output/mlir-demo
python examples/compare_mlir.py
npuforge compile output/mlir-demo/models/demo_mlp.onnx --compiler mlir \
  --emit-llvm --llvm-bin /path/to/llvm19/bin --output output/mlir-llvm
```

The eight-case comparison uses identical MLP inputs, weights and hardware for the
Python and MLIR optimizers, with optimization disabled/enabled on both mappings.
Saved binary programs run through the ordinary `npuforge run` command and no longer
require the MLIR toolchain.

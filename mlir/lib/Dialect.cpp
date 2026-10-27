#include "NPUForge.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringSet.h"
#include <cmath>

#include "NPUForgeDialect.cpp.inc"
#define GET_OP_CLASSES
#include "NPUForgeOps.cpp.inc"
using namespace mlir;
namespace npuforge::mlir_ext {
void NPUForgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPUForgeOps.cpp.inc"
      >();
}
namespace {
LogicalResult tensorType(Operation *op, Type type) {
  auto t = dyn_cast<RankedTensorType>(type);
  if (!t || !t.getElementType().isF32() || !t.hasStaticShape() || t.getRank() > 64)
    return op->emitOpError("requires static ranked FP32 tensors");
  uint64_t bytes = 4;
  for (int64_t d : t.getShape()) {
    if (d <= 0 || uint64_t(d) > (64ULL * 1024 * 1024) / bytes)
      return op->emitOpError("tensor size exceeds the positive 64 MiB bound");
    bytes *= d;
  }
  return success();
}
LogicalResult attributes(Operation *op, bool constant, bool gemm, bool fused) {
  for (auto a : op->getAttrs()) {
    auto n = a.getName().strref();
    if (n == "tensor_id" || (!constant && n == "id") || (constant && n == "value") ||
        (gemm && (n == "alpha" || n == "beta" || n == "transA" || n == "transB")) ||
        ((fused || gemm) && n == "fused_from"))
      continue;
    return op->emitOpError("unsupported attribute ") << n;
  }
  if (op->getAttrOfType<StringAttr>("tensor_id").getValue().empty() ||
      (!constant && op->getAttrOfType<StringAttr>("id").getValue().empty()))
    return op->emitOpError("tensor and operation identities must be nonempty");
  return success();
}
bool broadcastInto(ArrayRef<int64_t> source, ArrayRef<int64_t> target) {
  if (source.size() > target.size())
    return false;
  for (size_t j = 0; j < source.size(); ++j) {
    auto a = source[source.size() - 1 - j], b = target[target.size() - 1 - j];
    if (a != 1 && a != b)
      return false;
  }
  return true;
}
LogicalResult verifyTensorOp(Operation *op) {
  auto name = op->getName().getStringRef();
  bool gemm = name == "npuforge.gemm" || name == "npuforge.fused_gemm_relu";
  bool fused = name == "npuforge.fused_matmul_add_relu" || name == "npuforge.fused_gemm_relu";
  if (failed(attributes(op, false, gemm, fused)))
    return failure();
  for (auto type : op->getOperandTypes())
    if (failed(tensorType(op, type)))
      return failure();
  if (failed(tensorType(op, op->getResult(0).getType())))
    return failure();
  auto out = cast<RankedTensorType>(op->getResult(0).getType()).getShape();
  auto shape = [&](unsigned j) {
    return cast<RankedTensorType>(op->getOperand(j).getType()).getShape();
  };
  unsigned count = op->getNumOperands();
  if (name == "npuforge.relu") {
    if (count == 1 && shape(0) == out)
      return success();
  } else if (name == "npuforge.add") {
    if (count == 2 && broadcastInto(shape(0), out) && broadcastInto(shape(1), out)) {
      SmallVector<int64_t> expected(std::max(shape(0).size(), shape(1).size()), 1);
      for (unsigned operand = 0; operand < 2; ++operand) {
        auto s = shape(operand);
        for (size_t j = 0; j < s.size(); ++j)
          expected[expected.size() - 1 - j] =
              std::max(expected[expected.size() - 1 - j], s[s.size() - 1 - j]);
      }
      if (ArrayRef<int64_t>(expected) == out)
        return success();
    }
  } else {
    if ((!gemm && count != (fused ? 3U : 2U)) || (gemm && count != 2 && count != 3))
      return op->emitOpError("invalid matrix operand count");
    if (shape(0).size() != 2 || shape(1).size() != 2 || out.size() != 2)
      return op->emitOpError("matrix operands and result must have rank two");
    bool ta = gemm && op->getAttrOfType<IntegerAttr>("transA").getInt() != 0;
    bool tb = gemm && op->getAttrOfType<IntegerAttr>("transB").getInt() != 0;
    if (gemm)
      for (auto key : {"alpha", "beta"})
        if (!std::isfinite(op->getAttrOfType<FloatAttr>(key).getValueAsDouble()))
          return op->emitOpError("Gemm coefficients must be finite");
    if (shape(0)[ta ? 0 : 1] == shape(1)[tb ? 1 : 0] && out[0] == shape(0)[ta ? 1 : 0] &&
        out[1] == shape(1)[tb ? 0 : 1] && (count == 2 || broadcastInto(shape(2), out)))
      return success();
  }
  return op->emitOpError("operand/result shapes or broadcasting disagree");
}
} // namespace
LogicalResult ConstantOp::verify() {
  auto op = getOperation();
  if (failed(attributes(op, true, false, false)) || failed(tensorType(op, getResult().getType())))
    return failure();
  auto value = dyn_cast<DenseFPElementsAttr>(getValue());
  if (!value || value.getType() != getResult().getType())
    return emitOpError("constant must contain matching dense FP32 values");
  for (auto v : value.getValues<APFloat>())
    if (!v.isFinite())
      return emitOpError("non-finite initializer");
  return success();
}
LogicalResult MatmulOp::verify() { return verifyTensorOp(getOperation()); }
LogicalResult AddOp::verify() { return verifyTensorOp(getOperation()); }
LogicalResult ReluOp::verify() { return verifyTensorOp(getOperation()); }
LogicalResult FusedMatmulOp::verify() { return verifyTensorOp(getOperation()); }
LogicalResult GemmOp::verify() { return verifyTensorOp(getOperation()); }
LogicalResult FusedGemmOp::verify() { return verifyTensorOp(getOperation()); }

LogicalResult verifyGraph(ModuleOp module) {
  auto schema = module->getAttrOfType<IntegerAttr>("npuforge.schema");
  auto model = module->getAttrOfType<StringAttr>("npuforge.model");
  if (!schema || schema.getInt() != 1 || !model)
    return module.emitError("expected npuforge.schema = 1 and model name");
  auto functions = module.getOps<func::FuncOp>();
  if (std::distance(functions.begin(), functions.end()) != 1 ||
      std::distance(module.getBody()->begin(), module.getBody()->end()) != 1)
    return module.emitError("expected exactly one main function");
  auto f = *functions.begin();
  if (f.getName() != "main" || f.isExternal() || !llvm::hasSingleElement(f.getBody()))
    return f.emitError("expected one defined, single-block main function");
  auto inputs = f->getAttrOfType<ArrayAttr>("npuforge.input_names");
  auto outputs = f->getAttrOfType<ArrayAttr>("npuforge.output_names");
  auto ret = dyn_cast<func::ReturnOp>(f.getBody().front().getTerminator());
  if (!inputs || !outputs || !ret || inputs.size() != f.getNumArguments() || outputs.empty() ||
      outputs.size() != ret.getNumOperands())
    return f.emitError("input/output names must match the function signature");
  llvm::StringSet<> names, ids, publicNames;
  for (auto item : llvm::enumerate(inputs)) {
    auto name = dyn_cast<StringAttr>(item.value());
    if (!name || name.getValue().empty() || !names.insert(name.getValue()).second ||
        failed(tensorType(f, f.getArgument(item.index()).getType())))
      return f.emitError("invalid input tensor identity/type");
  }
  for (auto &op : f.getBody().front()) {
    if (isa<func::ReturnOp>(op))
      continue;
    if (op.getDialect() != module.getContext()->getLoadedDialect<NPUForgeDialect>() ||
        op.getNumResults() != 1)
      return op.emitError("unsupported operation in graph bridge");
    auto name = op.getAttrOfType<StringAttr>("tensor_id");
    auto id = op.getAttrOfType<StringAttr>("id");
    if (!name || !names.insert(name.getValue()).second || (id && !ids.insert(id.getValue()).second))
      return op.emitError("duplicate tensor or operation identity");
  }
  for (auto item : llvm::enumerate(outputs)) {
    auto name = dyn_cast<StringAttr>(item.value());
    auto value = ret.getOperand(item.index());
    StringAttr actual;
    if (auto arg = dyn_cast<BlockArgument>(value))
      actual = cast<StringAttr>(inputs[arg.getArgNumber()]);
    else
      actual = value.getDefiningOp()->getAttrOfType<StringAttr>("tensor_id");
    if (!name || !actual || actual != name || !publicNames.insert(name.getValue()).second)
      return ret.emitError("output identity disagrees with its returned SSA value");
  }
  return success();
}
} // namespace npuforge::mlir_ext

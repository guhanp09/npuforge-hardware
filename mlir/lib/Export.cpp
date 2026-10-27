#include "NPUForge.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace mlir;
namespace npuforge::mlir_ext {
namespace {
llvm::json::Array strings(ArrayAttr values) {
  llvm::json::Array result;
  for (auto a : values)
    result.push_back(cast<StringAttr>(a).getValue());
  return result;
}
llvm::json::Object tensor(StringRef name, RankedTensorType type, DenseFPElementsAttr data = {}) {
  llvm::json::Array shape;
  for (int64_t d : type.getShape())
    shape.push_back(d);
  llvm::json::Object result{
      {"id", name}, {"shape", std::move(shape)}, {"dtype", "float32"}, {"data_hex", nullptr}};
  if (data) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string bytes;
    bytes.reserve(type.getNumElements() * 8);
    for (auto value : data.getValues<APFloat>()) {
      uint32_t bits = static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
      for (unsigned j = 0; j < 4; ++j) {
        auto b = (bits >> (j * 8)) & 255;
        bytes.push_back(digits[b >> 4]);
        bytes.push_back(digits[b & 15]);
      }
    }
    result["data_hex"] = std::move(bytes);
  }
  return result;
}
struct ExportPass : PassWrapper<ExportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportPass)
  ExportPass() = default;
  ExportPass(const ExportPass &other) : PassWrapper(other) {}
  Option<std::string> output{*this, "output", llvm::cl::desc("Graph bridge JSON path"),
                             llvm::cl::init("")};
  StringRef getArgument() const final { return "npuforge-export-graph"; }
  StringRef getDescription() const final {
    return "Export verified SSA values to the NPUForge graph backend";
  }
  void runOnOperation() override {
    auto module = getOperation();
    if (failed(verifyGraph(module)) || output.empty()) {
      module.emitError("graph export requires a valid graph and output path");
      signalPassFailure();
      return;
    }
    auto function = *module.getOps<func::FuncOp>().begin();
    auto inputNames = function->getAttrOfType<ArrayAttr>("npuforge.input_names");
    llvm::DenseMap<Value, std::string> names;
    llvm::json::Array tensors, operations;
    for (auto arg : function.getArguments()) {
      auto name = cast<StringAttr>(inputNames[arg.getArgNumber()]).getValue();
      names[arg] = name.str();
      tensors.push_back(tensor(name, cast<RankedTensorType>(arg.getType())));
    }
    for (auto &op : function.getBody().front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      auto name = op.getAttrOfType<StringAttr>("tensor_id").getValue();
      names[op.getResult(0)] = name.str();
      auto constant = dyn_cast<ConstantOp>(op);
      tensors.push_back(tensor(name, cast<RankedTensorType>(op.getResult(0).getType()),
                               constant ? cast<DenseFPElementsAttr>(constant.getValue())
                                        : DenseFPElementsAttr{}));
      if (constant)
        continue;
      auto kind = llvm::StringSwitch<StringRef>(op.getName().getStringRef())
                      .Case("npuforge.matmul", "MatMul")
                      .Case("npuforge.add", "Add")
                      .Case("npuforge.relu", "Relu")
                      .Case("npuforge.gemm", "Gemm")
                      .Case("npuforge.fused_gemm_relu", "FusedGemmRelu")
                      .Case("npuforge.fused_matmul_add_relu", "FusedMatMulAddRelu")
                      .Default("");
      if (kind.empty()) {
        op.emitError("unsupported tensor operation in graph export");
        signalPassFailure();
        return;
      }
      llvm::json::Array inputs;
      for (auto value : op.getOperands())
        inputs.push_back(names.at(value));
      llvm::json::Object attrs;
      for (auto key : {"alpha", "beta"})
        if (auto a = op.getAttrOfType<FloatAttr>(key))
          attrs[key] = a.getValueAsDouble();
      for (auto key : {"transA", "transB"})
        if (auto a = op.getAttrOfType<IntegerAttr>(key))
          attrs[key] = a.getInt();
      if (auto a = op.getAttrOfType<ArrayAttr>("fused_from"))
        attrs["fused_from"] = strings(a);
      operations.push_back(llvm::json::Object{{"id", op.getAttrOfType<StringAttr>("id").getValue()},
                                              {"op", kind},
                                              {"inputs", std::move(inputs)},
                                              {"output", name},
                                              {"attrs", std::move(attrs)}});
    }
    llvm::json::Object graph{
        {"schema", "npuforge.mlir.bridge.v1"},
        {"name", module->getAttrOfType<StringAttr>("npuforge.model").getValue()},
        {"inputs", strings(inputNames)},
        {"outputs", strings(function->getAttrOfType<ArrayAttr>("npuforge.output_names"))},
        {"tensors", std::move(tensors)},
        {"operations", std::move(operations)}};
    std::error_code error;
    llvm::ToolOutputFile file(output, error, llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError("cannot open graph bridge: ") << error.message();
      signalPassFailure();
      return;
    }
    file.os() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(graph)));
    file.keep();
  }
};
} // namespace
void registerExportPass() { PassRegistration<ExportPass>(); }
} // namespace npuforge::mlir_ext

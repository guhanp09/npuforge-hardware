#include "NPUForge.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<npuforge::mlir_ext::NPUForgeDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect>();
  npuforge::mlir_ext::registerPasses();
  npuforge::mlir_ext::registerLoweringPass();
  npuforge::mlir_ext::registerExportPass();
  mlir::registerTransformsPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPUForge MLIR compiler\n", registry));
}

#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "NPUForgeDialect.h.inc"
#define GET_OP_CLASSES
#include "NPUForgeOps.h.inc"
namespace npuforge::mlir_ext {
void registerPasses();
void registerLoweringPass();
void registerExportPass();
mlir::LogicalResult verifyGraph(mlir::ModuleOp module);
} // namespace npuforge::mlir_ext

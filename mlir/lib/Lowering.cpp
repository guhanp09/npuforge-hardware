#include "NPUForge.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace npuforge::mlir_ext {
namespace {
Value empty(ConversionPatternRewriter &r, Location loc, RankedTensorType type) {
  return r.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
}
AffineMap broadcastMap(RankedTensorType source, unsigned rank, MLIRContext *context) {
  SmallVector<AffineExpr> expressions;
  for (unsigned j = 0; j < unsigned(source.getRank()); ++j)
    expressions.push_back(source.getDimSize(j) == 1
                              ? getAffineConstantExpr(0, context)
                              : getAffineDimExpr(rank - source.getRank() + j, context));
  return AffineMap::get(rank, 0, expressions, context);
}
Value elementwise(ConversionPatternRewriter &r, Location loc, RankedTensorType type,
                  ValueRange inputs, bool add, bool relu, float alpha = 1, float beta = 1) {
  SmallVector<AffineMap> maps;
  unsigned rank = type.getRank();
  for (auto v : inputs)
    maps.push_back(broadcastMap(cast<RankedTensorType>(v.getType()), rank, r.getContext()));
  maps.push_back(r.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);
  auto generic = r.create<linalg::GenericOp>(
      loc, TypeRange{type}, inputs, ValueRange{empty(r, loc, type)}, maps, iterators,
      [&](OpBuilder &b, Location at, ValueRange args) {
        Value value = args[0];
        if (alpha != 1) {
          auto factor = b.create<arith::ConstantOp>(at, b.getF32FloatAttr(alpha));
          value = b.create<arith::MulFOp>(at, value, factor);
        }
        if (add) {
          Value bias = args[1];
          if (beta != 1) {
            auto factor = b.create<arith::ConstantOp>(at, b.getF32FloatAttr(beta));
            bias = b.create<arith::MulFOp>(at, bias, factor);
          }
          value = b.create<arith::AddFOp>(at, value, bias);
        }
        if (relu) {
          auto zero = b.create<arith::ConstantOp>(at, b.getF32FloatAttr(0));
          value = b.create<arith::MaximumFOp>(at, value, zero);
        }
        b.create<linalg::YieldOp>(at, value);
      });
  return generic.getResult(0);
}
struct LowerTensor : ConversionPattern {
  LowerTensor(StringRef name, MLIRContext *context) : ConversionPattern(name, 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> inputs,
                                ConversionPatternRewriter &r) const override {
    auto loc = op->getLoc();
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    auto name = op->getName().getStringRef();
    if (name == "npuforge.constant") {
      r.replaceOpWithNewOp<arith::ConstantOp>(op, cast<TypedAttr>(op->getAttr("value")));
      return success();
    }
    if (name == "npuforge.add" || name == "npuforge.relu") {
      r.replaceOp(
          op, elementwise(r, loc, type, inputs, name == "npuforge.add", name == "npuforge.relu"));
      return success();
    }
    bool gemm = name == "npuforge.gemm" || name == "npuforge.fused_gemm_relu";
    bool relu = name == "npuforge.fused_gemm_relu" || name == "npuforge.fused_matmul_add_relu";
    bool ta = gemm && op->getAttrOfType<IntegerAttr>("transA").getInt() != 0;
    bool tb = gemm && op->getAttrOfType<IntegerAttr>("transB").getInt() != 0;
    auto zero = r.create<arith::ConstantOp>(loc, r.getF32FloatAttr(0));
    auto fill = r.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty(r, loc, type)});
    Value product;
    if (!ta && !tb) {
      product = r.create<linalg::MatmulOp>(loc, TypeRange{type}, ValueRange{inputs[0], inputs[1]},
                                           ValueRange{fill.getResult(0)})
                    .getResult(0);
    } else {
      auto m = r.getAffineDimExpr(0), n = r.getAffineDimExpr(1), k = r.getAffineDimExpr(2);
      SmallVector<AffineMap> maps{
          AffineMap::get(3, 0, ta ? ArrayRef<AffineExpr>{k, m} : ArrayRef<AffineExpr>{m, k},
                         r.getContext()),
          AffineMap::get(3, 0, tb ? ArrayRef<AffineExpr>{n, k} : ArrayRef<AffineExpr>{k, n},
                         r.getContext()),
          AffineMap::get(3, 0, {m, n}, r.getContext())};
      SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel,
                                                 utils::IteratorType::parallel,
                                                 utils::IteratorType::reduction};
      product = r.create<linalg::GenericOp>(loc, TypeRange{type}, ValueRange{inputs[0], inputs[1]},
                                            ValueRange{fill.getResult(0)}, maps, iterators,
                                            [&](OpBuilder &b, Location at, ValueRange args) {
                                              auto mul =
                                                  b.create<arith::MulFOp>(at, args[0], args[1]);
                                              auto sum = b.create<arith::AddFOp>(at, args[2], mul);
                                              b.create<linalg::YieldOp>(at, sum.getResult());
                                            })
                    .getResult(0);
    }
    float alpha = gemm ? op->getAttrOfType<FloatAttr>("alpha").getValue().convertToFloat() : 1;
    float beta = gemm ? op->getAttrOfType<FloatAttr>("beta").getValue().convertToFloat() : 1;
    bool bias = inputs.size() == 3;
    if (bias || relu || alpha != 1) {
      SmallVector<Value> epilogue{product};
      if (bias)
        epilogue.push_back(inputs[2]);
      product = elementwise(r, loc, type, epilogue, bias, relu, alpha, beta);
    }
    r.replaceOp(op, product);
    return success();
  }
};
struct LowerPass : PassWrapper<LowerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPass)
  StringRef getArgument() const final { return "convert-npuforge-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower static FP32 tensors to Linalg, Tensor and Arith";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    if (failed(verifyGraph(getOperation()))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    for (auto name :
         {"npuforge.constant", "npuforge.matmul", "npuforge.add", "npuforge.relu", "npuforge.gemm",
          "npuforge.fused_gemm_relu", "npuforge.fused_matmul_add_relu"})
      patterns.add<LowerTensor>(name, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, func::FuncDialect, arith::ArithDialect,
                           tensor::TensorDialect, linalg::LinalgDialect>();
    target.addIllegalDialect<NPUForgeDialect>();
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
void registerLoweringPass() { PassRegistration<LowerPass>(); }
} // namespace npuforge::mlir_ext

#include "NPUForge.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace npuforge::mlir_ext {
namespace {
struct FuseRelu : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp relu, PatternRewriter &rewriter) const override {
    auto *source = relu.getInputs()[0].getDefiningOp();
    if (!source || !source->hasOneUse())
      return failure();
    SmallVector<Value> inputs;
    SmallVector<Attribute> provenance;
    StringRef name;
    NamedAttrList attrs(relu->getAttrs());
    if (auto gemm = dyn_cast<GemmOp>(source)) {
      name = FusedGemmOp::getOperationName();
      inputs.assign(gemm.getInputs().begin(), gemm.getInputs().end());
      for (auto key : {"alpha", "beta", "transA", "transB"})
        attrs.set(key, gemm->getAttr(key));
      provenance = {gemm->getAttr("id"), relu->getAttr("id")};
    } else if (auto add = dyn_cast<AddOp>(source)) {
      for (unsigned slot = 0; slot < 2; ++slot) {
        auto mm = add.getInputs()[slot].getDefiningOp<MatmulOp>();
        if (!mm || !mm->hasOneUse() || add.getInputs()[0] == add.getInputs()[1] ||
            mm.getResult().getType() != add.getResult().getType())
          continue;
        name = FusedMatmulOp::getOperationName();
        inputs.assign(mm.getInputs().begin(), mm.getInputs().end());
        inputs.push_back(add.getInputs()[1 - slot]);
        provenance = {mm->getAttr("id"), add->getAttr("id"), relu->getAttr("id")};
        break;
      }
    }
    if (name.empty())
      return failure();
    attrs.set("fused_from", rewriter.getArrayAttr(provenance));
    OperationState state(relu.getLoc(), name);
    state.addOperands(inputs);
    state.addTypes(relu.getResult().getType());
    state.addAttributes(attrs);
    auto *fused = rewriter.create(state);
    rewriter.replaceOp(relu, fused->getResults());
    return success();
  }
};
struct OptimizePass : PassWrapper<OptimizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePass)
  StringRef getArgument() const final { return "npuforge-optimize"; }
  StringRef getDescription() const final {
    return "Fuse observable-safe tensor epilogues and eliminate dead pure operations";
  }
  void runOnOperation() override {
    if (failed(verifyGraph(getOperation()))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRelu>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))) ||
        failed(verifyGraph(getOperation())))
      signalPassFailure();
  }
};
struct VerifyPass : PassWrapper<VerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)
  StringRef getArgument() const final { return "npuforge-verify"; }
  StringRef getDescription() const final {
    return "Verify the graph bridge and public SSA identities";
  }
  void runOnOperation() override {
    if (failed(verifyGraph(getOperation())))
      signalPassFailure();
  }
};
} // namespace
void registerPasses() {
  PassRegistration<OptimizePass>();
  PassRegistration<VerifyPass>();
}
} // namespace npuforge::mlir_ext

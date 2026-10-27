include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NPUForge_Dialect : Dialect {
  let name = "npuforge";
  let cppNamespace = "::npuforge::mlir_ext";
  let summary = "Static FP32 tensor operations for the NPUForge compiler";
}
class NPUForge_Op<string mnemonic> : Op<NPUForge_Dialect, mnemonic, [Pure]> {
  let hasVerifier = 1;
}
def ConstantOp : NPUForge_Op<"constant"> {
  let arguments = (ins ElementsAttr:$value, StrAttr:$tensor_id);
  let results = (outs AnyRankedTensor:$result);
}
class TensorOp<string mnemonic> : NPUForge_Op<mnemonic> {
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       StrAttr:$id, StrAttr:$tensor_id);
  let results = (outs AnyRankedTensor:$result);
}
def MatmulOp : TensorOp<"matmul">;
def AddOp : TensorOp<"add">;
def ReluOp : TensorOp<"relu">;
def FusedMatmulOp : NPUForge_Op<"fused_matmul_add_relu"> {
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       StrAttr:$id, StrAttr:$tensor_id, StrArrayAttr:$fused_from);
  let results = (outs AnyRankedTensor:$result);
}
class GemmBase<string mnemonic> : NPUForge_Op<mnemonic> {
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       StrAttr:$id, StrAttr:$tensor_id,
                       F32Attr:$alpha, F32Attr:$beta, I64Attr:$transA, I64Attr:$transB,
                       OptionalAttr<StrArrayAttr>:$fused_from);
  let results = (outs AnyRankedTensor:$result);
}
def GemmOp : GemmBase<"gemm">;
def FusedGemmOp : GemmBase<"fused_gemm_relu">;

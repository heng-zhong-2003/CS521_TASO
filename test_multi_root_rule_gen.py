from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.graph import Graph
import ast
from onnx_gen.rule_gen import RuleGen
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
import torch
import torch.nn as nn
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import onnxscript


class MatMulAdd(nn.Module):
    def forward(self, a, b, c):
        return a @ b, a @ c


a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.randn(3, 4)
model = MatMulAdd()

torch.onnx.export(
    model,
    (a, b, c),
    "test_multi_root.onnx",
    opset_version=15,
    input_names=["a", "b", "c"],
    output_names=["out1", "out2"],
    do_constant_folding=False,
)

onnx_model = onnx.load("test_multi_root.onnx")
print("Original ONNX model:")
print(onnx_model)


def target(op, a, b, c):
    y0 = op.MatMul(a, b)
    y1 = op.MatMul(a, c)
    return y0, y1


def replacement(op, a, b, c):
    bc = op.Concat(b, c, axis=1)
    fused = op.MatMul(a, bc)
    out0, out1 = op.Split(
        fused, num_outputs=2, axis=1, _outputs=["out1", "out2"]
    )
    return out0, out1


def cond(ctx, a, b, c):
    n0, n1 = ctx.nodes
    # Require that the two MatMul are not the same to prevent
    #   infinite matching on the same node.
    return (n0 is not n1) and (n0.inputs[1] is not n1.inputs[1])


mrr = pattern.RewriteRule(target, replacement, condition_function=cond)

ir_model = ir.serde.deserialize_model(onnx_model)
pattern.RewriteRuleSet([mrr]).apply_to_model(ir_model, verbose=2)

# pattern.RewriteRuleSet([mrr]).apply_to_model(
#     onnx_model, verbose=2, tracer=pattern.MatchingTracer()
# )

rm = ir.serde.serialize_model(ir_model)

# rm = onnxscript.rewriter.rewrite(
#     model=onnx_model,
#     pattern_rewrite_rules=[mrr],
# )
print("Rewritten ONNX model:")
print(rm)

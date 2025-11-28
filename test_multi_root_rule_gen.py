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


def target_pattern(op, in1, in2, in3):
    return op.MatMul(in1, in2), op.MatMul(in1, in3)


def replacement_pattern(op, in1, in2, in3):
    return op.MatMul(in1, op.Sub(op.Add(in2, in3), in3)), \
        op.MatMul(in1, op.Sub(op.Add(in2, in3), in2))


mr_matcher = pattern.SimplePatternMatcher(
    onnx_model.graph,
)

mrr = pattern.RewriteRule(
    target_pattern,
    replacement_pattern,
    matcher=mr_matcher
)
rm = onnxscript.rewriter.rewrite(
    model=onnx_model,
    pattern_rewrite_rules=[mrr],
)
print("Rewritten ONNX model:")
print(rm)

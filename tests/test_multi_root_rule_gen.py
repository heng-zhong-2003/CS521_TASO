import itertools
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


class MatMulPair(nn.Module):
    def forward(self, a, b, c):
        return a @ b, a @ c


a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.randn(3, 4)
model = MatMulPair()

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


# This is an incorrect rewrite rule.
# Just to test that the rule generation for multi output operator rule works.
a1 = InputOperator()
b1 = InputOperator()
c1 = InputOperator()
mmab = MatmulOperator(a1, b1)
mmac = MatmulOperator(a1, c1)
g1 = Graph([a1, b1, c1])
g1.add_operator(mmab)
g1.add_operator(mmac)

a2 = InputOperator()
b2 = InputOperator()
c2 = InputOperator()
add1 = AddOperator(a2, b2)
add2 = AddOperator(a2, c2)
g2 = Graph([a2, b2, c2])
g2.add_operator(add1)
g2.add_operator(add2)

rg = RuleGen()
rule_ast = rg.generate_rule(g1, g2)
rule_ast = ast.fix_missing_locations(rule_ast)
print(ast.unparse(rule_ast))


# BEGIN: Auto generated rules
def taso_target_0(op, in_0, in_1, in_2):
    return (op.MatMul(in_0, in_1), op.MatMul(in_0, in_2))

def taso_replacement_0(op, in_0, in_1, in_2):
    return (op.Add(in_0, in_1), op.Add(in_0, in_2))

def taso_match_cond_0(ctx, in_0, in_1, in_2):
    return all([TASO_x is not TASO_y for TASO_x, TASO_y in itertools.combinations(ctx.nodes, 2)])
taso_rule_0 = pattern.RewriteRule(taso_target_0, taso_replacement_0, taso_match_cond_0)
# END: Auto generated rules

rm = onnxscript.rewriter.rewrite(
    model=onnx_model,
    pattern_rewrite_rules=[taso_rule_0],
)
print("Rewritten ONNX model:")
print(rm)

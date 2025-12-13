import itertools
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.graph import Graph
import ast
from onnx_gen.rule_gen import RuleGen
import torch
import torch.nn as nn
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import onnxscript
from onnx_gen.concat_info_inference import ConcatInfoInference


a1 = InputOperator()
b1 = InputOperator()
c1 = InputOperator()
mab = MatmulOperator(a1, b1)
mac = MatmulOperator(a1, c1)
g1 = Graph([a1, b1, c1])
g1.add_operator(mab)
g1.add_operator(mac)

a2 = InputOperator()
b2 = InputOperator()
c2 = InputOperator()
cat = ConcatOperator(b2, c2, 1)
macat = MatmulOperator(a2, cat)
spl = SplitOperator(macat, 1, cat)
g2 = Graph([a2, b2, c2])
g2.add_operator(cat)
g2.add_operator(macat)
g2.add_operator(spl)

inferrer1 = ConcatInfoInference(2, g1)
inferrer2 = ConcatInfoInference(2, g2)

print('Inferring g1:')
infer_rslt1 = inferrer1.infer_all()
assert infer_rslt1

print('Inferring g2:')
infer_rslt2 = inferrer2.infer_all()
assert infer_rslt2
print(f'Inferrer 2 op concat info map: {inferrer2.op_concat_info_map}')

rg = RuleGen()
rg.plug_in_inferrers(inferrer1, inferrer2)
rule_ast = rg.generate_rule(g1, g2)
rule_ast = ast.fix_missing_locations(rule_ast)
print(ast.unparse(rule_ast))


# BEGIN: Auto generated rules
def taso_target_0(op, in_0, in_1, in_2):
    return (
        op.MatMul(in_0, in_1),
        op.MatMul(in_0, in_2)
    )


def taso_replacement_0(op, in_0, in_1, in_2):
    return op.Split(
        op.MatMul(in_0, op.Concat(in_1, in_2, axis=1)),
        num_outputs=2,
        splits=[op.Shape(in_1)[1], op.Shape(in_2)[1]]
    )


def taso_match_cond_0(ctx, in_0, in_1, in_2):
    return all(
        [TASO_x is not TASO_y
         for TASO_x, TASO_y in itertools.combinations(ctx.nodes, 2)]
    )


taso_rule_0 = pattern.RewriteRule(
    taso_target_0,
    taso_replacement_0,
    taso_match_cond_0
)
# END: Auto generated rules

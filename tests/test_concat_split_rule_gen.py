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
from patterns.eval_graph import EvalGraph


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

# print('Inferring g1:')
infer_rslt1 = inferrer1.infer_all()
assert infer_rslt1

# print('Inferring g2:')
infer_rslt2 = inferrer2.infer_all()
assert infer_rslt2
# print(f'Inferrer 2 op concat info map: {inferrer2.op_concat_info_map}')

rg = RuleGen()
rg.plug_in_inferrers(inferrer1, inferrer2)
rule_ast = rg.generate_rule(g1, g2)
rule_ast = ast.fix_missing_locations(rule_ast)
print(ast.unparse(rule_ast))


# BEGIN: Auto generated rules
def taso_target_0(op, in_0, in_1, in_2):
    return (op.MatMul(in_0, in_1), op.MatMul(in_0, in_2))


def taso_replacement_0(op, in_0, in_1, in_2):
    return ((_split_0 := op.Split(op.MatMul(in_0, op.Concat(in_1, in_2, axis=1)), op.Concat(op.Unsqueeze(op.Gather(op.Shape(in_1), op.Constant(value_int=1), axis=0), axes=[0]), op.Unsqueeze(op.Gather(op.Shape(in_2), op.Constant(value_int=1), axis=0), axes=[0]), axis=0), axis=1, num_outputs=2, _outputs=2))[0], _split_0[1])


def taso_match_cond_0(ctx, in_0, in_1, in_2):
    return all([TASO_x is not TASO_y for TASO_x, TASO_y in itertools.combinations(ctx.nodes, 2)])


taso_rule_0 = pattern.RewriteRule(
    taso_target_0, taso_replacement_0, taso_match_cond_0)
# END: Auto generated rules

rm = onnxscript.rewriter.rewrite(
    model=onnx.load("test_multi_root.onnx"),
    pattern_rewrite_rules=[taso_rule_0],
)
print("Rewritten ONNX model:")
print(rm)

eg1 = EvalGraph(
    g1,
    [a.numpy(), b.numpy(), c.numpy()],
    {})

eg2 = EvalGraph(
    g2,
    [a.numpy(), b.numpy(), c.numpy()],
    {spl: b.numpy().shape[1]})

rslt1 = eg1.eval_graph()
rslt2 = eg2.eval_graph()

print(f'rslt1: {rslt1}')
print(f'rslt2: {rslt2}')

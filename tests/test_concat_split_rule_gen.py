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

infer_rslt1 = inferrer1.infer_all()
assert infer_rslt1
infer_rslt2 = inferrer2.infer_all()
assert infer_rslt2

rg = RuleGen()
rg.plug_in_inferrers(inferrer1, inferrer2)
rule_ast = rg.generate_rule(g1, g2)
rule_ast = ast.fix_missing_locations(rule_ast)
print(ast.unparse(rule_ast))

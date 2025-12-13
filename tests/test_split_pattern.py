from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
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
from patterns.eval_graph import EvalGraph
import numpy as np
from onnx_gen.concat_info_inference import ConcatInfoInference


a1 = InputOperator()
b1 = InputOperator()
c1 = ConcatOperator(a1, b1, 1)
s1 = SplitOperator(c1, 1, c1)
g1 = Graph([a1, b1])
g1.add_operator(c1)
g1.add_operator(s1)

eg1 = EvalGraph(
    g1,
    [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
    {s1: 2}
)

print(eg1.eval_graph())

cii = ConcatInfoInference(2, g1)
rslt = cii.infer_all()
print('Op -- concat info map:')
print(cii.op_concat_info_map)
print('Op -- dimension symbol map:')
print(cii.op_dim_pos_symbol_map)

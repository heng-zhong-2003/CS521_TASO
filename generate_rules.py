from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_conv2d import Conv2DOperator
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
from synthesizer.validate import RuleValidator
from synthesizer import build
from onnx_gen.rule_gen import RuleGen
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_interface import Operator
from onnx_gen.concat_info_inference import ConcatInfoInference
import ast
from typing import Type
import copy
import traceback


fingerprinter = Fingerprint()
# num of inputs -> D with that many inputs
num_inputs_D_map: dict[int, dict[int, list[Graph]]] = {}
graph_split_inferer_map: dict[Graph, ConcatInfoInference] = {}
P: list[Type[Operator]] = [
    AddOperator, Conv2DOperator]

for num_inputs in range(3, 4):
    I = []
    D: dict[int, list[Graph]] = {} # fingerprint -> list of graphs
    for _ in range(num_inputs):
        I.append(InputOperator())
    init_graph = Graph(I)
    build.build(
        n=1,
        G=init_graph,
        I=copy.copy(I),
        P=P,
        D=D,
        F=fingerprinter,
        threshold=4
    )
    num_inputs_D_map[num_inputs] = D

valid_num_inputs_D_map: dict[int, dict[int, list[Graph]]] = {}

for rank in range(2, 4):
    for num_inputs, D in num_inputs_D_map.items():
        for fp, graphs in D.items():
            for i in range(len(graphs)):
                g = graphs[i]
                split_infer = ConcatInfoInference(rank, g)
                infer_result = split_infer.infer_all()
                if infer_result:
                    if num_inputs not in valid_num_inputs_D_map:
                        valid_num_inputs_D_map[num_inputs] = {}
                    if fp not in valid_num_inputs_D_map[num_inputs]:
                        valid_num_inputs_D_map[num_inputs][fp] = []
                    valid_num_inputs_D_map[num_inputs][fp].append(g)
                    graph_split_inferer_map[g] = split_infer
                else:
                    pass


def validate_rules(fp_graph_map: dict[int, list[Graph]]) -> \
        list[tuple[Graph, Graph]]:
    ret = []
    for fp, graphs in fp_graph_map.items():
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                g1 = graphs[i]
                g2 = graphs[j]
                validator = RuleValidator(graph_split_inferer_map)
                try:
                    if validator.validate(g1, g2):
                        ret.append((g1, g2))
                except Exception as e:
                    print("exception while validation")
                    traceback.print_exc()
    return ret


valid_rules = []

for num_inputs, D in valid_num_inputs_D_map.items():
    valids = validate_rules(D)
    valid_rules.extend(valids)

# TODO: test performance here.

rulegen = RuleGen()

rule_strings = []

for lhs, rhs in valid_rules:
    rulegen.plug_in_inferrers(
        graph_split_inferer_map[lhs],
        graph_split_inferer_map[rhs])
    rule = rulegen.generate_rule(lhs, rhs)
    rule = ast.fix_missing_locations(rule)
    rule_strings.append(ast.unparse(rule))

with open("TASO_generated_rules.py", "w") as f:
    f.write(r'''
import itertools
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_conv2d import Conv2DOperator
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

''')
    for rule_str in rule_strings:
        f.write(rule_str + '\n')
    rules_lst = '['
    for i in range(rulegen.rule_counter):
        rules_lst += f'taso_rule_{i}, '
    rules_lst += ']'
    f.write(f'\nrules = {rules_lst}\n')

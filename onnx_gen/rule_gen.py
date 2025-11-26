from onnx_gen.pattern_to_onnx_expr import PatternToOnnxExpr
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import ast
from patterns.graph import Graph
import proj_utils


class RuleGen:
    def __init__(self) -> None:
        self.rule_counter = 0
    
    def target_pattern_name(self) -> str:
        return f'taso_target_{self.rule_counter}'
    
    def replacement_pattern_name(self) -> str:
        return f'taso_replacement_{self.rule_counter}'
    
    def generate_rule(self,
                      target_pattern: Graph,
                      replacement_pattern: Graph) -> ast.AST:
        proj_utils.todo()

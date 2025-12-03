from __future__ import annotations
from patterns.graph import Graph
from patterns.operator_interface import Operator
import ast
from ast import AST
import proj_utils


class MatchConditionsGen:
    """
    Create the matching conditions for a pattern graph.
    Apply it to source graphs (target pattern in onnx terminology) of rules.
    Including for multi output operator patterns that the matched output
        operators should be different (avoid infinite loops).
    One instance of this class should be created for every pattern graph.
    Furthermore, that instance should be created after the `PatternToOnnxExpr`
        object for the same pattern graph has been created,
        and its conversion (`pattern_to_onnx_pattern`) has been done.
    """
    def __init__(self,
                 input_ops_names_map: dict[Operator, str],
                 graph_onnx_node_map: dict[Operator, AST]) -> None:
        """
        input_ops_names_map: map {graph input op -> input name in onnx pattern}
        graph_onnx_node_map: map {graph op -> onnx pattern node}
        They come from the `PatternToOnnxExpr` object for the conversion of the
            same graph.
        """
        self.input_ops_names_map: dict[Operator, str] = input_ops_names_map
        self.graph_onnx_node_map: dict[Operator, AST] = graph_onnx_node_map
    
    def generate_matching_conditions(self, pattern_graph: Graph) -> AST:
        outputs = pattern_graph.get_outputs()
        if len(outputs) <= 1:
            return ast.Constant(value=True)
        proj_utils.todo()

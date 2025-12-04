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
    Including that for multi output operator patterns the matched output
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
        self.root_parameter_name = 'ctx'

    def generate_matching_conditions(self, pattern_graph: Graph) -> AST:
        outputs = pattern_graph.get_outputs()
        if len(outputs) <= 1:
            return ast.Constant(value=True)
        roots_ast = ast.Attribute(
            value=ast.Name(id=self.root_parameter_name, ctx=ast.Load()),
            attr='nodes',
            ctx=ast.Load()
        )
        cond_ast = ast.Call(
            func=ast.Name(id='all', ctx=ast.Load()),
            args=[
                ast.ListComp(
                    elt=ast.Compare(
                        left=ast.Name(id='TASO_x', ctx=ast.Load()),
                        ops=[ast.IsNot()],
                        comparators=[ast.Name(id='TASO_y', ctx=ast.Load())]
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Tuple(
                                elts=[
                                    ast.Name(id='TASO_x', ctx=ast.Load()),
                                    ast.Name(id='TASO_y', ctx=ast.Load())
                                ],
                                ctx=ast.Store()
                            ),
                            iter=ast.Call(
                                func=ast.Name(id='itertools.combinations',
                                              ctx=ast.Load()),
                                args=[
                                    roots_ast,
                                    ast.Constant(value=len(outputs))
                                ],
                                keywords=[]
                            ),
                            ifs=[],
                            is_async=0
                        )
                    ]
                )
            ],
            keywords=[]
        )
        return cond_ast

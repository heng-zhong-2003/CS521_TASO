import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import ast
from ast import AST
from patterns.graph import Graph
from patterns.operator_add import AddOperator
from patterns.operator_interface import Operator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator


class PatternToOnnxExpr:
    """
    Create a new object of this class for every single conversion.
    This means that for generating the target and replacement patterns
      of a rule, you need to create two separate objects of this class.
    """

    def __init__(self) -> None:
        self.onnx_pattern_op_name = 'op'
        self.input_op_counter = 0
        # map: {graph input op -> input name in onnx pattern}
        self.input_ops_names_map: dict[Operator, str] = {}
        # map: {graph op -> onnx pattern node}
        # Here, the onnx pattern node is actually an AST node
        self.graph_onnx_node_map: dict[Operator, AST] = {}

    def new_input_op_name(self) -> str:
        name = f'in_{self.input_op_counter}'
        self.input_op_counter += 1
        return name

    def add_to_onnx_pattern(self, gop: AddOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs)
        rhs_onnx = self.op_to_onnx_expr(rhs)
        add_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='Add',
                ctx=ast.Load()
            ),
            args=[lhs_onnx, rhs_onnx],
            keywords=[]
        )
        return add_ast

    def matmul_to_onnx_pattern(self, gop: MatmulOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs)
        rhs_onnx = self.op_to_onnx_expr(rhs)
        matmul_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='MatMul',
                ctx=ast.Load()
            ),
            args=[lhs_onnx, rhs_onnx],
            keywords=[]
        )
        return matmul_ast

    def input_to_onnx_pattern(self, gop: InputOperator):
        # name = self.new_input_op_name()
        # self.input_ops.append(name)
        name: str
        if gop in self.input_ops_names_map:
            name = self.input_ops_names_map[gop]
        else:
            name = self.new_input_op_name()
            self.input_ops_names_map[gop] = name
        input_ast = ast.Name(id=name, ctx=ast.Load())
        return input_ast

    def op_to_onnx_expr(self, gop: Operator) -> AST:
        if gop in self.graph_onnx_node_map:
            return self.graph_onnx_node_map[gop]

        result: AST
        match gop:
            case AddOperator():
                result = self.add_to_onnx_pattern(gop)
            case MatmulOperator():
                result = self.matmul_to_onnx_pattern(gop)
            case InputOperator():
                result = self.input_to_onnx_pattern(gop)
            case _:
                raise NotImplementedError(
                    f"ONNX pattern generation not implemented for {type(gop)}")
        self.graph_onnx_node_map[gop] = result
        return result

    def pattern_to_onnx_pattern(self, pattern_graph: Graph) -> \
            tuple[ast.AST, list[str]]:
        # if len(pattern.get_outputs()) > 1:
        #     raise NotImplementedError(
        #         f'Only single-root patterns for now, '
        #         f'rule has {len(pattern.get_outputs())} outputs')
        # print(f'translating node of type {type(pattern.get_outputs()[0])}')
        if len(pattern_graph.get_outputs()) > 1:
            onnx_outputs = []
            for out in pattern_graph.get_outputs():
                onnx_outputs.append(self.op_to_onnx_expr(out))
            outputs_tup = ast.Tuple(elts=onnx_outputs, ctx=ast.Load())
            return outputs_tup, list(self.input_ops_names_map.values())
        else:
            return self.op_to_onnx_expr(pattern_graph.get_outputs()[0]), \
                list(self.input_ops_names_map.values())

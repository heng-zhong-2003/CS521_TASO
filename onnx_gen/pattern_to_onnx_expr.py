import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import ast
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
        self.input_ops_names_map: dict[Operator, str] = {}

    def new_input_op_name(self) -> str:
        name = f'in_{self.input_op_counter}'
        self.input_op_counter += 1
        return name

    def add_to_onnx_pattern(self,gop: AddOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs)
        rhs_onnx = self.op_to_onnx_expr(rhs)
        add_ast = ast.BinOp(
            left=lhs_onnx,
            op=ast.Add(),
            right=rhs_onnx
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

    def op_to_onnx_expr(self, gop: Operator):
        match gop:
            case AddOperator():
                return self.add_to_onnx_pattern(gop)
            case MatmulOperator():
                return self.matmul_to_onnx_pattern(gop)
            case InputOperator():
                return self.input_to_onnx_pattern(gop)
            case _:
                raise NotImplementedError(
                    f"ONNX pattern generation not implemented for {type(gop)}")

    def pattern_to_onnx_pattern(self, pattern: Graph) -> tuple[ast.AST, list[str]]:
        if len(pattern.get_outputs()) > 1:
            raise NotImplementedError(
                f'Only single-root patterns for now, '
                f'rule has {len(pattern.get_outputs())} outputs')
        print(f'translating node of type {type(pattern.get_outputs()[0])}')
        return self.op_to_onnx_expr(pattern.get_outputs()[0]), \
            list(self.input_ops_names_map.values())

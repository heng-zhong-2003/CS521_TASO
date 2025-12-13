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
from patterns.operator_split import SplitOperator
from patterns.operator_concat import ConcatOperator
from onnx_gen.concat_info_inference import \
    ConcatInfo, ConcatInfoInference, OpDim
import proj_utils


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
        self.pos_symbol_runtime_value_ast_map: dict[str, ast.AST] = {}
        self.inferrer: ConcatInfoInference

    def plug_in_inferrer(self, inferrer: ConcatInfoInference) -> None:
        """
        Must call this on the inferrer applied on the same graph pattern before
          using pattern_to_onnx_pattern() if the pattern has concat-split.
        """
        self.inferrer = inferrer

    def new_input_op_name(self) -> str:
        name = f'in_{self.input_op_counter}'
        self.input_op_counter += 1
        return name

    def get_dim_runtime_value_ast(self, op: Operator, dim: int) -> AST:
        op_ast = self.graph_onnx_node_map[op]
        shape_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='Shape',
                ctx=ast.Load()
            ),
            args=[op_ast],  # type: ignore
            keywords=[]
        )
        dim_len_ast = ast.Subscript(
            value=shape_ast,
            slice=ast.Constant(value=dim),
            ctx=ast.Load()
        )
        return dim_len_ast

    def split_to_onnx_pattern(self, gop: SplitOperator):
        inp, = gop.get_inputs()
        inp_onnx: AST = self.op_to_onnx_expr(inp, user=gop)
        inp_concat_info: ConcatInfo = self.inferrer.op_concat_info_map[inp][0]
        undone_concat: Operator = inp_concat_info.cocat_op
        split_dim = inp_concat_info.concat_dim
        split_pos_symb: str = self.inferrer.op_dim_pos_symbol_map[
            OpDim(undone_concat, split_dim)]
        split_pos_ast: AST = \
            self.pos_symbol_runtime_value_ast_map[split_pos_symb]
        dim_len_ast = self.get_dim_runtime_value_ast(gop, split_dim)
        second_len_ast = ast.BinOp(
            left=dim_len_ast,  # type: ignore
            op=ast.Sub(),
            right=split_pos_ast  # type: ignore
        )
        assert isinstance(split_pos_ast, ast.expr)
        split_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='Split',
                ctx=ast.Load()
            ),
            args=[inp_onnx],  # type: ignore
            keywords=[
                ast.keyword(arg='num_outputs', value=ast.Constant(value=2)),
                ast.keyword(
                    arg='splits',
                    value=ast.List([split_pos_ast, second_len_ast],
                                   ctx=ast.Load()))
            ]
        )
        return split_ast

    def concat_to_onnx_pattern(self, gop: ConcatOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        this_concat_pos_symbol: str = \
            self.inferrer.op_dim_pos_symbol_map[OpDim(gop, gop.axis)]
        this_concat_dim = gop.axis
        this_concat_pos_ast = \
            self.get_dim_runtime_value_ast(gop, this_concat_dim)
        self.pos_symbol_runtime_value_ast_map[this_concat_pos_symbol] = \
            this_concat_pos_ast
        concat_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='Concat',
                ctx=ast.Load()
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore
            keywords=[
                ast.keyword(arg='axis', value=ast.Constant(value=gop.axis))
            ]
        )
        return concat_ast

    def add_to_onnx_pattern(self, gop: AddOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        add_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='Add',
                ctx=ast.Load()
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore
            keywords=[]
        )
        return add_ast

    def matmul_to_onnx_pattern(self, gop: MatmulOperator):
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        matmul_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr='MatMul',
                ctx=ast.Load()
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore
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
    
    def get_indexing_split_result_ast(self, split_ast: AST, index: int) -> AST:
        assert index in (0, 1)
        assert isinstance(split_ast, ast.expr)
        index_ast = ast.Constant(value=index)
        indexed_ast = ast.Subscript(
            value=split_ast,
            slice=index_ast,
            ctx=ast.Load()
        )
        return indexed_ast

    def op_to_onnx_expr(self,
                        gop: Operator,
                        user: None | Operator = None) -> AST:
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
            case ConcatOperator():
                result = self.concat_to_onnx_pattern(gop)
            case SplitOperator():
                assert user is not None
                split_ast = self.split_to_onnx_pattern(gop)
                # Determine which output of the split to use.
                user_index = gop.get_user_component(user)
                result = self.get_indexing_split_result_ast(
                    split_ast, user_index)
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
                onnx_outputs.append(self.op_to_onnx_expr(out, user=out))
            outputs_tup = ast.Tuple(elts=onnx_outputs, ctx=ast.Load())
            return outputs_tup, list(self.input_ops_names_map.values())
        else:
            return (
                self.op_to_onnx_expr(
                    pattern_graph.get_outputs()[0],
                    user=pattern_graph.get_outputs()[0]),
                list(self.input_ops_names_map.values())
            )

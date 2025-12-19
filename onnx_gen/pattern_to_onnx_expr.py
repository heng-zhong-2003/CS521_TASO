import ast
import copy
from ast import AST
from patterns.graph import Graph
from patterns.operator_add import AddOperator
from patterns.operator_interface import Operator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
from patterns.operator_split import SplitOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_conv2d import Conv2DOperator
from onnx_gen.concat_info_inference import (
    ConcatInfo,
    ConcatInfoInference,
    OpDim,
)
import proj_utils


class PatternToOnnxExpr:
    """
    Translate a pattern graph into ONNXScript expressions.
    """

    def __init__(self) -> None:
        self.onnx_pattern_op_name = "op"
        self.input_op_counter = 0
        self.input_ops_names_map: dict[Operator, str] = {}
        # Each operator may have multiple outputs (e.g., Split).
        self.op_outputs_map: dict[Operator, list[ast.expr]] = {}
        self.graph_onnx_node_map: dict[Operator, AST] = {}
        self.pos_symbol_runtime_value_ast_map: dict[str, ast.AST] = {}
        self.inferrer: ConcatInfoInference
        self.tmp_var_counter = 0

    def plug_in_inferrer(self, inferrer: ConcatInfoInference) -> None:
        self.inferrer = inferrer

    def _clone_expr(self, node: ast.expr) -> ast.expr:
        return copy.deepcopy(node)

    def _new_temp_name(self, prefix: str) -> str:
        name = f"_{prefix}_{self.tmp_var_counter}"
        self.tmp_var_counter += 1
        return name

    def new_input_op_name(self) -> str:
        name = f"in_{self.input_op_counter}"
        self.input_op_counter += 1
        return name

    def _ensure_graph_node_registered(self, op: Operator, expr: ast.expr) -> None:
        if op not in self.graph_onnx_node_map:
            self.graph_onnx_node_map[op] = self._clone_expr(expr)

    def get_dim_runtime_value_ast(self, op: Operator, dim: int) -> AST:
        if op not in self.graph_onnx_node_map:
            expr = self._ensure_outputs(op)[0]
            self.graph_onnx_node_map[op] = self._clone_expr(expr)
        op_ast = self._clone_expr(self.graph_onnx_node_map[op])
        shape_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Shape",
                ctx=ast.Load(),
            ),
            args=[op_ast],  # type: ignore[arg-type]
            keywords=[],
        )
        index_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Constant",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(arg="value_int", value=ast.Constant(value=dim))
            ],
        )
        dim_len_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Gather",
                ctx=ast.Load(),
            ),
            args=[shape_ast, index_ast],
            keywords=[ast.keyword(arg="axis", value=ast.Constant(value=0))],
        )
        return dim_len_ast

    def _unsqueeze_dim_expr(self, expr: ast.expr) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Unsqueeze",
                ctx=ast.Load(),
            ),
            args=[expr],
            keywords=[
                ast.keyword(
                    arg="axes",
                    value=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                )
            ],
        )

    def split_to_onnx_pattern(self, gop: SplitOperator) -> ast.Call:
        (inp,) = gop.get_inputs()
        inp_onnx: AST = self.op_to_onnx_expr(inp, user=gop)
        inp_concat_info: ConcatInfo = self.inferrer.op_concat_info_map[inp][0]
        undone_concat: Operator = inp_concat_info.cocat_op
        assert isinstance(undone_concat, ConcatOperator)
        split_dim = inp_concat_info.concat_dim
        split_pos_symb: str = self.inferrer.op_dim_pos_symbol_map[
            OpDim(undone_concat.get_inputs()[0], split_dim)
        ]
        split_pos_ast: AST = self.pos_symbol_runtime_value_ast_map[split_pos_symb]
        second_len_ast = self.get_dim_runtime_value_ast(
            undone_concat.get_inputs()[1], split_dim
        )
        assert isinstance(split_pos_ast, ast.expr)
        assert isinstance(second_len_ast, ast.expr)
        split_pos_unsq_ast = self._unsqueeze_dim_expr(split_pos_ast)
        second_len_unsq_ast = self._unsqueeze_dim_expr(second_len_ast)
        splits_tensor_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Concat",
                ctx=ast.Load(),
            ),
            args=[split_pos_unsq_ast, second_len_unsq_ast],
            keywords=[
                ast.keyword(arg="axis", value=ast.Constant(value=0)),
            ],
        )
        split_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Split",
                ctx=ast.Load(),
            ),
            args=[inp_onnx, splits_tensor_ast],  # type: ignore[arg-type]
            keywords=[
                ast.keyword(arg="axis", value=ast.Constant(value=split_dim)),
                ast.keyword(arg="num_outputs", value=ast.Constant(value=2)),
                ast.keyword(arg="_outputs", value=ast.Constant(value=2)),
            ],
        )
        return split_ast

    def concat_to_onnx_pattern(self, gop: ConcatOperator) -> ast.Call:
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        this_concat_pos_symbol: str = self.inferrer.op_dim_pos_symbol_map[
            OpDim(lhs, gop.axis)
        ]
        this_concat_dim = gop.axis
        this_concat_pos_ast = self.get_dim_runtime_value_ast(lhs, this_concat_dim)
        self.pos_symbol_runtime_value_ast_map[this_concat_pos_symbol] = (
            this_concat_pos_ast
        )
        concat_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Concat",
                ctx=ast.Load(),
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore[arg-type]
            keywords=[
                ast.keyword(arg="axis", value=ast.Constant(value=gop.axis)),
            ],
        )
        return concat_ast

    def add_to_onnx_pattern(self, gop: AddOperator) -> ast.Call:
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        add_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Add",
                ctx=ast.Load(),
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore[arg-type]
            keywords=[],
        )
        return add_ast

    def matmul_to_onnx_pattern(self, gop: MatmulOperator) -> ast.Call:
        lhs, rhs = gop.get_inputs()
        lhs_onnx = self.op_to_onnx_expr(lhs, user=gop)
        rhs_onnx = self.op_to_onnx_expr(rhs, user=gop)
        matmul_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="MatMul",
                ctx=ast.Load(),
            ),
            args=[lhs_onnx, rhs_onnx],  # type: ignore[arg-type]
            keywords=[],
        )
        return matmul_ast
    
    def conv2d_to_onnx_pattern(self, gop: Conv2DOperator) -> ast.Call:
        features, weights = gop.get_inputs()
        features_onnx = self.op_to_onnx_expr(features, user=gop)
        weights_onnx = self.op_to_onnx_expr(weights, user=gop)
        conv_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.onnx_pattern_op_name, ctx=ast.Load()),
                attr="Conv",
                ctx=ast.Load(),
            ),
            args=[features_onnx, weights_onnx],  # type: ignore[arg-type]
            keywords=[
                ast.keyword(
                    arg="strides",
                    value=ast.List(elts=[ast.Constant(value=gop.stride),
                                         ast.Constant(value=gop.stride)],
                                   ctx=ast.Load()))]
        )
        return conv_ast

    def input_to_onnx_pattern(self, gop: InputOperator) -> ast.expr:
        if gop in self.input_ops_names_map:
            name = self.input_ops_names_map[gop]
        else:
            name = self.new_input_op_name()
            self.input_ops_names_map[gop] = name
        return ast.Name(id=name, ctx=ast.Load())

    def _build_split_outputs(self, gop: SplitOperator) -> list[ast.expr]:
        split_call = self.split_to_onnx_pattern(gop)
        tmp_name = self._new_temp_name("split")
        first_output = ast.Subscript(
            value=ast.NamedExpr(
                target=ast.Name(id=tmp_name, ctx=ast.Store()),
                value=split_call,
            ),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
        second_output = ast.Subscript(
            value=ast.Name(id=tmp_name, ctx=ast.Load()),
            slice=ast.Constant(value=1),
            ctx=ast.Load(),
        )
        return [first_output, second_output]

    def _build_op_outputs(self, gop: Operator) -> list[ast.expr]:
        match gop:
            case AddOperator():
                outputs = [self.add_to_onnx_pattern(gop)]
            case MatmulOperator():
                outputs = [self.matmul_to_onnx_pattern(gop)]
            case InputOperator():
                outputs = [self.input_to_onnx_pattern(gop)]
            case ConcatOperator():
                outputs = [self.concat_to_onnx_pattern(gop)]
            case Conv2DOperator():
                outputs = [self.conv2d_to_onnx_pattern(gop)]
            case SplitOperator():
                outputs = self._build_split_outputs(gop)
            case _:
                raise NotImplementedError(
                    f"ONNX pattern generation not implemented for {type(gop)}"
                )
        if outputs:
            self._ensure_graph_node_registered(gop, outputs[0])
        return outputs

    def _ensure_outputs(self, gop: Operator) -> list[ast.expr]:
        if gop not in self.op_outputs_map:
            self.op_outputs_map[gop] = self._build_op_outputs(gop)
        return [self._clone_expr(expr) for expr in self.op_outputs_map[gop]]

    def get_indexing_split_result_ast(
        self, split_ast: ast.AST, index: int
    ) -> ast.Subscript:
        assert isinstance(index, int)
        assert isinstance(split_ast, ast.expr)
        index_ast = ast.Constant(value=index)
        return ast.Subscript(
            value=split_ast,
            slice=index_ast,
            ctx=ast.Load(),
        )

    def op_to_onnx_expr(
        self, gop: Operator, user: Operator | None = None
    ) -> ast.AST:
        outputs = self._ensure_outputs(gop)
        if isinstance(gop, SplitOperator) and user is not None:
            user_index = gop.get_user_component(user)
            return outputs[user_index]
        if len(outputs) == 1:
            return outputs[0]
        return ast.Tuple(elts=outputs, ctx=ast.Load())

    def pattern_to_onnx_pattern(
        self, pattern_graph: Graph
    ) -> tuple[ast.AST, list[str]]:
        outputs_exprs: list[ast.expr] = []
        for out in pattern_graph.get_outputs():
            outputs_exprs.extend(self._ensure_outputs(out))
        if not outputs_exprs:
            raise ValueError("Pattern graph has no outputs")
        if len(outputs_exprs) == 1:
            outputs_ast: ast.AST = outputs_exprs[0]
        else:
            outputs_ast = ast.Tuple(elts=outputs_exprs, ctx=ast.Load())
        return outputs_ast, list(self.input_ops_names_map.values())

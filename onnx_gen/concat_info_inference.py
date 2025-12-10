from __future__ import annotations
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from typing import Any
from dataclasses import dataclass
import sympy
from sympy import Symbol
import proj_utils


@dataclass
class ConcatInfo:
    """
    concat_dim: When split, split along this dimension.
    split_pos: When generate onnx rules, instantiate this symbolic value to the
      split position represented in the AST of the dimension of tensor.
    Pure value, copyable.
    """
    concat_dim: int
    split_pos: Any  # sympy.Symbol. sympy has limited support for typing


class OpDim:
    def __init__(self, op: Operator, dim: int) -> None:
        self.op = op
        self.dim = dim
    
    def __eq__(self, value: object) -> bool:
        """
        Returns true if:
        - dims are the same int
        - ops are the same **instance**
        """
        if not isinstance(value, OpDim):
            return False
        return self.op == value.op and self.dim == value.dim


class ConcatInfoInference:
    """
    One new instance of this class for the inference of each graph pattern.
    """

    def __init__(self) -> None:
        # value: list of the effects of previous concats.
        # The last element is the most recent concat effect.
        self.op_concat_info_map: dict[Operator, list[ConcatInfo]] = {}
        # Map that records: (which operator, which dim) maps to which symbolic
        #   position.
        # Using string instead of sympy Symbol here to avoid the same instance?
        #   same content different instance? issue.
        self.op_dim_pos_symbol_map: dict[OpDim, str] = {}
        self.symbol_counter = 0

    def get_new_pos(self) -> str:
        ret = f'pos_{self.symbol_counter}'
        self.symbol_counter += 1
        return ret

    def infer_one_step(self, op: Operator) -> bool:
        """
        Return value:
        - True: inference is valid, the rule should be retained.
        - False: inference invalid (e.g., two inputs both have concat info),
            The rule should be discarded.
        """
        match op:
            case AddOperator():
                return self.infer_add_one_step(op)
            case MatmulOperator():
                return self.infer_matmul_one_step(op)
            case SplitOperator():
                return self.infer_split_one_step(op)
            case InputOperator():
                return True
            case ConcatOperator():
                return self.infer_concat_one_step(op)
            case _:
                raise NotImplementedError(
                    f'Concat info inference not implemented for {type(op)}'
                )

    def infer_add_one_step(self, op) -> bool:
        inputs = op.get_inputs()
        lhs_info, rhs_info = [self.op_concat_info_map.get(i) for i in inputs]
        if not lhs_info and not rhs_info:  # not l means l is None or empty.
            return True
        elif lhs_info and rhs_info:
            # <= 1 input of add operator allowed to have concat info (to be a
            #   result of concat).
            return False
        elif lhs_info:
            self.op_concat_info_map[op] = lhs_info
            return True
        else:  # rhs_info is not None and not empty.
            assert rhs_info is not None
            self.op_concat_info_map[op] = rhs_info
            return True

    def infer_matmul_one_step(self, op) -> bool:
        inputs = op.get_inputs()
        lhs_info, rhs_info = [self.op_concat_info_map.get(i) for i in inputs]
        proj_utils.todo()

    def infer_concat_one_step(self, op: ConcatOperator) -> bool:
        inputs = op.get_inputs()
        lhs_info, rhs_info = [self.op_concat_info_map.get(i) for i in inputs]
        lhs, rhs = inputs
        if lhs_info and rhs_info:
            return False
        elif not lhs_info and not rhs_info:
            pos_symb = self.get_new_pos()
            pos_sp = Symbol(pos_symb, integer=True, positive=True)
            op_dim_lhs = OpDim(lhs, op.axis)
            self.op_dim_pos_symbol_map[op_dim_lhs] = pos_symb
            this_op_info = ConcatInfo(concat_dim=op.axis, split_pos=pos_sp)
            self.op_concat_info_map[op] = [this_op_info]
            return True
        else:
            # has_info_in = lhs if lhs_info else rhs
            # no_info_in = rhs if lhs_info else lhs
            in_info = lhs_info if lhs_info else rhs_info
            assert in_info
            pos_symb = self.get_new_pos()
            pos_sp = Symbol(pos_symb, integer=True, positive=True)
            op_dim_lhs = OpDim(lhs, op.axis)
            self.op_dim_pos_symbol_map[op_dim_lhs] = pos_symb
            new_info = ConcatInfo(concat_dim=op.axis, split_pos=pos_sp)
            this_op_info = in_info + [new_info]
            self.op_concat_info_map[op] = this_op_info
            return True

    def infer_split_one_step(self, op) -> bool:
        x, = op.get_inputs()
        x_info = self.op_concat_info_map.get(x)
        if not x_info:
            # self.op_concat_info_map[op] = None
            # Split must counteract (undo) the effect of a previous concat.
            return False
        else:
            self.op_concat_info_map[op] = x_info[:-1]
            return True

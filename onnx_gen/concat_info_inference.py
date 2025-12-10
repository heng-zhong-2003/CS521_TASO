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
class FromConcat:
    """
    concat_dim: When split, split along this dimension.
    split_pos: When generate onnx rules, instantiate this symbolic value to the
      split position represented in the AST of the dimension of tensor.
    """
    concat_dim: int
    split_pos: Any  # sympy.Symbol. sympy has limited support for typing


type ConcatInfo = FromConcat | None


class ConcatInfoInference:
    """
    One new instance of this class for the inference of each graph pattern.
    """

    def __init__(self) -> None:
        self.op_concat_info_map: dict[Operator, ConcatInfo] = {}
        self.symbol_counter = 0

    def get_new_pos(self) -> Any:
        """Returns sympy.Symbol. sympy has limited support for typing"""
        ret = Symbol(f'pos_{self.symbol_counter}',
                     nonnegative=True, integer=True)
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
                proj_utils.todo()
            case InputOperator():
                return True
            case ConcatOperator():
                proj_utils.todo()
            case _:
                raise NotImplementedError(
                    f'Concat info inference not implemented for {type(op)}'
                )

    def infer_add_one_step(self, op) -> bool:
        inputs = op.get_inputs()
        lhs_info, rhs_info = [self.op_concat_info_map.get(i) for i in inputs]
        if lhs_info is None and rhs_info is None:
            self.op_concat_info_map[op] = None
            return True
        elif lhs_info is not None and rhs_info is not None:
            # <= 1 input of add operator allowed to have concat info (to be a
            #   result of concat).
            return False
        elif lhs_info is not None:
            self.op_concat_info_map[op] = lhs_info
            return True
        else: # rhs_info is not None
            self.op_concat_info_map[op] = rhs_info
            return True
    
    def infer_matmul_one_step(self, op) -> bool:
        inputs = op.get_inputs()
        lhs_info, rhs_info = [self.op_concat_info_map.get(i) for i in inputs]
        proj_utils.todo()

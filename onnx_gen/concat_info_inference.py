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
from collections import deque, defaultdict


@dataclass
class ConcatInfo:
    """
    concat_op:  The concat operator resulting in this concat info.
    concat_dim: When split, split along this dimension.
    split_pos: When generate onnx rules, instantiate this symbolic value to the
      split position represented in the AST of the dimension of tensor.
    Pure value, copyable.
    """
    cocat_op: Operator
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
        return self.op is value.op and self.dim == value.dim
    
    def __hash__(self) -> int:
        return hash((self.op, self.dim))
    
    def __repr__(self) -> str:
        return f'OpDim(op={self.op}, dim={self.dim})'


class ConcatInfoInference:
    """
    One new instance of this class for the inference of each graph pattern.
    Run infer_all() before: fingerprint, validate, pattern to onnx.
    """

    def __init__(self, rank: int, g: Graph) -> None:
        # value: list of the effects of previous concats.
        # The last element is the most recent concat effect.
        self.op_concat_info_map: dict[Operator, list[ConcatInfo]] = {}
        # Map that records: (which operator, which dim) maps to which symbolic
        #   position.
        # Using string instead of sympy Symbol here to avoid the same instance?
        #   same content different instance? issue.
        self.op_dim_pos_symbol_map: dict[OpDim, str] = {}
        self.symbol_counter = 0
        # Rank of tensors in the graph pattern. Affects only matmul.
        self.rank = rank
        self.graph = g
    
    # def infer_all(self) -> bool:
    #     current_depth = 0
    #     visited: set[Operator] = set()
    #     queue: deque[Operator] = deque(self.graph.get_inputs())
    #     while queue:
    #         curr_depth_size: int = len(queue)
    #         for _ in range(curr_depth_size):
    #             op = queue.popleft()
    #             if op in visited:
    #                 continue
    #             visited.add(op)
    #             print(f'Inferring op {op}')
    #             valid = self.infer_one_step(op)
    #             if not valid:
    #                 # print(f'Infer concat info invalid at op {op}')
    #                 return False
    #             for user in op.get_users():
    #                 if user not in visited:
    #                     queue.append(user)
    #         current_depth += 1
    #     return True


    def infer_all(self) -> bool:
        indeg = defaultdict(int)

        for op in self.graph.operators:
            for user in op.get_users():
                indeg[user] += 1

        queue = deque(op for op in self.graph.operators if indeg[op] == 0)

        while queue:
            op = queue.popleft()

            if not self.infer_one_step(op):
                return False

            for user in op.get_users():
                indeg[user] -= 1
                if indeg[user] == 0:
                    queue.append(user)

        return True
    
    def infer_concrete(self, op_shape_map: dict[Operator, tuple[int, ...]]) \
        -> dict[SplitOperator, int]:
        proj_utils.todo()

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
        if lhs_info and rhs_info:
            return False
        elif not lhs_info and not rhs_info:
            return True
        # Last two dimensions of matmul inputs are involved in the actual
        #   matmul.
        # The dimensions before are batching dimensions.
        if rhs_info and rhs_info[-1].concat_dim == self.rank - 1:
            self.op_concat_info_map[op] = rhs_info
            return True
        if lhs_info and lhs_info[-1].concat_dim == self.rank - 2:
            self.op_concat_info_map[op] = lhs_info
            return True

        return False

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
            this_op_info = ConcatInfo(
                cocat_op=op,
                concat_dim=op.axis,
                split_pos=pos_sp)
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
            new_info = ConcatInfo(
                cocat_op=op,
                concat_dim=op.axis,
                split_pos=pos_sp)
            this_op_info = in_info + [new_info]
            self.op_concat_info_map[op] = this_op_info
            return True

    def infer_split_one_step(self, op) -> bool:
        assert isinstance(op, SplitOperator)
        x, = op.get_inputs()
        x_info = self.op_concat_info_map.get(x)
        # print(f'Split op {op} input concat op {x} info: {x_info}')
        if not x_info:
            # self.op_concat_info_map[op] = None
            # Split must counteract (undo) the effect of a previous concat.
            return False
        else:
            self.op_concat_info_map[op] = x_info[:-1]
            op.axis = x_info[-1].concat_dim
            return True

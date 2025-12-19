from __future__ import annotations
from typing import Callable, Dict, Tuple
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from collections import defaultdict, deque
from onnx_gen.concat_info_inference import ConcatInfoInference
import proj_utils


class SplitPosInstantiation:
    def __init__(
            self,
            graph: Graph,
            op_shape_map: dict[Operator, Tuple[int, ...]],
            split_infer: ConcatInfoInference) -> None:
        self.op_shape_map = op_shape_map
        self.graph = graph
        self.split_infer = split_infer
        # Used by EvalGraph
        self.split_pos_map: dict[SplitOperator, int] = {}
        self.symb_int_map: dict[str, int] = {}

    def instantiate(self) -> dict[SplitOperator, int]:
        indeg = defaultdict(int)
        for op in self.graph.operators:
            for user in op.get_users():
                indeg[user] += 1
        queue = deque(op for op in self.graph.operators if indeg[op] == 0)
        while queue:
            op = queue.popleft()
            self.traverse_op(op)
            for user in op.get_users():
                indeg[user] -= 1
                if indeg[user] == 0:
                    queue.append(user)

        return self.split_pos_map
    
    def traverse_op(self, op: Operator) -> None:
        match op:
            case AddOperator():
                self.traverse_add(op)
            case MatmulOperator():
                self.traverse_matmul(op)
            case ConcatOperator():
                self.traverse_concat(op)
            case SplitOperator():
                self.traverse_split(op)
            case _:
                pass
    
    def traverse_add(self, op: AddOperator) -> None:
        pass

    def traverse_matmul(self, op: MatmulOperator) -> None:
        pass

    def traverse_concat(self, op: ConcatOperator) -> None:
        symb_infos = self.split_infer.op_concat_info_map[op]
        symb_info = symb_infos[-1]
        symb_pos = symb_info.split_pos
        symb_name = symb_pos.name
        lhs_shape = self.op_shape_map[op.get_inputs()[0]]
        int_pos = lhs_shape[symb_info.concat_dim]
        self.symb_int_map[symb_name] = int_pos
    
    def traverse_split(self, op: SplitOperator) -> None:
        in_op, = op.get_inputs()
        symb_infos = self.split_infer.op_concat_info_map[in_op]
        symb_info = symb_infos[-1]
        symb_pos = symb_info.split_pos
        symb_name = symb_pos.name
        int_pos = self.symb_int_map[symb_name]
        self.split_pos_map[op] = int_pos

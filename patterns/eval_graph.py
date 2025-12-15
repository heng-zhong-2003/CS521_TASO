from __future__ import annotations
from typing import Callable, Any, cast
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_input import InputOperator
import proj_utils
from collections import deque
import numpy as np
import numpy.typing as npt


class EvalGraph:
    def __init__(self,
                 comp_graph: Graph,
                 inputs: list[npt.NDArray[Any]],
                 split_op_pos_map: dict[SplitOperator, int]) -> None:
        """
        inputs: list[npt.NDArray[np.int32 | np.float64]]
        split_op_pos_map: Concrete split positions. Instantiatied from symbolic
          split positions inferred by concat_info_inference.py
        """
        self.comp_graph = comp_graph
        self.inputs = inputs
        # {op -> np array for other ops}
        # {op -> (split_0, split_1) (two np arrays) for SplitOperator}
        self.op_results_map: dict[Operator, Any] = {}
        input_ops = comp_graph.get_inputs()
        for i in range(len(input_ops)):
            self.op_results_map[input_ops[i]] = inputs[i]
        self.split_op_pos_map = split_op_pos_map
        self.split_set: set[SplitOperator] = set()
    
    def eval_graph(self) -> dict[Operator, Any]:
        """
        Returns map: {output op ->
            its result value np array for other ops
          | (split_0, split_1) for SplitOperator}
        """
        current_depth = 0
        visited: set[Operator] = set()
        queue: deque[Operator] = deque(self.comp_graph.get_inputs())
        while queue:
            curr_depth_size: int = len(queue)
            # print("in eval_graph")
            for _ in range(curr_depth_size):
                op = queue.popleft()
                if op in visited:
                    continue
                visited.add(op)
                self.eval_op(op)
                is_empty = True
                for user in op.get_users():
                    is_empty = False
                    if user not in visited:
                        queue.append(user)
            current_depth += 1
        graph_outputs = self.comp_graph.get_outputs()
        output_op_rslt_map: dict[Operator, Any] = {}

        for out_op in graph_outputs:
            output_op_rslt_map[out_op] = self.op_results_map[out_op]
        return output_op_rslt_map

    def eval_op(self, op: Operator) -> None:
        match op:
            case AddOperator():
                self.eval_add(op)
            case MatmulOperator():
                self.eval_matmul(op)
            case ConcatOperator():
                self.eval_concat(op)
            case SplitOperator():
                self.eval_split(op)
            case InputOperator():
                self.eval_input(op)
            case _:
                raise NotImplementedError()

    def aux_get_result_val(self, op: Operator, user: Operator) -> npt.NDArray[Any]:
        """Get the input values for `op`, handling SplitOperators."""
        if isinstance(op, SplitOperator):
            return self.op_results_map[op][
                op.get_user_component(user)
            ]
        else:
            return self.op_results_map[op]
    
    def eval_input(self, op: InputOperator) -> None:
        pass

    def eval_add(self, op: AddOperator) -> None:
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = lhs_val + rhs_val
        self.op_results_map[op] = this_op_val

    def eval_matmul(self, op: MatmulOperator) -> None:
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = np.matmul(lhs_val, rhs_val)
        self.op_results_map[op] = this_op_val

    def eval_split(self, op: SplitOperator) -> None:
        input_op, = op.get_inputs()
        input_val = self.aux_get_result_val(input_op, op)
        split_pos = 4 #self.split_op_pos_map[op]
        self.split_set.add(op)
        split_0, split_1 = np.split(
            input_val, [split_pos], axis=op.axis
        )
        self.op_results_map[op] = (split_0, split_1)

    def eval_concat(self, op: ConcatOperator) -> None:
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = np.concatenate([lhs_val, rhs_val], axis=op.axis)
        self.op_results_map[op] = this_op_val

from __future__ import annotations
from typing import Callable, Any
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
import proj_utils
from collections import deque
import numpy as np
import numpy.typing as npt


def get_operator_kind(op: Operator) -> str:
    match op:
        case AddOperator():
            return 'add'
        case MatmulOperator():
            return 'matmul'
        case _:
            return ''


def eval_add(inputs: list[npt.NDArray[np.int32 | np.float64]]) \
        -> npt.NDArray[Any]:
    return inputs[0] + inputs[1]


operator_eval_func_map: \
    dict[str, Callable[[list[npt.NDArray[np.int32 | np.float64]]],
                       npt.NDArray[np.int32 | np.float64]]] = {
        'add': eval_add,
    }


def evaluate(comp_graph: Graph,
             inputs: list[npt.NDArray[np.int32 | np.float64]]) \
        -> dict[Operator, npt.NDArray[np.int32 | np.float64]]:
    if (len(inputs) != len(comp_graph.get_inputs())):
        raise ValueError('`evaluate` must have the same '
                         'number of inputs as `comp_graph`.')
    visited: set[Operator] = set()
    queue: deque[Operator] = deque(comp_graph.get_inputs())
    operators_results_map: \
        dict[Operator, npt.NDArray[np.int32 | np.float64]] = {}
    for i in range(len(inputs)):
        operators_results_map[comp_graph.get_inputs()[i]] = inputs[i]
        visited.add(comp_graph.get_inputs()[i])
    curr_depth = 0
    output_op_rslt_map: dict[Operator, npt.NDArray[np.int32 | np.float64]] = {}
    while queue:
        curr_depth_size: int = len(queue)
        for _ in range(curr_depth_size):
            op: Operator = queue.popleft()
            if op in visited:
                continue
            visited.add(op)
            eval_inputs: list[npt.NDArray[np.int32 | np.float64]] = \
                [operators_results_map[ipt] for ipt in op.get_inputs()]
            op_rslt: npt.NDArray[np.int32 | np.float64] = \
                operator_eval_func_map[get_operator_kind(op)](eval_inputs)
            operators_results_map[op] = op_rslt
            is_empty = True
            for user in op.get_users():
                is_empty = False
                if user not in visited:
                    queue.append(user)
            if is_empty:
                output_op_rslt_map[op] = op_rslt
        curr_depth += 1
    return output_op_rslt_map

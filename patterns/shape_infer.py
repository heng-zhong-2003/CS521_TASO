from __future__ import annotations
from typing import Callable, Dict, Tuple
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_conv2d import Conv2DOperator
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
        case ConcatOperator():
            return 'concat'
        case SplitOperator():
            return 'split'
        case Conv2DOperator():
            return 'conv2d'
        case _:
            return ''

def infer_add(op, parent_shapes: list[tuple[int, ...]], shape_map: dict[Operator, tuple[int, ...]]) \
        -> tuple[int, ...]:
    lhs_shape, rhs_shape = parent_shapes
    if lhs_shape != rhs_shape:
        raise ValueError(f"Add rank mismatch: {lhs_shape} vs {rhs_shape}")
    return lhs_shape

def infer_matmul(op, parent_shapes: list[tuple[int, ...]], shape_map: dict[Operator, tuple[int, ...]]) \
        -> tuple[int, ...]:
    lhs_shape, rhs_shape = parent_shapes
    if lhs_shape != rhs_shape:
        raise ValueError(f"Matmul rank mismatch: {lhs_shape} vs {rhs_shape}")
    return lhs_shape

def infer_concat(op, parent_shapes: list[tuple[int, ...]], shape_map: dict[Operator, tuple[int, ...]]) \
        -> tuple[int, ...]:
    lhs_shape, rhs_shape = parent_shapes
    axis = op.axis

    # check
    rank = len(lhs_shape)
    if len(rhs_shape) != rank:
        raise ValueError(f"Concat rank mismatch: {lhs_shape} vs {rhs_shape}")
    for i in range(rank):
        if i == axis:
            continue
        if lhs_shape[i] != rhs_shape[i]:
            raise ValueError(
                f"Concat dim mismatch at dim {i}: {lhs_shape[i]} vs {rhs_shape[i]}"
            )

    out = list(lhs_shape)
    out[axis] = lhs_shape[axis] + rhs_shape[axis]
    return tuple(out)

def infer_split(op, parent_shapes, shape_map: dict[Operator, tuple[int, ...]]) \
        -> tuple[int, ...]:
    concat_op = op.splitted_concat
    lhs_shape = shape_map[concat_op.lhs]
    rhs_shape = shape_map[concat_op.rhs]
    return (lhs_shape, rhs_shape)

def infer_conv2d(op, parent_shapes, shape_map: dict[Operator, tuple[int, ...]]):
    input_shape, weight_shape = parent_shapes
    # Very simplified Conv2D shape inference: assumes stride=1, padding=0, dilation=1
    return (2, input_shape[1] - weight_shape[1] + 1, input_shape[2] - weight_shape[2] + 1)

shape_inference_map: \
    dict[str, Callable[[Operator, list[tuple[int, ...]], dict[Operator, tuple[int, ...]]],
                tuple[int, ...]]] = {
        'add': infer_add,
        'matmul': infer_matmul,
        'concat': infer_concat,
        'split': infer_split,
        'conv2d': infer_conv2d,
    }

def traverse(comp_graph: Graph,
             inputs: tuple[int, ...]) \
        -> dict[Operator, tuple[int, ...]]:
    if (len(inputs) != len(comp_graph.get_inputs())):
        raise ValueError('`infer_shapes` must have the same number of inputs as `comp_graph`.')
    visited: set[Operator] = set()
    queue: deque[Operator] = deque(comp_graph.get_inputs())
    shape_map: \
        dict[Operator, tuple[int, ...]] = {}

    # Map input operators
    for i in range(len(inputs)):
        shape_map[comp_graph.get_inputs()[i]] = inputs[i]

    if len(comp_graph.get_inputs()) == len(comp_graph.operators):
    # No other operators; graph outputs are just its inputs
        return {inp: shape for inp, shape in zip(comp_graph.get_inputs(), inputs)}

    curr_depth = 0

    while queue:
        curr_depth_size: int = len(queue)
        for _ in range(curr_depth_size):
            op: Operator = queue.popleft()
            ready = all(parent in shape_map for parent in op.get_inputs())

            # If there are dependencies between ops
            # Only compute shape when all parents have known shapes
            if not ready:
                queue.append(op)
                continue

            if op in visited:
                continue
            visited.add(op)

            if op not in shape_map:
                op_kind = get_operator_kind(op)
                infer_func = shape_inference_map[op_kind]

                # input_ops = op.get_inputs()
                # parent_shapes = [shape_map[ipt] for ipt in input_ops]
                # shape_map[op] = infer_func(op, parent_shapes, shape_map)

                # For normal operators, shape_map[op] = Shape (tuple[int, ...])
                # For SplitOperator, shape_map[op] = (Shape, Shape)  # two outputs
                # Downstream users must use SplitOperator.user_component_map[user] to
                # select which component they see as input.
                parent_shapes = []
                for parent in op.get_inputs():
                    if isinstance(parent, SplitOperator):
                        comp = parent.user_component_map[op]
                        shape0, shape1 = shape_map[parent]
                        parent_shapes.append((shape0, shape1)[comp])
                    else:
                        parent_shapes.append(shape_map[parent])
                shape_map[op] = infer_func(op, parent_shapes, shape_map)

            for user in op.get_users():
                if user not in visited:
                    queue.append(user)

        curr_depth += 1

    return shape_map

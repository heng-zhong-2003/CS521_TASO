from __future__ import annotations
from typing import Callable, Dict, Tuple
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
import shape_infer

# 检查代码逻辑上有没有问题。处理split & 运用split。给SplitOperator的input_op、splitted_concat赋值了所以看起来能跑。
# 可删。
def _build_toy_graph() -> tuple[Graph, tuple[tuple[int, ...], ...], list[Operator]]:
    """
    构造一个包含 Split 的计算图：
        in0 (2,3)  ---\
                       Concat c1 (axis=1, shape=(2,8)) ---> Split s1 (axis=1)
        in1 (2,5)  ---/                                       |
                                                              ├---> component 0: (2,3)
                                                              └---> component 1: (2,5)

    期望：
        c1.shape = (2, 8)
        s1.shape = ((2, 3), (2, 5))  # 两个输出
    """
    in0 = InputOperator()
    in1 = InputOperator()
    graph = Graph(inputs=[in0, in1])
    c1 = ConcatOperator(lhs=in0, rhs=in1, axis=1)
    s1 = SplitOperator(input_op=c1, axis=1, splitted_concat=c1)
    graph.operators.extend([c1, s1])

    input_shapes: tuple[tuple[int, ...], ...] = (
        (2, 3),  # in0
        (2, 5),  # in1
    )

    all_ops: list[Operator] = [in0, in1, c1, s1]
    return graph, input_shapes, all_ops

def _build_split_with_users_graph() -> tuple[Graph, tuple[tuple[int, ...], ...], list[Operator]]:
    """
    构造一个 Split 输出被使用的图：
        in0 (2,3)  ---\
                       Concat c1 (axis=1, shape=(2,8)) ---> Split s1 (axis=1)
        in1 (2,5)  ---/                                       |
                                                              ├---> a1 (用 component 0)
                                                              └---> a2 (用 component 1)
        in2 (2,3)  ------------------------------------------------^

        in3 (2,5)  ------------------------------------------------^

    期望：
        s1.shape = ((2, 3), (2, 5))
        a1.shape = (2, 3)  # s1[0] + in2
        a2.shape = (2, 5)  # s1[1] + in3
    """
    in0 = InputOperator()
    in1 = InputOperator()
    in2 = InputOperator()
    in3 = InputOperator()

    graph = Graph(inputs=[in0, in1, in2, in3])
    c1 = ConcatOperator(lhs=in0, rhs=in1, axis=1)
    s1 = SplitOperator(input_op=c1, axis=1, splitted_concat=c1)
    a1 = AddOperator(s1, in2)
    s1.user_component_map[a1] = 0  # a1 使用第 0 个输出
    a2 = AddOperator(s1, in3)
    s1.user_component_map[a2] = 1  # a2 使用第 1 个输出

    graph.operators.extend([c1, s1, a1, a2])

    input_shapes: tuple[tuple[int, ...], ...] = (
        (2, 3),  # in0
        (2, 5),  # in1
        (2, 3),  # in2
        (2, 5),  # in3
    )

    all_ops: list[Operator] = [in0, in1, in2, in3, c1, s1, a1, a2]
    return graph, input_shapes, all_ops

def _debug_traverse() -> None:
    graph, input_shapes, all_ops = _build_toy_graph()
    shape_map = shape_infer.traverse(graph, input_shapes)

    print("=== shape_map ===")
    for op in all_ops:
        shp = shape_map.get(op, None)
        print(f"{type(op).__name__}: {shp}")


def _debug_split_with_users() -> None:
    graph, input_shapes, all_ops = _build_split_with_users_graph()
    shape_map = shape_infer.traverse(graph, input_shapes)

    print("=== shape_map ===")
    for op in all_ops:
        shp = shape_map.get(op, None)
        print(f"{type(op).__name__}: {shp}")

if __name__ == "__main__":
    print("### Test 1: Basic Concat + Add ###")
    _debug_traverse()

    print("\n### Test 2: Split with Users ###")
    _debug_split_with_users()

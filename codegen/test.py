# test_codegen.py
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator  # 假设你有输入节点
from patterns.graph import Graph
from codegen.codegen import Codegen
import proj_utils


def test_add():
    a = InputOperator()
    b = InputOperator()
    c = InputOperator()
    ab = AddOperator(a, b)
    result = AddOperator(ab, c)

    # Source: ((a + b) + c)
    source_graph = Graph(inputs=[a, b, c])
    source_graph.outputs = [result]

    # Target: (a + (b + c))
    bc = AddOperator(b, c)
    new = AddOperator(a, bc)

    target_graph = Graph(inputs=[a, b, c])
    target_graph.outputs = [new]

    codegen = Codegen()
    cpp_code = codegen.generate(source_graph, target_graph)

    print("=== Test Add ===")
    print(cpp_code)
    print()


def test_matmul():
    a = InputOperator()
    b = InputOperator()
    c = MatmulOperator(a, b)

    source_graph = Graph(inputs=[a, b])
    source_graph.outputs = [c]

    target_graph = Graph(inputs=[a, b])
    target_graph.outputs = [c]

    codegen = Codegen()
    cpp_code = codegen.generate(source_graph, target_graph)

    print("=== Test Matmul ===")
    print(cpp_code)
    print()


if __name__ == "__main__":
    test_add()
    test_matmul()
    print("All tests passed!")
from __future__ import annotations
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
# from patterns.evaluate import get_operator_kind

def get_operator_kind(op: Operator) -> str:
    match op:
        case AddOperator():
            return 'add'
        case MatmulOperator():
            return 'matmul'
        case _:
            return ''

class Graph:
    def __init__(self, inputs: list[InputOperator]) -> None:
        self.inputs: list[InputOperator] = inputs

        # BHsketch --- maintain a list of operators so it's easy to check for duplicates
        # initialized also, with just the inputs
        self.operators : list[Operator] = inputs
    
    def get_inputs(self) -> list[InputOperator]:
        return self.inputs

    def add_operator(self, op: Operator):
        self.operators.append(op)
        for i in op.get_inputs():
            i.add_users([op])

    def remove_operator(self, op: Operator):
        for operator in self.operators:
            if(operator == op):
                for i in operator.get_inputs():
                    i.remove_user(op)
                self.operators.remove(op)
                return

    def check_duplicates(self, op:Operator):
        for operator in self.operators:
            if(get_operator_kind(operator) == get_operator_kind(op)):
                # if operator kind matches, match each input
                inputs1 = list(operator.get_inputs())
                inputs2 = list(op.get_inputs())
                for i in range (0, len(inputs1)):
                    if(inputs1[i] != inputs2[i]):
                        return True # as in, yes there are duplicates
        return False # as in, there are no duplicates
                    
    def copy(self) -> Graph:
        """
        Deep-copy the graph structure, but reuse the same InputOperator objects.
        """
        # Map original -> copied operator
        op_map: dict[Operator, Operator] = {}

        # 1️⃣ Create new Graph with the same input objects (inputs are shared)
        new_graph = Graph(list(self.inputs))

        # 2️⃣ Copy operators in topological order (inputs already first)
        for op in self.operators:
            if isinstance(op, InputOperator):
                op_map[op] = op  # reuse
                continue

            # Remap this operator’s inputs to their copied versions
            copied_inputs = [op_map[i] for i in op.get_inputs()]
            op_class = type(op)
            new_op = op_class(copied_inputs)

            new_graph.add_operator(new_op)
            op_map[op] = new_op

        return new_graph



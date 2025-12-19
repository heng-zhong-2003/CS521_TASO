from __future__ import annotations
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_conv2d import Conv2DOperator
import copy
# from patterns.evaluate import get_operator_kind


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
        case InputOperator():
            return 'inputop'
    return 'nomatch'

class Graph:
    def __init__(self, inputs: list[InputOperator]) -> None:
        self.inputs: list[InputOperator] = inputs

        # BHsketch --- maintain a list of operators so it's easy to check for duplicates
        # initialized also, with just the inputs
        self.operators : list[Operator] = copy.copy(inputs) # type: ignore
    
    def get_inputs(self) -> list[InputOperator]:
        return self.inputs

    def get_outputs(self) -> list[Operator]:
        rslt: list[Operator] = []
        for op in self.operators:
            if all(False for _ in op.get_users()):
                rslt.append(op)
        return rslt

    def add_operator(self, op: Operator):
        self.operators.append(op)
        for i in op.get_inputs():
            i.add_users([op])

    def remove_operator(self, op: Operator):
        for i in op.get_inputs():
            print("     removed as user from an input")
            if(op in i.get_users()):
                # we need to check because in the case where an operators both inputs are the same operator (different outputs of a split),
                # this code will try to remove it twice from the users list
                i.remove_user(op)
        self.operators.remove(op)
        print("operators now has ", len(self.operators), " elements")
        # for operator in self.operators:
            # if(operator == op):
                # print("     inside remove_operator")
                # for i in op.get_inputs():
                    # print("     removed as user from an input")
                    # i.remove_user(op)
                # self.operators.remove(op)
                # return

    def check_duplicates(self, op:Operator):
        for operator in self.operators:
            areSame = True
            if(get_operator_kind(operator) == get_operator_kind(op)):
                # if operator kind matches, match each input
                inputs1 = list(operator.get_inputs())
                inputs2 = list(op.get_inputs())
                for i in range (0, len(inputs1)):
                    if(inputs1[i] != inputs2[i]):
                        areSame = False # as in, yes there are duplicates
                    elif(isinstance(inputs1[i], SplitOperator)):
                        if (inputs1[i].get_user_component(op) != inputs1[i].get_user_component(operator)):
                            areSame = False
            else:
                areSame = False
            if (areSame):
                return True
        return False # as in, there are no duplicates
                    
    def copy(self) -> Graph:
        """
        Deep-copy the graph structure, but reuse the same InputOperator objects.
        """
        op_map: dict[Operator, Operator] = {}

        # 1️⃣ Copy input operators
        new_inputs = []
        for inp in self.inputs:
            new_inp = InputOperator()           # create a fresh InputOperator
            new_inp.users = []                  # no users yet
            op_map[inp] = new_inp
            new_inputs.append(new_inp)

        new_graph = Graph(new_inputs)

        # 2️⃣ Copy remaining operators (topologically ordered)
        for old_op in self.operators:
            if isinstance(old_op, InputOperator):
                continue  # already handled

            # Remap inputs to their new copies
            copied_inputs = [op_map[i] for i in old_op.get_inputs()]

            # Create operator of same type
            if isinstance(old_op, AddOperator):
                new_op = AddOperator(copied_inputs[0], copied_inputs[1])

            elif isinstance(old_op, MatmulOperator):
                new_op = MatmulOperator(copied_inputs[0], copied_inputs[1])

            elif isinstance(old_op, ConcatOperator):
                new_op = ConcatOperator(copied_inputs[0], copied_inputs[1], axis=old_op.axis)

            elif isinstance(old_op, SplitOperator):
                new_op = SplitOperator(copied_inputs[0], axis=old_op.axis)
                new_op.user_component_map = dict(old_op.user_component_map)

            elif isinstance(old_op, Conv2DOperator):
                new_op = Conv2DOperator(copied_inputs[0], copied_inputs[1], stride=old_op.stride)

            else:
                raise TypeError(f"Unknown operator type {type(old_op)} in Graph.copy()")

            new_graph.add_operator(new_op)
            op_map[old_op] = new_op

        # 3️⃣ Return the new independent graph
        return new_graph





        # Map original -> copied operator
        # op_map: dict[Operator, Operator] = {}

        # # Create new Graph with the same input objects (inputs are shared)
        # new_graph = Graph(list(self.inputs))

        # # Copy operators in topological order (inputs already first)
        # for op in self.operators:
            # if isinstance(op, InputOperator):
                # op_map[op] = op  # reuse
                # continue

            # # Remap this operator’s inputs to their copied versions
            # copied_inputs = [op_map[i] for i in op.get_inputs()]
            # op_class = type(op)
            # # new_op = op_class(copied_inputs)

            # # Handle special cases
            # if isinstance(op, SplitOperator):
                # # SplitOperator takes input_op and axis
                # copied_op = SplitOperator(copied_inputs[0], axis=op.axis)
                # # If needed, copy user_component_map
                # copied_op.user_component_map = dict(op.user_component_map)

            # elif isinstance(op, ConcatOperator):
                # # ConcatOperator takes lhs, rhs, axis
                # copied_op = ConcatOperator(copied_inputs[0], copied_inputs[1], axis=op.axis)

            # else:
                # # Default: assume constructor takes list of inputs
                # copied_op = op_class(copied_inputs)


            # new_graph.add_operator(copied_op)
            # op_map[op] = copied_op

        # return new_graph

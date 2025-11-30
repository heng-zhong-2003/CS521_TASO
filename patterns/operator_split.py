from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class SplitOperator(Operator):
    def __init__(self, input_op: Operator, axis: int, splits: list[int]) -> None:
        """
        axis: split along the axis `axis`.
        splits: list of lengths of each slice, along `axis`.
        """
        self.input_op = input_op
        self.axis = axis
        self.splits = splits
        self.users: list[Operator] = []
        self.input_op.add_users([self])

    def get_inputs(self) -> list[Operator]:
        return [self.input_op]

    def get_users(self) -> list[Operator]:
        return self.users
    
    def add_users(self, new_users: Iterable[Operator]) -> None:
        for usr in new_users:
            if usr not in self.users:
                self.users.append(usr)

    def remove_user(self, op: Operator) -> None:
        self.users.remove(op)

from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class ConcatOperator(Operator):
    def __init__(self, lhs: Operator, rhs: Operator, axis: int) -> None:
        """
        axis: concatenate along the axis `axis`.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.axis = axis
        self.users: list[Operator] = []
        self.lhs.add_users([self])
        self.rhs.add_users([self])

    def get_inputs(self) -> list[Operator]:
        return [self.lhs, self.rhs]

    def get_users(self) -> list[Operator]:
        return self.users
    
    def add_users(self, new_users: Iterable[Operator]) -> None:
        for usr in new_users:
            if usr not in self.users:
                self.users.append(usr)

    def remove_user(self, op: Operator) -> None:
        self.users.remove(op)
    
    @classmethod
    def get_arity(cls) -> int:
        return 2

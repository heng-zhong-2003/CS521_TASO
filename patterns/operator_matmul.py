from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class MatmulOperator(Operator):
    def __init__(self, lhs: Operator, rhs: Operator) -> None:
        self.lhs: Operator = lhs
        self.rhs: Operator = rhs
        self.users: list[Operator] = []
        lhs.add_users([self])
        rhs.add_users([self])
    
    def get_inputs(self) -> Iterable[Operator]:
        return [self.lhs, self.rhs]

    def get_users(self) -> Iterable[Operator]:
        return self.users
    
    def add_users(self, new_users: Iterable[Operator]) -> None:
        for usr in new_users:
            if usr not in self.users:
                self.users.append(usr)

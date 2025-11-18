from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class AddOperator(Operator):
    def __init__(self, lhs: Operator | list[Operator], rhs: Operator | None = None) -> None:
        # Normalize inputs
        if isinstance(lhs, list):
            if len(lhs) != 2:
                raise ValueError(f"AddOperator expects 2 inputs, got {len(lhs)}")
            self.lhs, self.rhs = lhs
        else:
            if rhs is None:
                raise ValueError("AddOperator needs both lhs and rhs or a list of 2 inputs")
            self.lhs, self.rhs = lhs, rhs

        self.users: list[Operator] = []
        self.lhs.add_users([self])
        self.rhs.add_users([self])

    # def __init__(self, lhs: Operator, rhs: Operator) -> None:
        # self.lhs: Operator = lhs
        # self.rhs: Operator = rhs
        # self.users: list[Operator] = []
        # # self.arity: int = 2
        # lhs.add_users([self])
        # rhs.add_users([self])
    
    def get_inputs(self) -> Iterable[Operator]:
        return [self.lhs, self.rhs]

    def get_users(self) -> Iterable[Operator]:
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

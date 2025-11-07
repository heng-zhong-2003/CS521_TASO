from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class InputOperator(Operator):
    def __init__(self) -> None:
        self.users: list[Operator] = []

    def get_inputs(self) -> Iterable[Operator]:
        return []
    
    def get_users(self) -> Iterable[Operator]:
        return self.users
    
    def add_users(self, new_users: Iterable[Operator]) -> None:
        for usr in new_users:
            if usr not in self.users:
                self.users.append(usr)

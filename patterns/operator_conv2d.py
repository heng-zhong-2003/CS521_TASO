from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class Conv2DOperator(Operator):
    def __init__(self, inputOp: Operator, weightOp: Operator, stride: int = 1) -> None:
        self.inputOp: Operator = inputOp
        self.weightOp: Operator = weightOp
        self.users: list[Operator] = []
        self.stride: int = stride
        # lhs.add_users([self])
        # rhs.add_users([self])


    
    def get_inputs(self) -> Iterable[Operator]:
        return [self.inputOp, self.weightOp]

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

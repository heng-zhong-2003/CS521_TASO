from __future__ import annotations
from typing import Iterable
from patterns.operator_interface import Operator


class SplitOperator(Operator):
    def __init__(self, input_op: Operator, axis: int, splitted_concat: Operator) -> None:
        """
        axis: split along this dimension. This parameter at init is just a
            placeholder for ease in testing.
        The actual split position is inferred by concat_info_inference.py and
            stored in its map and also here self.axis.
        splitted_concat: This split "undos" the effect of op `splitted_concat`.
            This is an embedded implementation of the split tree in the paper.
        """
        self.input_op = input_op
        self.splitted_concat = splitted_concat
        self.users: list[Operator] = []
        self.axis = axis
        self.input_op.add_users([self])
        # Map {user_operator -> component}
        # The user is using the 0th or 1st output of split.
        self.user_component_map: dict[Operator, int] = {}

    def get_inputs(self) -> list[Operator]:
        return [self.input_op]

    def get_users(self) -> list[Operator]:
        return self.users

    def add_users(self, new_users: Iterable[Operator]) -> None:
        for usr in new_users:
            if usr not in self.users:
                self.users.append(usr)

    def add_user_component(self, user: Operator, component: int) -> None:
        self.user_component_map[user] = component

    def get_user_component(self, user: Operator) -> int:
        """When calling this function, `user` should exist in the map."""
        return self.user_component_map[user]

    def remove_user(self, op: Operator) -> None:
        self.users.remove(op)
        if op in self.user_component_map:
            del self.user_component_map[op]
    
    @classmethod
    def get_arity(cls) -> int:
        return 1

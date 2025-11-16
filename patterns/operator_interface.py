from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable


class Operator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_inputs(self) -> Iterable[Operator]:
        pass

    @abstractmethod
    def get_users(self) -> Iterable[Operator]:
        pass

    @abstractmethod
    def add_users(self, new_users: Iterable[Operator]) -> None:
        pass

    @abstractmethod
    def remove_user(self, op: Operator) -> None:
        pass

    @classmethod
    @abstractmethod
    def get_arity(cls) -> int:
        pass

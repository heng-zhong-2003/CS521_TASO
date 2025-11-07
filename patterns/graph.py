from __future__ import annotations
from patterns.operator_interface import Operator
from patterns.operator_input import InputOperator


class Graph:
    def __init__(self, inputs: list[InputOperator]) -> None:
        self.inputs: list[InputOperator] = inputs

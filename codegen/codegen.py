from __future__ import annotations
from typing import Callable
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.graph import Graph
import proj_utils


class Codegen:
    def __init__(self) -> None:
        # Will need to maintain a map from operator nodes here to matched
        #   HloInstruction * names in C++.
        proj_utils.todo()

    def generate(self, source_graph: Graph, target_graph: Graph) -> str:
        proj_utils.todo()
    
    def generate_prolog(self) -> str:
        proj_utils.todo()

    def generate_epilog(self) -> str:
        proj_utils.todo()
    
    def generate_match(self) -> str:
        proj_utils.todo()

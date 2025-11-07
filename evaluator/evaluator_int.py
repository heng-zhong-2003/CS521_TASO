from __future__ import annotations
from evaluator.evaluator_interface import Evaluator
from patterns.graph import Graph
import proj_utils
import numpy as np
import numpy.typing as npt


class IntEvaluator(Evaluator):
    def __init__(self) -> None:
        pass

    def evaluate(self, comp_graph: Graph,
                 inputs: list[npt.NDArray[np.int32 | np.float64]]) \
                    -> list[np.int32 | np.float64]:
        proj_utils.todo()

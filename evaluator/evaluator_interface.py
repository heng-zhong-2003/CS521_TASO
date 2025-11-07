from __future__ import annotations
from abc import ABC, abstractmethod
from patterns.graph import Graph
import numpy as np
import numpy.typing as npt


class Evaluator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, comp_graph: Graph,
                 inputs: list[npt.NDArray[np.int32 | np.float64]]) \
                    -> list[np.int32 | np.float64]:
        pass

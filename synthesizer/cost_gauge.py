import time
import numpy as np
import numpy.typing as npt
from patterns.graph import Graph
from patterns.evaluate import evaluate


class CostGauge:
    """
    Measure execution time of graph patterns on fixed random inputs.
    """
    def __init__(self) -> None:
        self.fixed_inputs: list[npt.NDArray[np.float64]] = [
            np.random.uniform(low=-10.0, high=10.0, size=(128, 128))
            for _ in range(10)
        ]

    def get_cost(self, graph: Graph) -> float:
        """
        Measure execution time of a graph in seconds.
        """
        graph_inputs = graph.get_inputs()
        inputs = self.fixed_inputs[0:len(graph_inputs)]

        start = time.perf_counter()
        _ = evaluate(graph, inputs)
        elapsed = time.perf_counter() - start

        return elapsed

from __future__ import annotations
from patterns import evaluate
from patterns.graph import Graph
from patterns.operator_input import InputOperator
import numpy as np
import numpy.typing as npt
import functools
import proj_utils


class RuleValidator():
    """
    When program starts, create one object of this class and use consistently
    """

    def __init__(self, eq_threshold: float = 1.0e-4) -> None:
        self.eq_threshold: float = eq_threshold
        self.zeros: npt.NDArray[np.float64] = np.zeros(
            (2, 4, 4), dtype=np.float64)
        self.rand_inputs: list[npt.NDArray[np.float64]] = [
            np.random.uniform(low=-10.0, high=10.0, size=(2, 4, 4))
            for _ in range(10)
        ]
        self.ones: npt.NDArray[np.float64] = np.ones(
            (2, 4, 4), dtype=np.float64)

    def eval_eq(self, lhs: Graph, rhs: Graph, inputs: list[npt.NDArray[np.float64]]) -> bool:
        lhs_inputs: list[InputOperator] = lhs.get_inputs()
        lhs_rslt = evaluate.evaluate(lhs, inputs)  # type: ignore
        rhs_rslt = evaluate.evaluate(rhs, inputs)  # type: ignore
        lhs_rslt_vals = sorted(list(lhs_rslt.values()))
        rhs_rslt_vals = sorted(list(rhs_rslt.values()))
        all_eq = True
        for (lv, rv) in zip(lhs_rslt_vals, rhs_rslt_vals):
            all_eq = bool(all_eq and np.all(
                np.isclose(lv, rv, atol=self.eq_threshold)))
        return all_eq

    def validate(self, lhs: Graph, rhs: Graph) -> bool:
        lhs_inputs: list[InputOperator] = lhs.get_inputs()
        rhs_inputs: list[InputOperator] = rhs.get_inputs()
        if (len(lhs_inputs) != len(rhs_inputs)):
            return False
        if not self.eval_eq(lhs, rhs, self.rand_inputs[0:len(lhs_inputs)]):
            return False
        if not self.eval_eq(lhs, rhs, [self.zeros for _ in lhs_inputs]):
            return False
        if not self.eval_eq(lhs, rhs, [self.zeros for _ in lhs_inputs]):
            return False
        return True

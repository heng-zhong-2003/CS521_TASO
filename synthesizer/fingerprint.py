from __future__ import annotations
from patterns import evaluate
from patterns.graph import Graph
from patterns.operator_interface import Operator
import proj_utils
from collections import Counter
import hashlib
import numpy as np
import numpy.typing as npt


class Fingerprint:
    """
    When program starts, create one object of this class and use consistently
    """

    def __init__(self) -> None:
        self.inputs: list[npt.NDArray[np.int32]] = [
            np.random.randint(0, 100, size=(2, 4, 4), dtype=np.int32)
            for _ in range(10)
        ]

    def fingerprint(self, comp_graph: Graph) -> int:
        # Have to ignore type checking because mypy is not covariant
        #   for container element types.
        # That is, t1 is subtype of t2 does not mean list[t1] is
        #   subtype of list[t2].
        print("about to evaluate")
        rslts: dict[Operator, npt.NDArray[np.int32 | np.float64]] = \
            evaluate.evaluate(
                comp_graph,
                self.inputs[0:len(comp_graph.get_inputs())])  # type: ignore
        print("initialized rslts to a dict of", len(rslts))
        rslts_list: list[npt.NDArray[np.int32 |
                                     np.float64]] = list(rslts.values())
        print("initialized rslts_list")
        if rslts_list[0].dtype != np.int32:
            raise TypeError('Graph evaluation results not np.int32 '
                            'when computing fingerprint.')
        print("got rslts_list")
        for r in rslts_list:
            print(r)
        return self.hash_tensor_set(rslts_list)  # type: ignore

    def hash_tensor(self, tensor: npt.NDArray[np.int32]) -> int:
        h = hashlib.sha256()
        h.update(tensor.shape.__repr__().encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.tobytes())
        return int.from_bytes(h.digest(), 'big')

    def hash_tensor_set(self, tensor_list: list[npt.NDArray[np.int32]]) -> int:
        hashes: list[int] = [self.hash_tensor(t) for t in tensor_list]
        cnt: Counter[int] = Counter(hashes)
        fs: frozenset[tuple[int, int]] = frozenset(cnt.items())
        return hash(fs)

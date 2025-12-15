from __future__ import annotations
from patterns import evaluate
from patterns.eval_graph import EvalGraph
from patterns.graph import Graph
from patterns.operator_interface import Operator
from onnx_gen.concat_info_inference import ConcatInfoInference
from patterns.operator_split import SplitOperator
import proj_utils
from collections import Counter
import hashlib
import numpy as np
from typing import Any
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

    def fingerprint(self,
                    comp_graph: Graph) -> int:
        evaluator = EvalGraph(comp_graph, self.inputs, {} )
        rslts: dict[Operator, Any] = evaluator.eval_graph()
        rslts_list: list[Any] = []
        for out_op, val in rslts.items():
            match val:
                case (s0, s1):
                    assert isinstance(out_op, SplitOperator)
                    rslts_list.extend([s0, s1])
                case x:
                    rslts_list.append(x)
        if rslts_list[0].dtype != np.int32:
            raise TypeError('Graph evaluation results not np.int32 '
                            'when computing fingerprint.')
        return self.hash_tensor_set(rslts_list)

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

from __future__ import annotations
from typing import Callable, Any, cast
from patterns.graph import Graph
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_input import InputOperator
from patterns.operator_conv2d import Conv2DOperator
import proj_utils
from collections import deque
import numpy as np
import numpy.typing as npt


class EvalGraph:
    def __init__(self,
                 comp_graph: Graph,
                 inputs: list[npt.NDArray[Any]],
                 split_op_pos_map: dict[SplitOperator, int]) -> None:
        """
        inputs: list[npt.NDArray[np.int32 | np.float64]]
        split_op_pos_map: Concrete split positions. Instantiatied from symbolic
          split positions inferred by concat_info_inference.py
        """
        self.comp_graph = comp_graph
        self.inputs = inputs
        # {op -> np array for other ops}
        # {op -> (split_0, split_1) (two np arrays) for SplitOperator}
        self.op_results_map: dict[Operator, Any] = {}
        input_ops = comp_graph.get_inputs()
        print("inside eval init. inputs has ", len(inputs), " elements, and graph has ", len(input_ops), " elements")
        for i in range(len(input_ops)):
            self.op_results_map[input_ops[i]] = inputs[i]
        self.split_op_pos_map = split_op_pos_map
        self.split_set: set[SplitOperator] = set()
    
    def eval_graph(self) -> dict[Operator, Any]:
        """
        Returns map: {output op ->
            its result value np array for other ops
          | (split_0, split_1) for SplitOperator}
        """
        print("starting eval")
        current_depth = 0
        visited: set[Operator] = set()

        indegree: dict[Operator, int] = {}

        for op in self.comp_graph.operators:
            indegree[op] = len(op.get_inputs())
        queue: deque[Operator] = deque(self.comp_graph.get_inputs())

        while queue:
            # curr_depth_size: int = len(queue)
            # print("in eval_graph")
            op = queue.popleft()
            if op in visited:
                continue
            visited.add(op)

            self.eval_op(op)
            print("processed ", type(op), " with ", len(op.get_users()), " users")
            
            for user in op.get_users():
                print("adding new user to queue")
                if user not in indegree:
                    print("user of type ", type(user)," not in indegree dict")
                indegree[user] -= 1
                print("adding new user to queue 2")
                if indegree[user] == 0:
                    queue.append(user)
                print("done adding new user to queue")

        graph_outputs = self.comp_graph.get_outputs()
        output_op_rslt_map: dict[Operator, Any] = {}

        for out_op in graph_outputs:
            if out_op not in self.op_results_map:
                raise RuntimeError(f"Output operator {out_op} was never evaluated.")
            output_op_rslt_map[out_op] = self.op_results_map[out_op]

        print("     done evaluating queue. Graph has:")
        for op in self.comp_graph.operators:
            print("     ", type(op), " with ", len(op.get_users()), " users")
        print("returning results from eval")
        return output_op_rslt_map

    def eval_op(self, op: Operator) -> None:
        print("begin eval op")
        try:
            match op:
                case AddOperator():
                    self.eval_add(op)
                case MatmulOperator():
                    self.eval_matmul(op)
                case ConcatOperator():
                    self.eval_concat(op)
                case SplitOperator():
                    self.eval_split(op)
                case InputOperator():
                    self.eval_input(op)
                case Conv2DOperator():
                    self.eval_conv2d(op):
                case _:
                    raise NotImplementedError()
        except KeyError as e:
            print(f"❌ KeyError while evaluating {type(op).__name__}: {e}")
            print("  Inputs:", [type(i).__name__ for i in op.get_inputs()])
            raise
        print("end eval op")

    def aux_get_result_val(self, op: Operator, user: Operator) -> npt.NDArray[Any]:
        """Get the input values for `op`, handling SplitOperators."""
        if op not in self.op_results_map:
            print("⚠️ aux_get_result_val missing key:", op, type(op), "inputs:", op.get_inputs())
            raise KeyError(op)
        if isinstance(op, SplitOperator):
            return self.op_results_map[op][
                op.get_user_component(user)
            ]
        else:
            return self.op_results_map[op]
    
    def eval_input(self, op: InputOperator) -> None:
        pass

    def eval_add(self, op: AddOperator) -> None:
        print("begin add eval")
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = lhs_val + rhs_val
        self.op_results_map[op] = this_op_val
        print("end add eval")

    def eval_matmul(self, op: MatmulOperator) -> None:
        print("begin matmul eval")
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = np.matmul(lhs_val, rhs_val)
        self.op_results_map[op] = this_op_val
        print("end matmul eval")

    def eval_split(self, op: SplitOperator) -> None:
        print("begin split eval")
        input_op, = op.get_inputs()
        input_val = self.aux_get_result_val(input_op, op)
        split_pos = 4 #self.split_op_pos_map[op]
        self.split_set.add(op)
        split_0, split_1 = np.split(
            input_val, [split_pos], axis=op.axis
        )
        self.op_results_map[op] = (split_0, split_1)
        print("end split eval")

    def eval_concat(self, op: ConcatOperator) -> None:
        print("begin concat eval")
        lhs, rhs = op.get_inputs()
        lhs_val = self.aux_get_result_val(lhs, op)
        rhs_val = self.aux_get_result_val(rhs, op)
        this_op_val = np.concatenate([lhs_val, rhs_val], axis=op.axis)
        self.op_results_map[op] = this_op_val
        print("end concat eval")

    def eval_conv2d(self, op: Conv2DOperator) -> None:
        input_op, weight_op = op.get_inputs()
        inp_val = self.aux_get_result_val(input_op, op)
        w_val = self.aux_get_result_val(weight_op, op)

        ## make sure the shapes are appropriate for both

        if inp_val.ndim == 3:
            inp_val = inp_val[:, np.newaxis, :, :]   # shape becomes (1, C, H, W)
        elif inp_val.ndim == 2:
            inp_val = inp_val[np.newaxis, np.newaxis, ...] # shape becomes (1, 1, H, W)

        # If weight has no input-channel dimension, assume it’s single-channel
        if w_val.ndim == 2: # [KH, KW]
            w_val = w_val[np.newaxis, np.newaxis, :, :]  # [1, 1, KH, KW]
        elif w_val.ndim == 3:   # [C_out, KH, KW]
            w_val = w_val[:, np.newaxis, :, :]           # [C_out, 1, KH, KW]
        elif w_val.ndim != 4:
            raise ValueError(f"Unexpected weight shape for Conv2D: {w_val.shape}")

        # Assuming shapes: input [N, C_in, H, W], weight [C_out, C_in, KH, KW]
        stride = op.stride

        B, C_in, H, W = inp_val.shape
        C_out, Cw, KH, KW = w_val.shape
        assert C_in == Cw, f"Conv2D channel mismatch: input {C_in}, weight {Cw}"

        H_out = (H - KH) // stride + 1
        W_out = (W - KW) // stride + 1
        out = np.zeros((B, C_out, H_out, W_out), dtype=inp_val.dtype)

        # --- Convolution ---
        for b in range(B):
            for co in range(C_out):
                for h in range(0, H - KH + 1, stride):
                    for w in range(0, W - KW + 1, stride):
                        patch = inp_val[b, :, h:h+KH, w:w+KW]  # shape [C_in, KH, KW]
                        out[b, co, h // stride, w // stride] = np.sum(patch * w_val[co])

        if out.shape[1] == 1:
            out = out.reshape(out.shape[0], out.shape[2], out.shape[3])

        self.op_results_map[op] = out

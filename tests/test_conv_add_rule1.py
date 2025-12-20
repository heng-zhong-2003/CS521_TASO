import itertools
from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_conv2d import Conv2DOperator
from patterns.operator_matmul import MatmulOperator
from patterns.graph import Graph
import ast
from onnx_gen.rule_gen import RuleGen
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
import torch
import torch.nn as nn
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import onnxscript

# --- Step 1: Define baseline model: add(conv(a, c), conv(b, c)) ---
class AddConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Same kernel used for both convolutions
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, a, b):
        c1 = self.conv(a)
        c2 = self.conv(b)
        return c1 + c2


# --- Step 2: Export to ONNX ---
def export_model_to_onnx(model, path="add_conv.onnx"):
    model.eval()
    a = torch.randn(1, 1, 4, 4)
    b = torch.randn(1, 1, 4, 4)
    torch.onnx.export(
        model,
        (a, b),
        path,
        input_names=["a", "b"],
        output_names=["out"],
        opset_version=18,
    )
    return onnx.load(path)


# --- Step 3: Define the TASO-style rewrite rule in ONNXScript form ---
def taso_target_0(op, in_0, in_1, in_2):
    return op.Add(op.Conv(in_0, in_1, strides=[1, 1]), op.Conv(in_2, in_1, strides=[1, 1]))

def taso_replacement_0(op, in_0, in_1, in_2):
    return op.Conv(op.Add(in_0, in_1), in_2, strides=[1, 1])

def taso_match_cond_0(ctx, in_0, in_1, in_2):
    return True
taso_rule_0 = pattern.RewriteRule(taso_target_0, taso_replacement_0, taso_match_cond_0)



# def taso_target_0(g: graph_builder.GraphBuilder, in_0, in_1, in_2):
    # return g.Add(
        # g.Conv(in_0, in_1, strides=[1, 1]),
        # g.Conv(in_2, in_1, strides=[1, 1]),
    # )


# def taso_replacement_0(g: graph_builder.GraphBuilder, in_0, in_1, in_2):
    # return g.Conv(
        # g.Add(in_0, in_1),
        # in_2,
        # strides=[1, 1],
    # )


# def taso_match_cond_0(ctx, in_0, in_1, in_2):
    # return True


# taso_rule_0 = rewriter.RewriteRule(
    # "taso_rule_0", taso_target_0, taso_replacement_0, taso_match_cond_0
# )


# --- Step 4: Test function that applies rewrite ---
def test_add_conv_rewrite_onnx():
    model = AddConvModel()
    onnx_model = export_model_to_onnx(model)

    print("Before rewrite, ops:", [n.op_type for n in onnx_model.graph.node])

    rewritten_model = onnxscript.rewriter.rewrite(model=onnx_model, pattern_rewrite_rules=[taso_rule_0])

    print("Rewritten ONNX model:")
    print(rewritten_model)
    print("After rewrite, ops:", [n.op_type for n in rewritten_model.graph.node])
    # Verify structure changed: single Conv + one Add (inside input)
    op_types = [n.op_type for n in rewritten_model.graph.node]
    assert op_types.count("Conv") == 1, f"Expected 1 Conv, found {op_types.count('Conv')}"
    assert op_types.count("Add") == 1, f"Expected 1 Add, found {op_types.count('Add')}"

    print("Rewrite succeeded — add(conv(a, c), conv(b, c)) → conv(add(a, b), c)")


if __name__ == "__main__":
    test_add_conv_rewrite_onnx()

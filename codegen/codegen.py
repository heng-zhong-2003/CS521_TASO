from __future__ import annotations
from typing import Callable
from patterns.operator_interface import Operator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.operator_input import InputOperator
from patterns.graph import Graph
import proj_utils


class Codegen:
    def __init__(self) -> None:
        # Will need to maintain a map from operator nodes here to matched
        #   HloInstruction * names in C++.
        self.op_map: dict[type, str] = {
            AddOperator: "Add",
            MatmulOperator: "Dot",
        }
        self.node_map: dict[Operator, str] = {}
        self.cpp_id: int = 0
        self.source_graph: Graph | None = None
        self.target_graph: Graph | None = None
        self.visitor_class_name = 'TASOVisitor'

    def get_opcode(self, node: Operator) -> str:
        """
        check
        """
        op_type = type(node)
        if op_type not in self.op_map:
            raise ValueError(f"Unsupported operator type: {op_type.__name__}")
        return self.op_map[op_type]

    def create_cpp_name(self, node: Operator) -> str:
        """
        cpp names for old/new pointers
        """
        if node not in self.node_map:
            self.node_map[node] = f"str{self.cpp_id}"
            self.cpp_id += 1
        return self.node_map[node]

    def generate(self, source_graph: Graph, target_graph: Graph) -> str:
        self.source_graph = source_graph
        self.target_graph = target_graph
        self.node_map.clear()
        self.cpp_id = 0
        code_parts = [self.generate_prolog(), self.generate_source(), self.generate_target(),
                      self.generate_epilog()]
        return "\n".join(code_parts)

    def generate_cc_prolog(self) -> str:
        includes = ''
        begins = f'''\
namespace xla {{
namespace {{
absl::Status {self.visitor_class_name}::HandleAll(HloInstruction *root) {{
'''
        proj_utils.todo()

    def generate_cc_epilog(self) -> str:
        ends = f'''\
}}
}}
return absl::OkStatus();
}}
'''
        return ends

    def generate_h(self) -> str:
        proj_utils.todo()

    def generate_source(self) -> str:
        """
        convert source graph to cpp
        """
        input_nodes = self.source_graph.get_inputs()
        var_names = []
        for node in input_nodes:
            cpp_var = self.create_cpp_name(node)
            var_names.append(cpp_var)

        # generate Match function
        if var_names:
            declare = "  HloInstruction *" + ", *".join(var_names) + ";\n"
        else:
            declare = ""
        root_node = self.source_graph.outputs[0]
        match_function = self.build_Match(root_node)
        return f"""{declare}
          // Match source pattern
          if (Match(root, {match_function})) {{
        """

    def generate_target(self) -> str:
        """
        convert target graph to cpp
        """
        if not self.target_graph:
            return "    // No replacement\n  }\n  return absl::OkStatus();\n"

        code_lines = ["\n  // Create replacement pattern"]

        # generate new pointer
        # for example: sum_of_constants = add->AddInstruction(
        #           HloInstruction::CreateBroadcast(add->shape(), sum_of_constants, {}));
        sorted_nodes = self.search(self.target_graph)
        for node in sorted_nodes:
            inputs = list(node.get_inputs())

            if len(inputs) == 0:
                continue
            if node in self.node_map and node in self.search(self.source_graph):
                continue
            if node == self.target_graph.outputs[0]:
                continue

            cpp_var = self.create_cpp_name(node)
            xla_opcode = self.get_opcode(node)
            operand_names = [self.create_cpp_name(inp) for inp in inputs]
            operands_str = ", ".join(operand_names)
            code_lines.append(
                f"    HloInstruction* {cpp_var} = root->AddInstruction("
                f"HloInstruction::CreateBinary(root->shape(), HloOpcode::{xla_opcode}, {operands_str}));"
            )

        # generate ReplaceWithNewInstruction
        # for example: return ReplaceWithNewInstruction(
        #         add,
        #         HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, rhs, lhs));
        new_root_node = self.target_graph.outputs[0]
        xla_opcode = self.get_opcode(new_root_node)
        inputs = list(new_root_node.get_inputs())
        operand_names = [self.create_cpp_name(inp) for inp in inputs]
        operands_str = ", ".join(operand_names)

        code_lines.append(
            f"    return ReplaceWithNewInstruction("
            f"root, HloInstruction::CreateBinary(root->shape(), HloOpcode::{xla_opcode}, {operands_str}));"
        )

        code_lines.append("  }")
        # code_lines.append("  return absl::OkStatus();")

        return "\n".join(code_lines) + "\n"

    def search(self, graph: Graph) -> list[Operator]:
        visited = set()
        result = []

        def dfs(node: Operator):
            if node in visited:
                return
            visited.add(node)
            for inp in node.get_inputs():
                dfs(inp)
            result.append(node)

        for output in graph.outputs:
            dfs(output)

        return result

    def build_Match(self, node: Operator) -> str:
        """
        create Match() function:
        if (Match(add, m::Add(m::Subtract(m::Constant(&c1), m::NonConstant(&a)), m::Constant(&c2)))
        """
        inputs = list(node.get_inputs())

        if len(inputs) == 0:
            cpp_var = self.create_cpp_name(node)
            return f"m::Op(&{cpp_var})"

        xla_opcode = self.get_opcode(node)
        sub_patterns = [self.build_Match(inp) for inp in inputs]

        if len(sub_patterns) == 1:
            return f"m::{xla_opcode}({sub_patterns[0]})"
        elif len(sub_patterns) == 2:
            return f"m::{xla_opcode}({sub_patterns[0]}, {sub_patterns[1]})"
        else:
            patterns_str = ", ".join(sub_patterns)
            return f"m::{xla_opcode}({patterns_str})"

from onnx_gen.pattern_to_onnx_expr import PatternToOnnxExpr
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import ast
from patterns.graph import Graph
import proj_utils


class RuleGen:
    def __init__(self) -> None:
        self.rule_counter = 0
    
    def target_pattern_name(self) -> str:
        return f'taso_target_{self.rule_counter}'
    
    def replacement_pattern_name(self) -> str:
        return f'taso_replacement_{self.rule_counter}'
    
    def rule_name(self) -> str:
        return f'taso_rule_{self.rule_counter}'
    
    def generate_rule(self,
                      target_pattern: Graph,
                      replacement_pattern: Graph) -> ast.AST:
        tpn = self.target_pattern_name()
        rpn = self.replacement_pattern_name()
        ptoe_target = PatternToOnnxExpr()
        target_ast, target_ins = \
            ptoe_target.pattern_to_onnx_pattern(target_pattern)
        ptoe_replacement = PatternToOnnxExpr()
        replacement_ast, replacement_ins = \
            ptoe_replacement.pattern_to_onnx_pattern(replacement_pattern)
        target_par_op = [ast.arg(arg=ptoe_target.onnx_pattern_op_name, annotation=None)]
        target_par_inst = [ast.arg(arg=n, annotation=None) for n in ptoe_target.input_ops]
        target_args = ast.arguments(
            posonlyargs=[],
            args=target_par_op + target_par_inst,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        target_func_def = ast.FunctionDef(
            name=tpn,
            args=target_args,
            body=[ast.Return(value=target_ast)], # type: ignore
            decorator_list=[],
            type_params=[]
        )
        replacement_par_op = [ast.arg(arg=ptoe_replacement.onnx_pattern_op_name, annotation=None)]
        replacement_par_inst = [ast.arg(arg=n, annotation=None) for n in ptoe_replacement.input_ops]
        replacement_args = ast.arguments(
            posonlyargs=[],
            args=replacement_par_op + replacement_par_inst,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        replacement_func_def = ast.FunctionDef(
            name=rpn,
            args=replacement_args,
            body=[ast.Return(value=replacement_ast)], # type: ignore
            decorator_list=[],
            type_params=[]
        )
        rule_def = ast.Assign(
            targets=[ast.Name(id=self.rule_name(), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='pattern.RewriteRule', ctx=ast.Load()),
                args=[
                    ast.Name(id=tpn, ctx=ast.Load()),
                    ast.Name(id=rpn, ctx=ast.Load())
                ],
                keywords=[]
            )
        )
        module_def = ast.Module(
            body=[
                target_func_def,
                replacement_func_def,
                rule_def
            ],
            type_ignores=[]
        )
        self.rule_counter += 1
        return module_def

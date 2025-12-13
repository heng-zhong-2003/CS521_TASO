from onnx_gen.pattern_to_onnx_expr import PatternToOnnxExpr
from onnx_gen.match_conditions_gen import MatchConditionsGen
import onnx
from onnxscript.rewriter import pattern
from onnxscript import ir
import ast
from patterns.graph import Graph
import proj_utils
from onnx_gen.concat_info_inference import ConcatInfoInference


class RuleGen:
    def __init__(self) -> None:
        self.rule_counter = 0
        self.target_inferrer: ConcatInfoInference
        self.replacement_inferrer: ConcatInfoInference
    
    def plug_in_inferrers(self,
                          target_inferrer: ConcatInfoInference,
                          replacement_inferrer: ConcatInfoInference) -> None:
        """
        Must call this on the inferrer applied on the same graph patterns
          before using generate_rule() if the patterns have concat-split.
        """
        self.target_inferrer = target_inferrer
        self.replacement_inferrer = replacement_inferrer

    def target_pattern_name(self) -> str:
        return f'taso_target_{self.rule_counter}'

    def replacement_pattern_name(self) -> str:
        return f'taso_replacement_{self.rule_counter}'

    def matching_conditions_name(self) -> str:
        return f'taso_match_cond_{self.rule_counter}'

    def rule_name(self) -> str:
        return f'taso_rule_{self.rule_counter}'

    def generate_rule(self,
                      target_pattern: Graph,
                      replacement_pattern: Graph) -> ast.AST:
        tpn = self.target_pattern_name()
        rpn = self.replacement_pattern_name()
        ptoe_target = PatternToOnnxExpr()

        ptoe_target.plug_in_inferrer(self.target_inferrer)

        target_ast, target_ins = \
            ptoe_target.pattern_to_onnx_pattern(target_pattern)
        ptoe_replacement = PatternToOnnxExpr()
        
        ptoe_replacement.plug_in_inferrer(self.replacement_inferrer)

        print('Translate replacement pattern')
        replacement_ast, replacement_ins = \
            ptoe_replacement.pattern_to_onnx_pattern(replacement_pattern)
        # print("Target AST:", ast.dump(target_ast))
        # print("Replacement AST:", ast.dump(replacement_ast))
        target_par_op = [
            ast.arg(arg=ptoe_target.onnx_pattern_op_name, annotation=None)]
        target_par_ins = [ast.arg(arg=n, annotation=None) for n in target_ins]
        target_args = ast.arguments(
            posonlyargs=[],
            args=target_par_op + target_par_ins,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        target_func_def = ast.FunctionDef(
            name=tpn,
            args=target_args,
            body=[ast.Return(value=target_ast)],  # type: ignore
            decorator_list=[],
            type_params=[]
        )
        replacement_par_op = [
            ast.arg(arg=ptoe_replacement.onnx_pattern_op_name, annotation=None)]
        replacement_par_ins = [
            ast.arg(arg=n, annotation=None) for n in replacement_ins]
        replacement_args = ast.arguments(
            posonlyargs=[],
            args=replacement_par_op + replacement_par_ins,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        replacement_func_def = ast.FunctionDef(
            name=rpn,
            args=replacement_args,
            body=[ast.Return(value=replacement_ast)],  # type: ignore
            decorator_list=[],
            type_params=[]
        )
        mcg = MatchConditionsGen(
            ptoe_target.input_ops_names_map,
            ptoe_target.graph_onnx_node_map
        )
        cond_ast = mcg.generate_matching_conditions(target_pattern)
        cond_func_def = ast.FunctionDef(
            name=self.matching_conditions_name(),
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=mcg.root_parameter_name, annotation=None)]
                + target_par_ins,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[ast.Return(value=cond_ast)],  # type: ignore
            decorator_list=[],
            type_params=[]
        )
        rule_def = ast.Assign(
            targets=[ast.Name(id=self.rule_name(), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='pattern.RewriteRule', ctx=ast.Load()),
                args=[
                    ast.Name(id=tpn, ctx=ast.Load()),
                    ast.Name(id=rpn, ctx=ast.Load()),
                    ast.Name(id=self.matching_conditions_name(), ctx=ast.Load())
                ],
                keywords=[]
            )
        )
        module_def = ast.Module(
            body=[
                target_func_def,
                replacement_func_def,
                cond_func_def,
                rule_def
            ],
            type_ignores=[]
        )
        self.rule_counter += 1
        return module_def

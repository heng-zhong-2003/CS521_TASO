from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
from synthesizer.validate import RuleValidator
from synthesizer.build import build  # your Build() implementation
from codegen.codegen import Codegen
import itertools


lf = Fingerprint()         # for fingerprinting graphs
validator = RuleValidator()  # for comparing equivalent graphs
D = {}                      # fingerprint → list of graphs

# 2️⃣ Create initial input operators
in1 = InputOperator()
in2 = InputOperator()
inputs = [in1, in2]

# 3️⃣ Create an initial graph with inputs
graph = Graph(inputs)

# 4️⃣ Define available operator classes
P = [AddOperator, MatmulOperator]

# 5️⃣ Build all possible small graphs (threshold controls graph depth)
build(
    n=1,
    G=graph,
    I=inputs,
    P=P,
    D=D,
    F=lf,
    threshold=2,   # keep small for testing
)

generator = Codegen()

file = open('rslt.cpp', 'w')

for fp, graphs in D.items():
        if len(graphs) > 1:
            for g1, g2 in itertools.combinations(graphs, 2):
                if validator.validate(g1, g2):
                    gen = generator.generate(g1, g2)
                    print(gen, file=file)
                print('')

file.close()

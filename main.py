from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
from synthesizer.validate import RuleValidator
from synthesizer.build import build  # your Build() implementation
from synthesizer.build import print_graph
from codegen.codegen import Codegen
from patterns.operator_concat import ConcatOperator
from patterns.operator_conv2d import Conv2DOperator
from patterns.operator_split import SplitOperator
import itertools
import copy

lf = Fingerprint()         # for fingerprinting graphs
validator = RuleValidator()  # for comparing equivalent graphs
D = {}                      # fingerprint → list of graphs

A = InputOperator()
B = InputOperator()
C = InputOperator()
inputs=[A, B, C]
G1 = Graph(inputs)
op1 = MatmulOperator(A, B)
op2 = MatmulOperator(A, C)
G1.add_operator(op1)
G1.add_operator(op2)

fpTest= lf.fingerprint(G1)

A2 = InputOperator()
B2 = InputOperator()
C2 = InputOperator()
inputs=[A2, B2, C2]
G2 = Graph(inputs)
mm1 = MatmulOperator(A2, B2)
mm2 = MatmulOperator(A2, C2)
add1 = AddOperator(mm1, mm2)
G2.add_operator(mm1)
G2.add_operator(mm2)
G2.add_operator(add1)

fpTest2 = lf.fingerprint(G2)

A3 = InputOperator()
B3 = InputOperator()
C3 = InputOperator()
inputs=[A3, B3, C3]
G3 = Graph(inputs)
conv1 = Conv2DOperator(A3, C3, stride=1)
conv2 = Conv2DOperator(B3, C3, stride=1)
add1 = AddOperator(conv1, conv2)
G3.add_operator(conv1)
G3.add_operator(conv2)
G3.add_operator(add1)

fpTest3 = lf.fingerprint(G3)


# 2️⃣ Create initial input operators
in1 = InputOperator()
in2 = InputOperator()
in3 = InputOperator()
inputs = [in1, in2, in3]

# 3️⃣ Create an initial graph with inputs
graph = Graph(inputs)

# 4️⃣ Define available operator classes
P = [AddOperator, MatmulOperator]

# 5️⃣ Build all possible small graphs (threshold controls graph depth)
build(
    n=1,
    G=graph,
    I=copy.copy(inputs),
    P=P,
    D=D,
    F=lf,
    threshold=4,   # keep small for testing
)

generator = Codegen()

file = open('rslt.cpp', 'w')

count_f=0
count_g=0
for fp, graphs in D.items():
    if fp == fpTest:
        for gr in graphs:
            print_graph(gr, f"/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files/fp_want", graphNumber=count_g)
            count_g += 1
    if fp == fpTest2:
        for gr in graphs:
            print_graph(gr, f"/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files/fp_want2", graphNumber=count_g)
            count_g += 1
    if fp == fpTest3:
        print("MATCHED")
        for gr in graphs:
            print_graph(gr, f"/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files/fp_want3", graphNumber=count_g)
            count_g += 1
    # if len(graphs) > 1:
        # for gr in graphs:
            # print_graph(gr, f"/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files/fp_{count_f}", graphNumber=count_g)
            # count_g += 1
    count_f+=1
            # for g1, g2 in itertools.combinations(graphs, 2):
                # if validator.validate(g1, g2):
                    # print_graph(g1, "/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files", graphNumber=count)
                    # count += 1
                    # print_graph(g2, "/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files", graphNumber=count)
                    # count += 1
                    # # gen = generator.generate(g1, g2)
                    # # print(gen, file=file)

    # print('')

file.close()


def test_matmul_split_concat_equivalence():
    # Create shared fingerprinting context (same random inputs)
    F = Fingerprint()

    # --- Graph 1: two separate matmuls ---
    A = InputOperator()
    B = InputOperator()
    C = InputOperator()
    inputs=[A, B, C]
    G1 = Graph(inputs)
    op1 = MatmulOperator(A, B)
    op2 = MatmulOperator(A, C)
    G1.add_operator(op1)
    G1.add_operator(op2)

    fp1 = F.fingerprint(G1)
    print(f"Fingerprint(G1) = {fp1}")

    # --- Graph 2: single matmul with concat and split ---
    A2 = InputOperator()
    B2 = InputOperator()
    C2 = InputOperator()
    inputs=[A2, B2, C2]
    G2 = Graph(inputs)
    concat = ConcatOperator(B2, C2, axis=2)
    mm = MatmulOperator(A2, concat)
    split = SplitOperator(mm, axis=2)
    G2.add_operator(concat)
    G2.add_operator(mm)
    G2.add_operator(split)

    fp2 = F.fingerprint(G2)
    print(f"Fingerprint(G2) = {fp2}")

    # --- Check equivalence ---
    if fp1 == fp2:
        print("✅ Fingerprinting detected equivalence.")
    else:
        print("❌ Fingerprinting did NOT detect equivalence.")

# test_matmul_split_concat_equivalence()

def test_conv():
    F = Fingerprint()

    A3 = InputOperator()
    B3 = InputOperator()
    C3 = InputOperator()
    inputs=[A3, B3, C3]
    G3 = Graph(inputs)
    conv1 = Conv2DOperator(A3, C3, stride=1)
    conv2 = Conv2DOperator(B3, C3, stride=1)
    add1 = AddOperator(conv1, conv2)
    G3.add_operator(conv1)
    G3.add_operator(conv2)
    G3.add_operator(add1)

    fp1 = F.fingerprint(G3)
    print(f"Fingerprint(G3) = {fp1}")


    A4 = InputOperator()
    B4 = InputOperator()
    C4 = InputOperator()
    inputs=[A4, B4, C4]
    G4 = Graph(inputs)
    add1 = AddOperator(A4, B4)
    conv1 = Conv2DOperator(add1, C4, stride=1)
    G4.add_operator(add1)
    G4.add_operator(conv1)

    fp2 = F.fingerprint(G4)
    print(f"Fingerprint(G4) = {fp2}")

    if fp1 == fp2:
        print("✅ Fingerprinting detected equivalence.")
    else:
        print("❌ Fingerprinting did NOT detect equivalence.")



test_conv()

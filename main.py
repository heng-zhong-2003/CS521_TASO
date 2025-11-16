from patterns.operator_input import InputOperator
from patterns.operator_add import AddOperator
from patterns.operator_matmul import MatmulOperator
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
from synthesizer.validate import RuleValidator
from synthesizer.build import build  # your Build() implementation
import itertools


def main():
    # 1️⃣ Initialize helpers
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
    P = [AddOperator]

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

    # 6️⃣ Print summary of fingerprints
    print(f"\nTotal unique fingerprints generated: {len(D)}")

    for fp, graphs in D.items():
        print(f"Fingerprint {fp} → {len(graphs)} graph(s)")

    # 7️⃣ Optionally test equivalence between graphs with the same fingerprint
    for fp, graphs in D.items():
        if len(graphs) > 1:
            for g1, g2 in itertools.combinations(graphs, 2):
                if validator.validate(g1, g2):
                    print(f"✅ Equivalent graphs found for fingerprint {fp}")
                else:
                    print(f"❌ Different graphs for fingerprint {fp}")


if __name__ == "__main__":
    main()

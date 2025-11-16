import synthesizer.fingerprint
from patterns.operator_interface import Operator
from patterns.evaluate import get_operator_kind
import itertools
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint

## Implementing the BUILD function to generate random graphs given a list of operators ##
## ------------------------------------------------------------------------------------##

def build(n: int, 
          G: Graph, 
          I: list[Operator], 
          P: list[type[Operator]], 
          D: dict[int, list[Graph]],
          F: Fingerprint, 
          threshold: int):
    # Recursively building a random graph

    print("inside build")
    # Store current graph
    try:
        fp = F.fingerprint(G)
    except Exception as e:
        print(f"[Fingerprint Error] {type(e).__name__}: {e}")
        # Skip graphs that can't be evaluated
        return

    # store graph in hash table (D)
    if fp not in D:
        D[fp] = []
    D[fp].append(G.copy())   # make sure to store a copy, not reference

    # Depth cutoff
    if n >= threshold:
        return

    # Step 2: enumerate operators and their input tensor combinations

    for opClass in P:
        arity = opClass.get_arity()  # assume each operator class defines this
        # create combinations of arity number of objects at a time from the list I
        print("inside for opclass in P")
        for inputs in itertools.permutations(I, arity):

            new_op = opClass(list(inputs))

            # avoid duplicate computation. This is being done here instead of 
            # in the beginning of the function for efficiency
            print("checking duplicates")
            if(G.check_duplicates(new_op)):
                # if duplicate found, don't use this operator combination
                continue

            print("adding new operator")
            # append to the graph (this automatically updates users list for the inputs)
            # also update the list of inputs available to further iterations
            G.add_operator(new_op)
            I.append(new_op)

            # recurse
            build(n + 1, G, I, P, D, F, threshold)

            # backtrack
            # for _ in new_outputs: I.pop()
            # G.pop()
            G.remove_operator(new_op)
            I.remove(new_op)



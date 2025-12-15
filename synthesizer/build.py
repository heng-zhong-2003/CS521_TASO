import synthesizer.fingerprint
from patterns.operator_interface import Operator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.evaluate import get_operator_kind
import itertools
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
import sys
from proj_config import *

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

    print("inside build", file=sys.stderr)
    # Store current graph
    try:
        fp = F.fingerprint(G)
    except Exception as e:
        print(f"[Fingerprint Error] {type(e).__name__}: {e}", file=sys.stderr)
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
        print("inside for opclass in P", file=sys.stderr)
        
        # making sure operators with multiple outputs are enumerated correctly --- once per output instead of just once overall
        # instead of inserting back just the operator, we insert a tuple with the operator and the corresponding output it represents
        # if not a multi-output op, we just replace the existing element with a single-element tuple
        # for index in range(0,len(I)):
            # op = I[index]
            # if isinstance(op, SplitOperator):
                # op = I.pop(index)
                # I.append((op, 0))
                # I.append((op, 1))
            # else:
                # op = I.pop(index)
                # I.append((op,))

        new_I = []
        for op in I:
            if isinstance(op, SplitOperator):
                new_I.append((op, 0))
                new_I.append((op, 1))
            else:
                new_I.append((op,))   # wrap *every* operator
        # I = new_I

        for inputs in itertools.permutations(new_I, arity):
            # returns a list of new operators to add in this position, 
            # where each list element is the same operator with different parameter combinations
            new_op_list = create_new_operator(opClass, arity, inputs)

            for new_op in new_op_list:
                # set user map for multi-output operators
                for op_and_pos in inputs:
                    if(len(op_and_pos) == 2): # this is a multi-output op
                       op_and_pos.add_user_component(new_op, op_and_pos[1]) 

                # avoid duplicate computation. This is being done here instead of
                # in the beginning of the function for efficiency
                # TODO This does not consider multi output operators yet
                print("checking duplicates", file=sys.stderr)
                if (G.check_duplicates(new_op)):
                    # if duplicate found, don't use this operator combination
                    continue

                print("adding new operator", file=sys.stderr)
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


def create_new_operator(opClass, arity, inputs):
    # for inputs that represent only a particular output of some operator,
    # we update the user map for that operator with the appropriate value

    if(arity == 1):
        if opClass is SplitOperator:
            # the axis is currently a placeholder. The correct one will be updated
            # after inference is run on the graph.
            return [opClass(inputs[0][0], 1)]
        else:
            return [opClass(inputs[0][0])]

    elif(arity == 2):
        if opClass is ConcatOperator:
            oplist=[] # multiple possible axes
            for axis in range(0,MAX_AXIS_NUM):
                oplist.append(opClass(inputs[0][0], inputs[1][0], axis))
            return oplist
        else:
            # if (isinstance(inputs[0][0], tuple)):
                # print("operator is a tuple..?")
            return [opClass(inputs[0][0], inputs[1][0])]

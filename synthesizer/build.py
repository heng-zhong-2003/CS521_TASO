import synthesizer.fingerprint
from patterns.operator_interface import Operator
from patterns.operator_concat import ConcatOperator
from patterns.operator_split import SplitOperator
from patterns.operator_add import AddOperator
from patterns.operator_input import InputOperator
from patterns.evaluate import get_operator_kind
from patterns.operator_conv2d import Conv2DOperator
import itertools
from patterns.graph import Graph
from synthesizer.fingerprint import Fingerprint
import sys
from proj_config import *
import os

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

    print("inside build, n = ",n)
    # Store current graph
    try:
        print("about to fingerprint. Graph has ", len(G.get_inputs()), " inputs")
        fp = F.fingerprint(G)
    except Exception as e:
        print(f"[Fingerprint Error] {type(e).__name__}: {e}")
        # Skip graphs that can't be evaluated
        return
    
    # print_graph(G, "/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files", graphNumber=fp)

    # store graph in hash table (D)
    if fp not in D:
        D[fp] = []
    D[fp].append(G.copy())   # make sure to store a copy, not reference

    print("     after adding graph to hash table, graph has:")
    for op in G.operators:
        print("     ", type(op), " with ", len(op.get_users()), " users")
    # Depth cutoff
    if n >= threshold:
        print("threshold met")
        return

    # Step 2: enumerate operators and their input tensor combinations

    for opClass in P:
        arity = opClass.get_arity()  # assume each operator class defines this
        # create combinations of arity number of objects at a time from the list I
        # print("inside for opclass in P")
        
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
        print("replacing inputs with tuples")
        for op in I:
            print("     ", type(op), ", ", end="")
            if isinstance(op, SplitOperator):
                new_I.append((op, 0))
                new_I.append((op, 1))
            else:
                new_I.append((op,))   # wrap *every* operator
        # I = new_I
        print("")


        for inputs in itertools.permutations(new_I, arity):
            # returns a list of new operators to add in this position, 
            # where each list element is the same operator with different parameter combinations
            new_op_list = create_new_operator(opClass, arity, inputs)

            print("after create_new_operator. Graph has ", len(G.get_inputs()), " inputs")
            for new_op in new_op_list:
                # set user map for multi-output operators
                for op_and_pos in inputs:
                    if(len(op_and_pos) == 2): # this is a multi-output op
                       op_and_pos[0].add_user_component(new_op, op_and_pos[1]) 

                # avoid duplicate computation. This is being done here instead of
                # in the beginning of the function for efficiency
                # TODO This does not consider multi output operators yet
                print("checking duplicates")
                if (G.check_duplicates(new_op)):
                    # if duplicate found, don't use this operator combination
                    print("duplicate found")
                    continue

                kind1 = get_operator_kind(new_op)
                print("adding new ", kind1,  "operator with identity ",id(new_op) ,"; operand types are: ", end="")
                for xyz in inputs:
                    kind2 = type(xyz[0]) #get_operator_kind(xyz[0])
                    print(kind2,", ", end="")
                print("\n")
                # append to the graph (this automatically updates users list for the inputs)
                # also update the list of inputs available to further iterations
                G.add_operator(new_op)
                print("after adding operator. Graph has ", len(G.get_inputs()), " inputs")
                I.append(new_op)
                print("I.append. Graph has ", len(G.get_inputs()), " inputs")

                # print("before calling build recursively, inputs have ", len(inputs[0][0].get_users()), " and ", len(inputs[1][0].get_users()), " users respectively, with first elements of type ", type(inputs[0][0].get_users()[0]), " and ", type(inputs[1][0].get_users()[0]), " with id ", id(inputs[0][0].get_users()[0]))
                # recurse
                build(n + 1, G, I, P, D, F, threshold)

                # print("after calling build recursively, inputs have ", len(inputs[0][0].get_users()), " and ", len(inputs[1][0].get_users()), " users respectively, with first elements of type ", type(inputs[0][0].get_users()[0]), " and ", type(inputs[1][0].get_users()[0]), " with id ", id(inputs[0][0].get_users()[0]))
                # backtrack
                # for _ in new_outputs: I.pop()
                # G.pop()
                print("removing operator with id ", id(new_op))
                G.remove_operator(new_op)
                I.remove(new_op)
                # print("after removing op, inputs have ", len(inputs[0][0].get_users()), " and ", len(inputs[1][0].get_users()), " users respectively of type ", type(inputs[0][0].get_users()[0]), " and ", type(inputs[1][0].get_users()[0]), " with id ", id(inputs[0][0].get_users()[0]))

def build_hardcoded(n: int,
          G: Graph,
          I: list[Operator],
          P: list[type[Operator]],
          D: dict[int, list[Graph]],
          F: Fingerprint,
          threshold: int):
    # Recursively building a random graph

    # Store current graph
    try:
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

        fpTest3 = F.fingerprint(G3)

        A4 = InputOperator()
        B4 = InputOperator()
        C4 = InputOperator()
        inputs=[A4, B4, C4]
        G4 = Graph(inputs)
        add1 = AddOperator(A4, B4)
        conv1 = Conv2DOperator(add1, C4, stride=1)
        G4.add_operator(add1)
        G4.add_operator(conv1)

        fpTest4 = F.fingerprint(G4)
    except Exception as e:
        # Skip graphs that can't be evaluated
        return


    # print_graph(G, "/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files", graphNumber=fp)

    # store graph in hash table (D)
    if fpTest3 not in D:
        D[fpTest3] = []
    D[fpTest3].append(G3.copy())   # make sure to store a copy, not reference

    if fpTest4 not in D:
        D[fpTest4] = []
    D[fpTest4].append(G4.copy())   # make sure to store a copy, not reference


def create_new_operator(opClass, arity, inputs):
    # for inputs that represent only a particular output of some operator,
    # we update the user map for that operator with the appropriate value

    if(arity == 1):
        if opClass is SplitOperator:
            # the axis is currently a placeholder. The correct one will be updated
            # after inference is run on the graph.
            return [opClass(inputs[0][0], axis=2)]
        else:
            return [opClass(inputs[0][0])]

    elif(arity == 2):
        if opClass is ConcatOperator:
            oplist=[] # multiple possible axes
            for axis in range(2,MAX_AXIS_NUM):
                oplist.append(opClass(inputs[0][0], inputs[1][0], axis))
            return oplist
        elif opClass is Conv2DOperator:
            oplist = []
            for stride in range(1, MAX_STRIDE_NUM):
                oplist.append(opClass(inputs[0][0], inputs[1][0], stride))
            return oplist
        else:
            # if (isinstance(inputs[0][0], tuple)):
                # print("operator is a tuple..?")
            return [opClass(inputs[0][0], inputs[1][0])]

def print_graph(G, dot_path: str | None = None, show_users=True, show_inputs=True, graphNumber: int = 0):
    """
    Pretty-print a computational graph for debugging, and optionally
    export it to a Graphviz `.dot` file for visualization.

    Parameters
    ----------
    G : Graph
        The graph to print.
    dot_path : str | None
        If provided, saves a .dot file of the graph at this path.
    show_users : bool
        Whether to print each operator's users.
    show_inputs : bool
        Whether to print each operator's inputs.
    """
    os.makedirs(dot_path, exist_ok=True)
    dot_path = dot_path + "/" + f"graph{graphNumber}.dot"
    print("\n🧩 Graph structure:")
    print("-" * 60)
    print(f"Total operators: {len(G.operators)}")
    print(f"Inputs: {[hex(id(inp)) for inp in G.inputs]}")
    print("-" * 60)

    dot_lines = [
        "digraph G {",
        "    rankdir=LR;",
        "    node [shape=box, style=rounded, fontname=Helvetica];"
    ]

    for op in G.operators:
        op_type = type(op).__name__
        op_id = f"n{id(op)}"
        dot_lines.append(f'    {op_id} [label="{op_type}\\n{id(op)}"];')

        print(f"🔸 {op_type} ({hex(id(op))})")

        # Print and connect inputs
        if show_inputs:
            inputs = op.get_inputs() if hasattr(op, "get_inputs") else []
            if inputs:
                print("   ↳ inputs:")
                for i in inputs:
                    print(f"      - {type(i).__name__} ({hex(id(i))})")
                    # Draw edge from input -> op
                    dot_lines.append(f"    n{id(i)} -> {op_id};")
            else:
                print("   ↳ inputs: None")

        # Print users (optional)
        if show_users and hasattr(op, "get_users"):
            users = op.get_users()
            if users:
                print("   ↳ users:")
                for u in users:
                    print(f"      - {type(u).__name__} ({hex(id(u))})")
            else:
                print("   ↳ users: None")

        # For SplitOperator, show user_component_map edges
        # if hasattr(op, "user_component_map") and op.user_component_map:
            # print("   ↳ user_component_map:")
            # for usr, comp in op.user_component_map.items():
                # print(f"      - {type(usr).__name__} ({hex(id(usr))}) -> component {comp}")
                # dot_lines.append(f"    n{id(op)} -> n{id(usr)} [label=\"{comp}\"];")

        print("-" * 60)

    dot_lines.append("}")

    if dot_path:
        with open(dot_path, "w") as f:
            f.write("\n".join(dot_lines))
        print(f"✅ Graphviz file written to: {dot_path}")
        print("   You can visualize it using:")
        print(f"     dot -Tpng {dot_path} -o graph.png")
        print(f"     xdot {dot_path}")


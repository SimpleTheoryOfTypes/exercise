#!/usr/env/python3
#cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;mlir" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="Hexagon" -DLLVM_TARGETS_TO_BUILD="X86"
#cmake --build . --target check-llvm-codegen-hexagon
#cmake --build . --target check-llvm-codegen-hexagon-pipeliner
#./test/CodeGen/Hexagon/swp-vsum.ll
#dev/LLVM/llvm-project/build/test/CodeGen/Hexagon/Output/swp-vsum.ll.script
import networkx as nx
G = nx.DiGraph()
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("C", "T0")
G.add_edge("C", "S0")
G.add_edge("B", "m0")
G.add_edge("B", "D")
G.add_edge("D", "T1")
G.add_edge("D", "S1")
G.add_edge("A", "E")
G.add_edge("E", "F")
G.add_edge("F", "G")
G.add_edge("F", "m1")
G.add_edge("E", "H")
G.add_edge("H", "T2")
G.add_edge("H", "m2")
G.add_edge("H", "S2")
G.add_edge("G", "m3")
G.add_edge("G", "a0")
G.add_edge("Z", "W")
G.add_edge("W", "X")
G.add_edge("X", "m4")
G.add_edge("X", "m6")
G.add_edge("W", "m7")
G.add_edge("W", "Y")
G.add_edge("Y", "m5")

print(G)

roots = [v for v, indegree in G.in_degree() if indegree == 0]
root = roots[0]

# A - B - C - T0
#           - S0
#       - m0
#       - D
#           - T1
#           - S1
#   - E - F - G - m3
#               - a0
#           - m1
#       - H - T2
#           - m2
#           - S2
# Z - W - X - m4
#           - m6
#       - m7
#       - Y - m5


def map_loop(root):
    loopnests = []
    state = {}
    stack = [root,]
    state[root] = "pending visit"
    while stack:
        node = stack[-1]
        #print(node, " is in ", state[node], " state")
    
        if state[node] == "visiting":
            state[node] = "done"
            stack.pop()
            assert loopnests[-1] == node
            loopnests.pop()
            #print("Done with node ", node)
            continue
    
        # Reach a leaf
        if G.out_degree[node] == 0:
            print("Instruction ", node, loopnests)
            stack.pop()
            continue
    
        loopnests.append(node)
        for _,child in G.out_edges(node):
            #print(node, " --> ", child)
            stack.append(child)
            state[child] = "pending visit"
    
        state[node] = "visiting"

for root in roots:
    map_loop(root)

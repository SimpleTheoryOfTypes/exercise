#!/usr/env/python3
import networkx as nx
G = nx.DiGraph()

engine_kind = {"n1":"load", \
               "n2":"load", \
               "n3":"add", \
               "n4":"mul", \
               "n5":"add", \
               "n6":"mul", \
               "n7":"store", \
               "n8":"mul", \
               "n9":"load", \
               "n10":"add", \
               "n11":"mul", \
               "n12":"store"}

G.add_edge("n1", "n3")
G.add_edge("n1", "n5")
G.add_edge("n1", "n6")
G.add_edge("n1", "n4")
G.add_edge("n5", "n8")
G.add_edge("n2", "n8")
G.add_edge("n8", "n10")
G.add_edge("n9", "n10")
G.add_edge("n10", "n11")
G.add_edge("n6", "n11")
G.add_edge("n11", "n12")
G.add_edge("n3", "n4")
G.add_edge("n4", "n7")

print(engine_kind)

stack = [node for node, in_degree in G.in_degree() if in_degree == 0]
count = dict(G.in_degree())


latency = {"add":2, "mul":2, "load":2, "store":1}

ASAP = {}
while stack:
    node = stack[-1]

    if (G.in_degree[node] == 0):
        ASAP[node] = 0
    else:
        possible_asaps = [ASAP[v] + latency[engine_kind[v]] for v,indegree in G.in_edges(node)]
        ASAP[node] = max(possible_asaps)
            
    stack.pop()

    for _,child in G.out_edges(node):
        count[child] -= 1
        if (count[child] == 0):
            stack.append(child)


print(ASAP)










#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List

#g = nx.read_adjlist('./sample.graph')
g = nx.DiGraph()

# Book example
#g.add_weighted_edges_from([("u", "v", 0.75), \
#                           ("v", "y", 0.75), \
#                           ("y", "x", 0.75), \
#                           ("u", "x", 0.75), \
#                           ("x", "v", 0.75), \
#                           ("w", "y", 0.75), \
#                           ("w", "z", 0.75), \
#                           ("z", "z", 0.75)  \
#                          ])

# Real diamond
g.add_weighted_edges_from([("u", "x", 0.75), \
                           ("u", "y", 0.75), \
                           ("u", "z", 0.75), \
                           ("x", "y", 0.75), \
                           ("y", "w", 0.75), \
                           ("z", "w", 0.75), \
                          ])

# Viz Method 1: use matplotlib
plt.subplot(121)
nx.draw(g, with_labels=True, font_weight='bold')
plt.show()

# Begin DFS
nodes = g
color = {}
depth = {}
finish = {}
parent = {}
for node in nodes:
  print(node)
  color[node] = 'White'

time = 0

def DFS_visit(g, color, depth, finish, parent, node):
  global time
  color[node] = 'Gray'
  time = time + 1
  depth[node] = time
  for child in iter(g[node]):
      if color[child] == 'White':
          print("Visiting \n", child)
          parent[child] = node
          DFS_visit(g, color, depth, finish, parent, child)
  color[node] = 'Black'
  time = time + 1
  finish[node] = time

def DFS_visit_iter(g, color, depth, finish, parent, node):
    global time
    time = time + 1
    depth[node] = time
    color[node] = 'Gray'
    # Using list as a stack:
    # https://docs.python.org/2/tutorial/datastructures.html#using-lists-as-stacks
    stack = [(node, iter(list(g[node])))] # i.e., (v, adj(v))

    topo_order = []

    while stack:
        cur_node = stack[-1][0]
        children = stack[-1][1] # iterator across children
        v = next(children, None)
        if v is None:
            time = time + 1
            color[cur_node] = 'Black'
            finish[cur_node] = time
            stack.pop()
            # Insert each vertex to the front of the topo_oder list as it 
            # finishes.
            topo_order = [cur_node] + topo_order
            continue

        if (color[v] == 'White'):
            parent[v] = node
            time = time + 1
            depth[v] = time
            color[v] = 'Gray'
            stack.append((v, iter(list(g[v]))))

    return topo_order

for node in nodes:
    if color[node] == 'White':
        print("Visiting \n - top level", node)
        #DFS_visit(g, color, depth, finish, parent, node)
        topo_order = DFS_visit_iter(g, color, depth, finish, parent, node)
        print('Topological order starting at [', node, "] is: \n    ", topo_order)
print(depth)
print(finish)
print(parent)

print("\n[Golden Reference]:")
print("""list(nx.dfs_edges(g, source="u"))""")
print(list(nx.dfs_edges(g, source="u")))

print("""list(nx.dfs_tree(g, source="u"))""")
print(list(nx.dfs_tree(g, source="u")))

print("""list(nx.dfs_labeled_edges(g, source="u"))""")
print(list(nx.dfs_labeled_edges(g, source="u")))
# Viz Method 2: use graphviz
#h = nx.nx_agraph.from_agraph(g)
#nx.write_dot(h, './graph.dot')

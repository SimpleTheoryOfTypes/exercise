#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt

#g = nx.read_adjlist('./sample.graph')
g = nx.DiGraph()
g.add_weighted_edges_from([("u", "v", 0.75), \
                           ("v", "y", 0.75), \
                           ("y", "x", 0.75), \
                           ("u", "x", 0.75), \
                           ("x", "v", 0.75), \
                           ("w", "y", 0.75), \
                           ("w", "z", 0.75), \
                           ("z", "z", 0.75)  \
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
    stack = [node]
    while len(stack) > 0:
        x = stack[-1]
        if color[x] == 'White':
            print("  visiting ", x)
            time = time + 1
            depth[x] = time
            color[x] = 'Gray'
        elif color[x] == 'Black':
            stack.pop()
            continue

        all_children_visited = True
        for child in iter(g[x]):
            if color[child] == 'White':
                all_children_visited = False
                parent[child] = x
                stack.append(child)
                print("  adding ", child, " | current stack: ", stack)

        if color[x] == 'Gray' and all_children_visited == True:
            color[x] = 'Black'
            time = time + 1
            finish[x] = time
            stack.pop()
            print("  popping ", x, " | current stack: ", stack)

for node in nodes:
    if color[node] == 'White':
        print("Visiting \n - top level", node)
        #DFS_visit(g, color, depth, finish, parent, node)
        DFS_visit_iter(g, color, depth, finish, parent, node)

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

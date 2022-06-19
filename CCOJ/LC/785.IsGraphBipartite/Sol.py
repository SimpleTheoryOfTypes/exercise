class Solution:
    def isBipartite(self, graph):
      N = len(graph)
      for n in range(0, N, 1):
        if not graph[n]:
          continue
        color = {}
        queue = [n]
        color[n] = 0
        while queue:
            u = queue.pop(0)
            c = 1 - color[u]
            for v in graph[u]:
              if (v in color):
                if color[v] == color[u]:
                  return False
              else:
                   color[v] = 1 - color[u]
                   queue.append(v)
      return True

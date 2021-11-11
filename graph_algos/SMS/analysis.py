import networkx as nx
import typing

# TODO: all these compute functions perform a walk thru
# either the topo ordering or reverse topo ordering. Refactor
# the topo ordering piece out as a common utility function,
def compute_asap(DFG: nx.DiGraph,
                 latency: typing.Dict[str, int],
                 engine_kind: typing.Dict[str,str]) -> typing.Dict[str, int]:

  stack = [node for node, in_degree in DFG.in_degree() if in_degree == 0]
  count = dict(DFG.in_degree())

  ASAP = {}
  while stack:
    u = stack[-1]

    if (DFG.in_degree[u] == 0):
      ASAP[u] = 0
    else:
      possible_asaps = [ASAP[v] + latency[engine_kind[v]] for v,indegree in DFG.in_edges(u)]
      ASAP[u] = max(possible_asaps)

    stack.pop()

    for _,child in DFG.out_edges(u):
      count[child] -= 1
      if (count[child] == 0):
        stack.append(child)

  return ASAP

def compute_alap(ASAP: typing.Dict[str, int],
                 DFG: nx.DiGraph,
                 latency: typing.Dict[str, int],
                 engine_kind: typing.Dict[str,str]) -> typing.Dict[str, int]:

  maxASAP = max(ASAP.values())

  rDFG = DFG.reverse(copy=True)

  stack = [node for node, in_degree in rDFG.in_degree() if in_degree == 0]
  count = dict(rDFG.in_degree())

  ALAP = {}
  while stack:
    u = stack[-1]

    if (rDFG.in_degree[u] == 0):
      ALAP[u] = maxASAP
    else:
      possible_alaps = [ALAP[v] - latency[engine_kind[u]] for v,indegree in rDFG.in_edges(u)]
      ALAP[u] = min(possible_alaps)

    stack.pop()

    for _,child in rDFG.out_edges(u):
      count[child] -= 1
      if (count[child] == 0):
        stack.append(child)

  return ALAP

def compute_mob(ALAP: typing.Dict[str, int],
                ASAP: typing.Dict[str, int]) -> typing.Dict[str, int]:

  assert len(ALAP) == len(ASAP), "ALAP and ASAP mismatch in size."
  MOB = {}
  for k,v in ALAP.items():
    u = k
    MOB[u] = ALAP[u] - ASAP[u]

  return MOB

def compute_depth(DFG: nx.DiGraph,
                  latency: typing.Dict[str, int],
                  engine_kind: typing.Dict[str,str]) -> typing.Dict[str, int]:

  stack = [node for node, in_degree in DFG.in_degree() if in_degree == 0]
  count = dict(DFG.in_degree())

  Depth = {}
  while stack:
    u = stack[-1]

    if (DFG.in_degree[u] == 0):
      Depth[u] = 0
    else:
      possible_depths = [Depth[v] + latency[engine_kind[v]] for v,indegree in DFG.in_edges(u)]
      Depth[u] = max(possible_depths)

    stack.pop()

    for _,child in DFG.out_edges(u):
      count[child] -= 1
      if (count[child] == 0):
        stack.append(child)

  return Depth 

def compute_height(DFG: nx.DiGraph,
                   latency: typing.Dict[str, int],
                   engine_kind: typing.Dict[str,str]) -> typing.Dict[str, int]:

  rDFG = DFG.reverse(copy=True)

  stack = [node for node, in_degree in rDFG.in_degree() if in_degree == 0]
  count = dict(rDFG.in_degree())

  Height = {}
  while stack:
    u = stack[-1]

    if (rDFG.in_degree[u] == 0):
      Height[u] = 0 
    else:
      possible_heights = [Height[v] + latency[engine_kind[u]] for v,indegree in rDFG.in_edges(u)]
      Height[u] = max(possible_heights)

    stack.pop()

    for _,child in rDFG.out_edges(u):
      count[child] -= 1
      if (count[child] == 0):
        stack.append(child)

  return Height 



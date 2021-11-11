import networkx as nx

def build_fig2():
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

  # Fig. 2
  # Hardware configurations
  hw_config = {"add": 1, "mul": 1, "LDST": 2}
  # Latencies
  latency = {"add":2, "mul":2, "load":2, "store":1}

  return G, engine_kind, hw_config, latency


import networkx as nx
from collections import Counter

def create_subgraph_using_nodes(graph, nodes):
    return graph.subgraph(nodes)

def create_subgraph_with_x_nodes(graph, x):
    clus_coef = nx.clustering(graph)
    c = Counter(clus_coef)
    nodes = c.most_common(x)
    return graph.subgraph(nodes)
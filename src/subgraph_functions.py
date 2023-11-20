import networkx as nx
from collections import Counter

def create_subgraph_using_nodes(graph, nodes):
    return graph.subgraph(nodes)

def create_subgraph_with_x_nodes(graph, x):
    count = 0
    nodes = []
    for node in graph:
        if count < x:
            nodes.append(node)
            count += 1
        else:
            break
    return graph.subgraph(nodes)
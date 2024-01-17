import networkx as nx
import torch
import torch_geometric
from torch_geometric.loader import LinkNeighborLoader
print("import complete")

graph_train = nx.Graph()
graph_train.add_nodes_from([0,1,2,3])
graph_train.add_edges_from([(0,1),(1,2),(2,3),(3,0)])

graph_test = nx.Graph()
graph_test .add_nodes_from([0,1,2,3])
graph_test.add_edge(0,2)

graph_val = nx.Graph()
graph_val.add_nodes_from([0,1,2,3])
graph_val.add_edges_from([(0,2),(1,3)])
print("created graphs")

torch_train = from_networkx(graph_train)
torch_test = from_networkx(graph_test)
torch_val = from_networkx(graph_val)
print("converted graphs")

import dataset_functions as df
from tqdm import tqdm
import networkx as nx
import pickle

before_2008_graph_path = df.path_to_data(1, "before_2008_graph.pkl")
after_2008_graph_path = df.path_to_data(1, "after_2008_graph.pkl")

with open(before_2008_graph_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_graph_path, 'rb') as fp:
    after_2008_graph = pickle.load(fp)

print("original graph number of edges:", after_2008_graph.size())
print("original graph number of nodes:", after_2008_graph.number_of_nodes())
print(" ")

# Only nodes that existed before 2008
# Only edges that didn't exist before 2008
nodes_list = []
for node in tqdm(after_2008_graph):
    if before_2008_graph.has_node(node):
        for sub_node in nodes_list:
            if before_2008_graph.has_edge(node, sub_node) and after_2008_graph.has_edge(node, sub_node):
                after_2008_graph.remove_edge(node, sub_node)
        nodes_list.append(node)

filtered_after_2008_graph = after_2008_graph.subgraph(nodes_list)

after_2008_graph_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")
with open(after_2008_graph_filtered_path, 'wb') as fp:
    pickle.dump(filtered_after_2008_graph, fp)

print("filtered after 2008 graph number of edges:", filtered_after_2008_graph.size())
print("filtered after 2008 graph number of nodes:", filtered_after_2008_graph.number_of_nodes())
print(" ")

for node in tqdm(after_2008_graph):
    if not before_2008_graph.has_node(node):
        print(node)

'''
Results:
original graph number of edges: 1281979
original graph number of nodes: 9654

filtered after 2008 graph number of edges: 362987
filtered after 2008 graph number of nodes: 9653

filtered before 2008 graph number of edges: 5338564
filtered before 2008 graph number of nodes: 9653

Molecular Dynamics Simulation
'''
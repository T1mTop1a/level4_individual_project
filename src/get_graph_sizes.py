import pickle
import dataset_functions as df

before_2008_subgraph_path = df.path_to_data(1, "before_2008_graph.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_graph.pkl")
after_2008_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")

with open(before_2008_subgraph_path, 'rb') as bg:
    before_2008_graph = pickle.load(bg)
with open(after_2008_subgraph_path, 'rb') as ag:
    after_2008_graph = pickle.load(ag)
with open(after_2008_filtered_path, 'rb') as fg:
    after_2008_graph_filtered = pickle.load(fg)
print("opened data")

print(len(before_2008_graph.edges))
print(len(after_2008_graph.edges))
print(len(after_2008_graph.edges)/len(before_2008_graph.edges))
print()

print(len(before_2008_graph.nodes))
print(len(after_2008_graph.nodes))
print(len(after_2008_graph.nodes)/len(before_2008_graph.nodes))
print()

print(len(after_2008_graph_filtered.nodes))
print(len(after_2008_graph_filtered.edges))
print(len(after_2008_graph_filtered.edges)/len(before_2008_graph.edges))
print()
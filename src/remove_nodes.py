import dataset_functions as df
import pickle

before_2008_graph_path = df.path_to_data(1, "before_2008_graph.pkl")
after_2008_graph_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")

with open(before_2008_graph_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_graph_filtered_path, 'rb') as fp:
    after_2008_graph_filtered = pickle.load(fp)

print(after_2008_graph_filtered.nodes)
nodes_to_remove = []
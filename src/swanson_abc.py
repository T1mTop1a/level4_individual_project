import subgraph_functions as sf
import dataset_functions as df
import knowledge_graph_functions as kgf # If you want to display
import pickle

before_2008_graph_filtered_path = df.path_to_data(1, "before_2008_graph_filtered.pkl")
after_2008_graph_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")

with open(before_2008_graph_filtered_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_graph_filtered_path, 'rb') as fp:
    after_2008_graph = pickle.load(fp)
print("opened data")

after_2008_subgraph = sf.create_subgraph_with_x_nodes(after_2008_graph, 10)
print("created after subgraph")

nodes = []
for node in after_2008_subgraph:
    nodes.append(node)
before_2008_subgraph = sf.create_subgraph_using_nodes(before_2008_graph, nodes)
print("created before subgraph")
print(" ")

print("before 2008 subgraph number of edges:", before_2008_subgraph.size())
print("before 2008 subgraph number of nodes:", before_2008_subgraph.number_of_nodes())
print(" ")
print("after 2008 subgraph number of edges:", after_2008_subgraph.size())
print("after 2008 subgraph number of nodes:", after_2008_subgraph.number_of_nodes())

# kgf.display_graph(before_2008_subgraph, "before 2008 subgraph")
# kgf.display_graph(after_2008_subgraph, "after 2008 subgraph")
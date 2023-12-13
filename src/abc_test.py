import subgraph_functions as sf
import dataset_functions as df
import abc_functions as af
import knowledge_graph_functions as kgf # If you want to display
import pickle
import networkx as nx

before_2008_graph_path = df.path_to_data(1, "before_2008_graph.pkl")
after_2008_graph_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")

with open(before_2008_graph_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_graph_filtered_path, 'rb') as fp:
    after_2008_graph = pickle.load(fp)
print("opened data")

after_2008_subgraph = sf.create_subgraph_with_x_nodes(after_2008_graph, 10)
print("created after subgraph")

nodes = []
for node in after_2008_subgraph:
    nodes.append(node)

before_2008_subgraph = before_2008_graph.subgraph(nodes)

biodiversity_c = af.dict_of_c_given_a_weight(before_2008_graph, "Biodiversity")
biodiversity_c_sorted = af.sort_c(biodiversity_c)

biodiversity_after_neighbours = list(after_2008_graph.neighbors("Biodiversity"))

for i in range(10):
    name = biodiversity_c_sorted[i][0]
    if name not in biodiversity_after_neighbours:
        print(name, "is a new link, and was in position", i)
    else:
        print(name, "is not a new link")
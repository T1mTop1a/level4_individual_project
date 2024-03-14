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

c_weight = af.dict_of_c_given_a_weight(before_2008_subgraph, "United States Public Health Service")
c_sorted_weight = af.sort_c(c_weight)

c_frequency = af.dict_of_c_given_a_frequency(before_2008_subgraph, "United States Public Health Service")
c_sorted_frequency = af.sort_c(c_frequency)

c_combined = af.dict_of_c_given_a_weight_frequency(before_2008_subgraph, "United States Public Health Service")
c_sorted_combined = af.sort_c(c_combined)

print("sorted by weight")
for i in range(len(c_sorted_weight)):
    print("pos:", i, "weight:",c_sorted_weight[i][1] ,"name:", c_sorted_weight[i][0])

print("sorted by frequency")
for i in range(len(c_sorted_frequency)):
    print("pos:", i, "frequency:",c_sorted_frequency[i][1] ,"name:", c_sorted_frequency[i][0])

print("sorted by combined")
for i in range(len(c_sorted_combined)):
    print("pos:", i, "score:",c_sorted_combined[i][1] ,"name:", c_sorted_combined[i][0])
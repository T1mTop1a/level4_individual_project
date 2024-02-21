import dataset_functions as df
import subgraph_functions as sf
import pickle

before_2008_graph_filtered_path = df.path_to_data(1, "before_2008_graph.pkl")
after_2008_graph_filtered_path = df.path_to_data(1, "after_2008_graph_filtered.pkl")

with open(before_2008_graph_filtered_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_graph_filtered_path, 'rb') as fp:
    after_2008_graph = pickle.load(fp)
print("opened data")

after_2008_subgraph = sf.create_subgraph_with_x_nodes(after_2008_graph, 1000)
before_2008_subgrpah = before_2008_subgraph = before_2008_graph.subgraph(after_2008_subgraph.nodes())
print("created subgraphs")

print("before 2008 graph edges:", after_2008_subgraph.size())
print("after 2008 graph edges:", before_2008_subgraph.size())

after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph_1k.pkl")
with open(after_2008_subgraph_path, 'wb') as fp:
    pickle.dump(after_2008_subgraph, fp)

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph_1k.pkl")
with open(before_2008_subgraph_path, 'wb') as fp:
    pickle.dump(before_2008_subgraph, fp)

print("dumped data")
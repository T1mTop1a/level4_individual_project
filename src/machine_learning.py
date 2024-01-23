import dataset_functions as df
import pickle
import torch
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

print("import complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph.pkl")

with open(before_2008_subgraph_path, 'rb') as bg:
    before_2008_graph = pickle.load(bg)
with open(after_2008_subgraph_path, 'rb') as ag:
    after_2008_graph = pickle.load(ag)
print("opened data")

pytorch_graph_before = from_networkx(before_2008_graph)
pytorch_graph_after = from_networkx(after_2008_graph)
print("converted graphs")

transform = T.RandomLinkSplit(
    num_val = 0.49,
    num_test = 0.49,
    add_negative_train_samples = False,
    is_undirected = True,
)
train_graph, validate_graph, test_graph = transform(pytorch_graph_after)
print("split validate, test data")

training_load = LinkNeighborLoader(
    data = pytorch_graph_before,
    batch_size = 128,
    num_neighbors = [20,10],
    shuffle = True,
)
print("Defined batch loader")


print("======================================")
print(pytorch_graph_before.edge_index)
print(pytorch_graph_before.num_edges)

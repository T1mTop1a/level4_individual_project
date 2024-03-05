import dataset_functions as df
import numpy as np
import statistics
import pickle
import torch
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
print("import complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph_1k.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph_1k.pkl")

with open(before_2008_subgraph_path, 'rb') as bg:
    before_2008_graph = pickle.load(bg)
with open(after_2008_subgraph_path, 'rb') as ag:
    after_2008_graph = pickle.load(ag)
print("opened data")

pytorch_graph_before = Data()
pytorch_graph_after = Data()
print("converted graphs")

pytorch_graph_before.edge_index = torch.tensor([[0,0,0,0,0,0,0,0,1,2,3,4],[2,3,4,5,6,7,8,9,5,5,5,5]])
pytorch_graph_after.edge_index = torch.tensor([[0,1],[1,5]])

pytorch_graph_before.node_id = torch.arange(10)
pytorch_graph_after.node_id = torch.arange(10)

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_nodes, n_factors=64):
        super().__init__()
        self.node_factors = torch.nn.Embedding(n_nodes, n_factors)


    def forward(self, node1, node2):
        return (self.node_factors(node1) * self.node_factors(node2)).sum(1)
    
print("defined model")
edge_index_train_pos = torch.ones(8)
edge_index_train_neg = torch.zeros(4)
edge_index_train_tot = torch.cat((edge_index_train_pos, edge_index_train_neg))

num_edges_train = edge_index_train_tot.size(dim=0)

sum = 0
for i in range(100):
    model = MatrixFactorization(n_nodes=1000)

    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

    for i in range(num_edges_train):
        optimizer.zero_grad()

        value = torch.FloatTensor([edge_index_train_tot[i].item()])
        row = torch.LongTensor([pytorch_graph_before.edge_index[0][i].item()])
        col = torch.LongTensor([pytorch_graph_before.edge_index[1][i].item()])

        prediction = model(row, col)
        loss = loss_func(prediction, value)

        loss.backward()

        optimizer.step()

    edge_index_val_pos = torch.ones(1)
    edge_index_val_neg = torch.zeros(1)
    edge_index_val_tot = torch.cat((edge_index_val_pos, edge_index_val_neg))

    num_edges_test = edge_index_val_tot.size(dim=0)

    predictions = []
    for i in range(num_edges_test):

        row = torch.LongTensor([pytorch_graph_after.edge_index[0][i].item()])
        col = torch.LongTensor([pytorch_graph_after.edge_index[1][i].item()])

        prediction = model(row, col)
        predictions.append(prediction.item())

    ground_truth = edge_index_val_tot
    auc = roc_auc_score(ground_truth, predictions)
    sum+=auc

print(sum/100)
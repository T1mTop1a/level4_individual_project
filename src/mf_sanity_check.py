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
from torch.utils.data import TensorDataset, DataLoader
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

pytorch_graph_before.edge_index = torch.tensor([[0,0,0,0,1,1,1,1,6,6,6,6],[2,3,4,5,2,3,4,5,2,3,4,5]])
pytorch_graph_after.edge_index = torch.tensor([[0,0,0,0,1,1,1,1,6,6,6,6],[2,3,4,5,2,3,4,5,2,3,4,5]])

#pytorch_graph_before.edge_index = torch.tensor([[0,0,0,1,1,1,0,5,5,5],[2,3,4,2,3,4,6,2,3,4]])
#pytorch_graph_after.edge_index = torch.tensor([[1,5],[6,6]])

pytorch_graph_before.node_id = torch.arange(7)
pytorch_graph_after.node_id = torch.arange(7)

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

dataset_edges = torch.transpose(pytorch_graph_before.edge_index, 0, 1)
train_dataset = TensorDataset(dataset_edges, edge_index_train_tot)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = MatrixFactorization(n_nodes=7)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(1,101):
    for batch_edges, batch_labels in train_loader:
        optimizer.zero_grad()
        
        values = []
        rows = []
        cols = []
        for i in range(batch_labels.size(dim=0)):

            values.append(batch_labels[i].item())
            rows.append(batch_edges[i][0].item())
            cols.append(batch_edges[i][1].item())

        value = torch.FloatTensor(values)
        row = torch.LongTensor(rows)
        col = torch.LongTensor(cols)

        prediction = model(row, col)
        loss = F.binary_cross_entropy_with_logits(prediction, value)

        loss.backward()

        optimizer.step()
    print(f"Epoch: {epoch:03d}")

print("Completed training")


edge_index_val_pos = torch.ones(8)
edge_index_val_neg = torch.zeros(4)
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

print(auc)
import dataset_functions as df
import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
import torch
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data

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

pre_neg_sampling_edges_before_graph =  pytorch_graph_before.num_edges
pre_neg_sampling_edges_after_graph = pytorch_graph_after.num_edges

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, emb, edge_index):
        edge_feat_1 = emb[edge_index[0]]
        edge_feat_2 = emb[edge_index[1]]
        return (edge_feat_1 * edge_feat_2).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.emb = torch.nn.Embedding(pytorch_graph_before.node_id.size(dim=0), hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
    
    def forward(self, data):
        x = self.emb(data.node_id)
        x = self.gnn(x, data.edge_index)
        pred = self.classifier(
            x,
           data.edge_index,
        )
        return pred

model = Model(hidden_channels=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
edge_index_train_pos = torch.ones(7)
edge_index_train_neg = torch.zeros(3)
edge_index_train_tot = torch.cat((edge_index_train_pos, edge_index_train_neg))

sum = 0
for i in range(100):
    for epoch in range(1,6):
        for batch_edges, batch_labels in train_loader:
            total_loss = total_examples = 0
            optimizer.zero_grad()
            pytorch_graph_before.to(device)
            pred = model(pytorch_graph_before)
            ground_truth = edge_index_train_tot
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()


    preds_val = []
    ground_truths_val = []
    edge_index_val_pos = torch.ones(1)
    edge_index_val_neg = torch.zeros(1)
    edge_index_val_tot = torch.cat((edge_index_val_pos, edge_index_val_neg))
    with torch.no_grad():
        pytorch_graph_after.to(device)
        preds_val.append(model(pytorch_graph_after))
        ground_truths_val.append(edge_index_val_tot)
    pred_val = torch.cat(preds_val, dim=0).cpu().numpy()
    ground_truth_val = torch.cat(ground_truths_val, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth_val, pred_val)
    sum += auc

print(sum/100)
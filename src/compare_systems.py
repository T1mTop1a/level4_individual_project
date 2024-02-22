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

# =========================== DATA PREP ===========================

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph_1k.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph_1k.pkl")

with open(before_2008_subgraph_path, 'rb') as bg:
    before_2008_graph = pickle.load(bg)
with open(after_2008_subgraph_path, 'rb') as ag:
    after_2008_graph = pickle.load(ag)
print("opened data")

pytorch_graph_before = from_networkx(before_2008_graph)
pytorch_graph_after = from_networkx(after_2008_graph)
print("converted graphs")

values_before = {}
for i in pytorch_graph_before.edge_index[0]:
    val = i.item()
    if val in values_before:
        values_before[val] += 1
    else:
        values_before[val] = 1

values_after = {}
for i in pytorch_graph_after.edge_index[0]:
    val = i.item()
    if val in values_after:
        values_after[val] += 1
    else:
        values_after[val] = 1

top_values_before, top_values_after= {}, {}
for i in range(100):
    top_before = max(values_before, key=values_before.get)
    top_values_before[top_before] = values_before[top_before]
    values_before.pop(top_before, None)

    top_after = max(values_after, key=values_after.get)
    top_values_after[top_after] = values_after[top_after]
    values_after.pop(top_after, None)

shared = {}
for node, val in top_values_before.items():
    if node in top_values_after:
        shared[node] = (val,top_values_after[node])
print("Find shared popular nodes")

edge_tensors = {}
for key in shared:
    edges = {}
    indecies = (pytorch_graph_before.edge_index[0] == key).nonzero(as_tuple=True)[0]
    for index in indecies:
        ind = index.item()
        edges[pytorch_graph_before.edge_index[1][ind].item()] = 0
    new_edges = []
    for i in range(1000):
        if i not in edges:
            new_edges.append(i)
    new_edges.remove(key)
    node_array = [key] * len(new_edges)
    tensor = torch.tensor([node_array, new_edges])
    edge_tensors[key] = tensor

edge_values = {}
for key, edge_info in edge_tensors.items():
    edges = {}
    indecies = (pytorch_graph_after.edge_index[0] == key).nonzero(as_tuple=True)[0]
    for index in indecies:
        ind = index.item()
        edges[pytorch_graph_before.edge_index[1][ind].item()] = 0
    
    values = []
    for node in edge_info[1]:
        if node.item() in edges:
            values.append(1)
        else:
            values.append(0)
    tensor = torch.tensor(values)
    edge_values[key] = tensor

node_graphs = {}
for key, val in edge_tensors.items():
    graph = Data(edge_index = val)
    graph.node_id = torch.arange(1000)
    graph.ground_truth = edge_values[key]
    node_graphs[key] = graph

print(node_graphs.keys())


pytorch_graph_before.node_id = torch.arange(1000)
pre_neg_sampling_edges_before_graph =  pytorch_graph_before.num_edges
before_negative_edges = negative_sampling(pytorch_graph_before.edge_index, force_undirected = True)
pytorch_graph_before.edge_index = torch.cat((pytorch_graph_before.edge_index, before_negative_edges),1)

# =========================== GRAPH NEURAL NETWORK ===========================

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
print("define model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
edge_index_train_pos = torch.ones(pre_neg_sampling_edges_before_graph)
edge_index_train_neg = torch.zeros(pytorch_graph_before.num_edges - pre_neg_sampling_edges_before_graph)
edge_index_train_tot = torch.cat((edge_index_train_pos, edge_index_train_neg))
for epoch in range(1,6):
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
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
print("training complete")

gnn_output = {}
for key, value in node_graphs.items():
    preds_val = []
    ground_truths_val = []
    with torch.no_grad():
        value.to(device)
        preds_val.append(model(value))
        ground_truths_val.append(value.ground_truth)
    pred_val = torch.cat(preds_val, dim=0).cpu().numpy()
    ground_truth_val = torch.cat(ground_truths_val, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth_val, pred_val)

    results = {}
    results["ground_truth"] = ground_truth_val
    results["prediction"] = pred_val
    results["auc"] = auc
    results["edge_index"] = value.edge_index
    results["positive_edges"] = shared[key][1]
    gnn_output[key] = results

# =========================== MATRIX FACTORIZATION ===========================

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_nodes, n_factors=64):
        super().__init__()
        self.node_factors = torch.nn.Embedding(n_nodes, n_factors)


    def forward(self, node1, node2):
        return (self.node_factors(node1) * self.node_factors(node2)).sum(1)
    
model = MatrixFactorization(n_nodes=1000)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
print("defined model")

num_edges_train = edge_index_train_tot.size(dim=0)

for i in range(num_edges_train):
    optimizer.zero_grad()
    
    value = torch.FloatTensor([edge_index_train_tot[i].item()])
    row = torch.LongTensor([pytorch_graph_before.edge_index[0][i].item()])
    col = torch.LongTensor([pytorch_graph_before.edge_index[1][i].item()])

    prediction = model(row, col)
    loss = loss_func(prediction, value)

    loss.backward()

    optimizer.step()
print("Completed training")

mf_output = {}
for key, graph in node_graphs.items():
    predictions = []
    for i in range(graph.num_edges):

        row = torch.LongTensor([graph.edge_index[0][i].item()])
        col = torch.LongTensor([graph.edge_index[1][i].item()])

        prediction = model(row, col)
        predictions.append(prediction.item())

    ground_truth = graph.ground_truth
    auc = roc_auc_score(ground_truth, predictions)
    results = {}
    results["ground_truth"] = ground_truth
    results["prediction"] = predictions
    results["auc"] = auc
    results["edge_index"] = graph.edge_index
    results["positive_edges"] = shared[key][1]
    mf_output[key] = results

# =========================== ABC SYSTEM ===========================
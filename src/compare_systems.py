import dataset_functions as df
import abc_functions as af
import pickle
import heapq
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
from torch.utils.data import TensorDataset, DataLoader
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
    positive = 0
    for node in edge_info[1]:
        if node.item() in edges:
            values.append(1)
            positive += 1
        else:
            values.append(0)
    tensor = torch.tensor(values)
    info = {}
    info["values"] = tensor
    info["positive"] = positive

    edge_values[key] = info

node_graphs = {}
for key, val in edge_tensors.items():
    graph = Data(edge_index = val)
    graph.node_id = torch.arange(1000)
    graph.ground_truth = edge_values[key]["values"]
    graph.positives = edge_values[key]["positive"]
    node_graphs[key] = graph

print("data prep complete")

# =========================== ABC SYSTEM ===========================    

abc_output = {}
for key, value in node_graphs.items():
    abc_predictions = []
    a_indecies = (pytorch_graph_before.edge_index[0] == key).nonzero(as_tuple=True)[0]
    ab_edges = {}
    for a_index in a_indecies:
        a_ind = a_index.item()
        ab_edges[pytorch_graph_before.edge_index[1][a_ind].item()] = 0

    for target in value.edge_index[1]:
        c_node = target.item()
        indecies = (pytorch_graph_before.edge_index[0] == c_node).nonzero(as_tuple=True)[0]
        bc_edges = {}
        for index in indecies:
            ind = index.item()
            bc_edges[pytorch_graph_before.edge_index[1][ind].item()] = 0

        b_count = 0
        for connection in bc_edges:
            if connection in ab_edges:
                b_count +=1

        abc_predictions.append(b_count)


    abc_output[key] = abc_predictions

print("abc complete")

# =========================== GRAPH NEURAL NETWORK ===========================

pytorch_graph_before.node_id = torch.arange(1000)
pre_neg_sampling_edges_before_graph =  pytorch_graph_before.num_edges
before_negative_edges = negative_sampling(pytorch_graph_before.edge_index, force_undirected = True)
pytorch_graph_before.edge_index = torch.cat((pytorch_graph_before.edge_index, before_negative_edges),1)

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

    gnn_output[key] = pred_val

print ("GNN complete")
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

dataset_edges = torch.transpose(pytorch_graph_before.edge_index, 0, 1)
train_dataset = TensorDataset(dataset_edges, edge_value_train_tot)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(1,6):
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

mf_output = {}
for key, graph in node_graphs.items():
    predictions = []
    for i in range(graph.num_edges):

        row = torch.LongTensor([graph.edge_index[0][i].item()])
        col = torch.LongTensor([graph.edge_index[1][i].item()])

        prediction = model(row, col)
        predictions.append(prediction.item())
    mf_output[key] = predictions

print("Matrix Factorization complete")

# =========================== EVALUATION ===========================
def relative_score(score, x, positives, length):
    return (score / x) / (positives / length)

def score_topx(results, original, x):
    topx = heapq.nlargest(x, range(len(results)), results.__getitem__)
    score = 0
    for index in topx:
        if original.ground_truth[index].item() == 1:
            score +=1
    
    rel_score = (score / x) / (original.positives / list(original.ground_truth.size())[0])
    return score, rel_score
    

print("Evaluation nodes info")    
print("node id || total edges || positive edges")
for key, value in node_graphs.items():
    print(key, " || ", list(value.ground_truth.size())[0], " || ", value.positives)
print("\n")
print("\n")

for key, value in node_graphs.items():
    abc_result = abc_output[key]
    gnn_result = gnn_output[key]
    mf_result = mf_output[key]

    x_values = [1,10,20,50,100]

    print("node: ", key)
    for x_val in x_values:
        abc_top, abc_top_rel = score_topx(abc_result, value, x_val)
        gnn_top, gnn_top_rel = score_topx(gnn_result, value, x_val)
        mf_top, mf_top_rel = score_topx(mf_result, value, x_val)

        print("top x || number of positive returns || score relative to x and positive ratio")
        print(x_val, "|| abc score", abc_top, "|| abc relative score", abc_top_rel)
        print(x_val, "|| gnn score", gnn_top, "|| gnn relative score", gnn_top_rel)
        print(x_val, "|| mf score", mf_top, "|| mf relative score", mf_top_rel, "\n")

print("evaluation done")

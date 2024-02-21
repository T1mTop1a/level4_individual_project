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

print("import complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

pytorch_graph_before.node_id = torch.arange(1000)
pytorch_graph_after.node_id = torch.arange(1000)

pre_neg_sampling_edges_before_graph =  pytorch_graph_before.num_edges
pre_neg_sampling_edges_after_graph = pytorch_graph_after.num_edges

before_negative_edges = negative_sampling(pytorch_graph_before.edge_index, force_undirected = True)
pytorch_graph_before.edge_index = torch.cat((pytorch_graph_before.edge_index, before_negative_edges),1)

after_negative_edges = negative_sampling(pytorch_graph_after.edge_index, force_undirected = True)
pytorch_graph_after.edge_index = torch.cat((pytorch_graph_after.edge_index, after_negative_edges),1)

transform = T.RandomLinkSplit(
    num_val = 0.49,
    num_test = 0.49,
    add_negative_train_samples = False,
    is_undirected = True,
)
train_graph, validate_graph, test_graph = transform(pytorch_graph_after)

validate_graph.node_id = torch.arange(1000)
pre_neg_sampling_edges_validate_graph = validate_graph.num_edges
validate_negative_edges = negative_sampling(validate_graph.edge_index, force_undirected = True)
validate_graph.edge_index = torch.cat((validate_graph.edge_index, validate_negative_edges),1)

test_graph.node_id = torch.arange(1000)
pre_neg_sampling_edges_test_graph = test_graph.num_edges
test_negative_edges = negative_sampling(test_graph.edge_index, force_undirected = True)
test_graph.edge_index = torch.cat((test_graph.edge_index, test_negative_edges),1)
print("split validate, test data")

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

preds_val = []
ground_truths_val = []
edge_index_val_pos = torch.ones(pre_neg_sampling_edges_after_graph)
edge_index_val_neg = torch.zeros(pytorch_graph_after.num_edges - pre_neg_sampling_edges_after_graph)
edge_index_val_tot = torch.cat((edge_index_val_pos, edge_index_val_neg))
with torch.no_grad():
    pytorch_graph_after.to(device)
    preds_val.append(model(pytorch_graph_after))
    ground_truths_val.append(edge_index_val_tot)
pred_val = torch.cat(preds_val, dim=0).cpu().numpy()
ground_truth_val = torch.cat(ground_truths_val, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth_val, pred_val)
print()
print(f"Validation AUC: {auc:.4f}")

pred_tensor = 
for key in shared:

    pass







'''
class EarlyStopping:
    def __init__(self, wait=5, min_delta=0):

        self.wait = wait
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

early_stopping = EarlyStopping(wait=3, min_delta=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

edge_index_train_pos = torch.ones(pre_neg_sampling_edges_before_graph)
edge_index_train_neg = torch.zeros(pytorch_graph_before.num_edges - pre_neg_sampling_edges_before_graph)
edge_index_train_tot = torch.cat((edge_index_train_pos, edge_index_train_neg))

preds_val = []
ground_truths_val = []
edge_index_val_pos = torch.ones(pre_neg_sampling_edges_validate_graph)
edge_index_val_neg = torch.zeros(validate_graph.num_edges - pre_neg_sampling_edges_validate_graph)
edge_index_val_tot = torch.cat((edge_index_val_pos, edge_index_val_neg))

for epoch in range(1,101):
    total_loss = total_examples = 0
    optimizer.zero_grad()
    pytorch_graph_before.to(device)
    pred = model(pytorch_graph_before)
    ground_truth = edge_index_train_tot
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Validation Loss: {total_loss / total_examples:.4f}")

    with torch.no_grad(): 
        total_loss_val = total_examples_val = 0
        pytorch_graph_after.to(device)
        preds_val = model(validate_graph)
        ground_truths_val = edge_index_val_tot

    loss_val = F.binary_cross_entropy_with_logits(preds_val, ground_truths_val)
    combined_loss = loss + loss_val
    combined_loss.backward()
    optimizer.step()
    total_loss_val += float(loss_val) * preds_val.numel()
    total_examples_val += preds_val.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss_val / total_examples_val:.4f}")

    early_stopping(loss, loss_val)
    if early_stopping.early_stop:
      print("We are at epoch:", epoch)
      break
print("training complete")

preds_test = []
ground_truths_test = []
edge_index_test_pos = torch.ones(pre_neg_sampling_edges_test_graph)
edge_index_test_neg = torch.zeros(test_graph.num_edges - pre_neg_sampling_edges_test_graph)
edge_index_test_tot = torch.cat((edge_index_test_pos, edge_index_test_neg))

with torch.no_grad():
    test_graph.to(device)
    preds_test.append(model(test_graph))
    ground_truths_test.append(edge_index_test_tot)
pred_test = torch.cat(preds_test, dim=0).cpu().numpy()
ground_truth_test = torch.cat(ground_truths_test, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth_test, pred_test)
print()
print(f"Validation AUC: {auc:.4f}")
'''
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

pre_neg_sampling_edges_before_graph =  pytorch_graph_before.num_edges
pre_neg_sampling_edges_after_graph = pytorch_graph_after.num_edges

before_negative_edges = negative_sampling(pytorch_graph_before.edge_index, force_undirected = True)
after_negative_edges = negative_sampling(pytorch_graph_after.edge_index, force_undirected = True)
pytorch_graph_before.edge_index = torch.cat((pytorch_graph_before.edge_index, before_negative_edges),1)
pytorch_graph_after.edge_index = torch.cat((pytorch_graph_after.edge_index, after_negative_edges),1)

print("added negative edges")

pytorch_graph_before.node_id = torch.arange(500)
pytorch_graph_after.node_id = torch.arange(500)

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_nodes, n_factors=64):
        super().__init__()
        self.node_factors = torch.nn.Embedding(n_nodes, n_factors)


    def forward(self, node1, node2):
        return (self.node_factors(node1) * self.node_factors(node2)).sum(1)
    
model = MatrixFactorization(n_nodes=500)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
print("defined model")

edge_value_train_pos = torch.ones(pre_neg_sampling_edges_before_graph)
edge_value_train_neg = torch.zeros(pytorch_graph_before.num_edges - pre_neg_sampling_edges_before_graph)
edge_value_train_tot = torch.cat((edge_value_train_pos, edge_value_train_neg))

num_edges_train = edge_value_train_tot.size(dim=0)

for i in range(num_edges_train):

    optimizer.zero_grad()
    
    value = torch.FloatTensor([edge_value_train_tot[i].item()])
    row = torch.LongTensor([pytorch_graph_before.edge_index[0][i].item()])
    col = torch.LongTensor([pytorch_graph_before.edge_index[1][i].item()])

    prediction = model(row, col)
    loss = loss_func(prediction, value)

    loss.backward()

    optimizer.step()

print("Completed training")

edge_value_test_pos = torch.ones(pre_neg_sampling_edges_after_graph)
edge_value_test_neg = torch.zeros(pytorch_graph_after.num_edges - pre_neg_sampling_edges_after_graph)
edge_value_test_tot = torch.cat((edge_value_test_pos, edge_value_test_neg))

num_edges_test = edge_value_test_tot.size(dim=0)

predictions = []
for i in range(num_edges_test):

    value = torch.FloatTensor([edge_value_test_tot[i].item()])
    row = torch.LongTensor([pytorch_graph_after.edge_index[0][i].item()])
    col = torch.LongTensor([pytorch_graph_after.edge_index[1][i].item()])

    prediction = model(row, col)
    predictions.append(prediction.item())

tp = 0
fp = 0
tn = 0
fn = 0

cut_off = statistics.mean(predictions)
for i in range(len(predictions)):
    if predictions[i] >= cut_off and edge_value_test_tot[i] == 1:
        tp += 1
    elif predictions[i] >= cut_off and edge_value_test_tot[i] == 0:
        fp += 1
    elif predictions[i] < cut_off and edge_value_test_tot[i] == 0:
        tn += 1
    else:
        fn += 1
        


accuracy = (tp + tn) / (tp + tn + fn + fp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
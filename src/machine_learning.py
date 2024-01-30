import dataset_functions as df
import pickle
from sklearn.metrics import roc_auc_score
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

print(pytorch_graph_after.has_isolated_nodes())

pytorch_graph_before.node_id = pytorch_graph_before.edge_index[0].unique()
pytorch_graph_after.node_id = pytorch_graph_after.edge_index[0].unique()
print(pytorch_graph_after)

transform = T.RandomLinkSplit(
    num_val = 0.49,
    num_test = 0.49,
    add_negative_train_samples = False,
    is_undirected = True,
)
train_graph, validate_graph, test_graph = transform(pytorch_graph_after)

validate_graph.node_id = validate_graph.edge_index[0].unique()
test_graph.node_id = test_graph.edge_index[0].unique()
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
            pytorch_graph_before.edge_index,
        )
        return pred

model = Model(hidden_channels=64)
print("define model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
edge_index_train = torch.ones(pytorch_graph_before.num_edges)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    optimizer.zero_grad()
    pytorch_graph_before.to(device)
    pred = model(pytorch_graph_before)
    ground_truth = edge_index_train
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
print("training complete")

preds_val = []
ground_truths_val = []
edge_index_validate = torch.ones(pytorch_graph_after.num_edges)
with torch.no_grad():
    pytorch_graph_after.to(device)
    preds_val.append(model(pytorch_graph_after))
    ground_truths_val.append(edge_index_validate.edge_label)
pred_val = torch.cat(preds_val, dim=0).cpu().numpy()
ground_truth_val = torch.cat(ground_truths_val, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth_val, pred_val)
print()
print(f"Validation AUC: {auc:.4f}")
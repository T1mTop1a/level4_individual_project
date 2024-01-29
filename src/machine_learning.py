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

print(pytorch_graph_before.edge_label_index)

pytorch_graph_before.node_id = pytorch_graph_before.edge_index[0].unique()

transform = T.RandomLinkSplit(
    num_val = 0.49,
    num_test = 0.49,
    add_negative_train_samples = False,
    is_undirected = True,
)
train_graph, validate_graph, test_graph = transform(pytorch_graph_after)
print("split validate, test data")

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        print(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_1, x_2, edge_index):
        edge_feat_1 = x_1[edge_index[0]]
        edge_feat_2 = x_2[edge_index[1]]
        return (edge_feat_1 * edge_feat_2).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.emb = torch.nn.Embedding(pytorch_graph_before.num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()
    
    def forward(self, data):
        x_dict = {
          "emb": self.emb(data.node_id),
          "lin": self.emb(data.node_id),
        } 
        x_dict = self.gnn(x_dict, data.edge_index)
        pred = self.classifier(
            x_dict["emb"],
            x_dict["lin"],
            pytorch_graph_before.edge_index,
        )
        return pred

model = Model(hidden_channels=64)
print("define model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    optimizer.zero_grad()
    pytorch_graph_before.to(device)
    pred = model(pytorch_graph_before)
    ground_truth = test_graph
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
print("training complete")
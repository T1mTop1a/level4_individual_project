import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import download_url, extract_zip, Data
import torch_geometric.transforms as T
from torch_geometric.loader import LinkLoader
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

print("import complete")

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

movies_path = './ml-latest-small/movies.csv'
ratings_path = './ml-latest-small/ratings.csv'

# Load the entire movie data frame into memory:
movies_df = pd.read_csv(movies_path, index_col='movieId')
print("movie data loaded")

# Split genres and convert into indicator variables:
genres = movies_df['genres'].str.get_dummies('|')
print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
# Use genres as movie input features:
movie_feat = torch.from_numpy(genres.values).to(torch.float)
assert movie_feat.size() == (9742, 20)  # 20 genres in total.

# Load the entire ratings data frame into memory:
ratings_df = pd.read_csv(ratings_path)
print("movie data loaded")

unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})

unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})
print("data mapped")

ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                            left_on='userId', right_on='userId', how='left')
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                            left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)


edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
assert edge_index_user_to_movie.size() == (2, 100836)
print("constructed edge_index")

data = Data()
data.x = movie_feat
data.edge_index = edge_index_user_to_movie
data.node_id = torch.arange(len(unique_user_id) + len(movies_df))

print("prepared graph")

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
)
train_data, val_data, test_data = transform(data)
print("define data split")

'''
train_loader = LinkLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    batch_size=128,
    shuffle=True,
)
print("define batch loader")
'''

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
    def forward(self, x_user, x_movie, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data.num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data.num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
          "user": self.user_emb(data.node_id),
          "movie": self.movie_lin(data.x) + self.movie_emb(data.node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            train_data.edge_label_index,
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
    train_data.to(device)
    pred = model(train_data)
    ground_truth = train_data.edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

print("training complete")

val_loader = LinkLoader(
    data=val_data,
    num_neighbors=[20, 10],
    batch_size=3 * 128,
    shuffle=False,
)

preds = []
ground_truths = []

with torch.no_grad():
    val_data.to(device)
    preds.append(model(val_data))
    ground_truths.append(val_data.edge_label)

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")
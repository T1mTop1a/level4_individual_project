import numpy as np
from scipy.sparse import rand as sprand
import torch

# Make up some random explicit feedback ratings
# and convert to a numpy array
n_users = 1_000
n_items = 1_000
ratings = sprand(n_users, n_items, density=0.01, format="csr")
ratings.data = np.random.randint(1, 5, size=ratings.nnz).astype(np.float64)
ratings = ratings.toarray()

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

model = MatrixFactorization(n_users, n_items, n_factors=20)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# Sort our data
rows, cols = ratings.nonzero()
p = np.random.permutation(len(rows))
rows, cols = rows[p], cols[p]

for row, col in zip(*(rows, cols)):
    # Set gradients to zero
    optimizer.zero_grad()
    
    # Turn data into tensors
    rating = torch.FloatTensor([ratings[row, col]])
    row = torch.LongTensor([row])
    col = torch.LongTensor([col])

    # Predict and calculate loss
    prediction = model(row, col)
    loss = loss_func(prediction, rating)

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()

print("completed training")

test_loss = 0
num_samples = 0
for row, col in zip(*(rows, cols)):
    rating = torch.FloatTensor([ratings[row, col]])
    row = torch.LongTensor([row])
    col = torch.LongTensor([col])
    prediction = model(row, col)
    test_loss += loss_func(prediction, rating).item()
    num_samples += 1

rmse = np.sqrt(test_loss / num_samples)
mae = test_loss / num_samples

print("RMSE:", rmse)
print("MAE:", mae)
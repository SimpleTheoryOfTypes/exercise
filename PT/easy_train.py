import torch
import numpy as np

torch.manual_seed(666)

# hyperparameters
lr = 0.1
num_embeddings = 7
embedding_dim = 5

class EasyEmbedding(torch.nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super(EasyEmbedding, self).__init__()
    self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    self.fc = torch.nn.Linear(embedding_dim, 3, bias=False)
    self.mse_loss = torch.nn.MSELoss(reduction='sum')

  def forward(self, x, y_true):
    x = self.embedding(x)
    y_hat = self.fc(x)
    mse_loss = self.mse_loss(y_hat, y_true)
    return mse_loss

# prepare inputs
indices = [0, 2, 5, 6]
y_true = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]]
pt_model = EasyEmbedding(num_embeddings, embedding_dim)

# run pt version of the model
pt_x = torch.LongTensor(indices)
pt_y_true = torch.tensor(y_true)
loss = pt_model(pt_x, pt_y_true)
print(loss)
print(pt_model.embedding.weight)
np.save("embedding_w.npy", pt_model.embedding.weight.detach().numpy())
print(pt_model.embedding.weight.grad)
print(pt_model.fc.weight)
np.save("fc_w.npy", pt_model.fc.weight.detach().numpy())
print(pt_model.fc.weight.grad)
optimizer = torch.optim.SGD(pt_model.parameters(), lr=lr, momentum=0.0)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("======= PyTorch: After one iteration ======")
print(pt_model.embedding.weight)
print(pt_model.embedding.weight.grad)

## My own numpy implementation.
E = np.load("embedding_w.npy") # reload embedding from saved model.
W = np.load("fc_w.npy") # reload FC's weight from saved model.

X = E[indices]
diff = np.matmul(X, np.transpose(W)) - y_true
print("Loss = ", np.sum(diff * diff))

DLy = 2.0 * diff
Dye = W

Egrad = np.matmul(DLy, Dye)
Enew = E[indices] - lr * Egrad

# Write back the updated embeddings.
count = 0
for idx in indices:
  E[idx] = Enew[count]
  count += 1

print(Egrad)
print(E)

# check loss again.
X = E[indices]
diff = np.transpose(np.matmul(W, np.transpose(X))) - y_true
loss = np.sum(diff * diff)
print(loss)

#======== Testing ======
E_train_by_pt = pt_model.embedding.weight.detach().numpy()
avg_mismatch = np.average(np.abs(E - E_train_by_pt))
print("avg_mismatch = ", avg_mismatch)
assert avg_mismatch < 1e-8, "Your numpy implementation does not match PyTorch"
print("PASS.")

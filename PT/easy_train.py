  loss = pt_model(pt_x, pt_y_true)
  print(loss)
  print(pt_model.embedding.weight)
  print(pt_model.embedding.weight.grad)
  print(pt_model.fc.weight)
  print(pt_model.fc.weight.grad)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

## My own numpy implementation.
E = np.load("embedding_w.npy") # reload embedding from saved model.
W = np.load("fc_w.npy") # reload FC's weight from saved model.

X = E[indices]
diff = np.transpose(np.matmul(W, np.transpose(X))) - y_true
loss = np.sum(diff * diff)
print(loss)

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


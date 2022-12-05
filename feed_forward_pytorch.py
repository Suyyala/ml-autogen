import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the input and output dimensions
input_dim = 2
output_dim = 1

# Define the hidden layers and units
hidden_units = [2]

# Define the activation function
activation = 'relu'

# Define the L2 regularization parameter
l2_reg = 0.1

# Create the feedforward layer
layer = Feedforward(input_dim, output_dim, hidden_units, activation, l2_reg)

# Define the optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

# Define the training data
X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.Tensor([[0], [1], [1], [0]])

# Train the layer
for _ in range(10000):
  # Compute the output of the layer
  y_pred = layer(X)

  # Compute the loss
  loss = F.mse_loss(y_pred, y)

  # Backpropagate the error and update the weights and biases
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# Test the trained layer
print(layer(torch.Tensor([[0, 0]])))  # Expected output: [0]
print(layer(torch.Tensor([[0, 1]])))  # Expected output: [1]
print(layer(torch.Tensor([[1, 0]])))  # Expected output: [1]
print(layer(torch.Tensor([[1, 1]])))  # Expected output: [0]

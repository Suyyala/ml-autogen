# Define a class to represent the feedforward layer
class Feedforward:
  def __init__(self, input_dim, output_dim, hidden_units, activation):
    # Define the weights and biases for the hidden layers
    self.weights = []
    self.biases = []
    prev_units = input_dim
    for units in hidden_units:
      self.weights.append([[random.random() for _ in range(prev_units)] for _ in range(units)])
      self.biases.append([random.random() for _ in range(units)])
      prev_units = units

    # Define the weights and biases for the output layer
    self.weights.append([[random.random() for _ in range(prev_units)] for _ in range(output_dim)])
    self.biases.append([random.random() for _ in range(output_dim)])

    # Define the activation function to use
    if activation == 'relu':
      self.activation = lambda x: max(0.0, x)
    elif activation == 'sigmoid':
      self.activation = lambda x: 1 / (1 + math.exp(-x))
    elif activation == 'tanh':
      self.activation = lambda x: math.tanh(x)

  def forward(self, input_data):
    # Compute the output of the hidden layers
    hidden = []
    for i in range(len(self.weights) - 1):
      h = [0.0 for _ in range(len(self.biases[i]))]
      for j in range(len(self.biases[i])):
        for k in range(len(input_data)):
          h[j] += self.weights[i][j][k] * input_data[k]
        h[j] += self.biases[i][j]
      hidden.append([self.activation(x) for x in h])
      input_data = hidden[-1]

    # Compute the output of the output layer
    output = [0.0 for _ in range(len(self.biases[-1]))]
    for i in range(len(self.biases[-1])):
      for j in range(len(input_data)):
        output[i] += self.weights[-1][i][j] * input_data[j]
      output[i] += self.biases[-1][i]

    # Return the output of the layer
    return output

# Define the model with two hidden layers and a ReLU activation function
model = Feedforward(100, 10, [64, 32], 'relu')

# Compute the output of the model for some input data
output = model.forward([random.random() for _ in range(100)])

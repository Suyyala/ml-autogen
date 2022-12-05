# Import the math package
import math

# Define a class to represent the feedforward layer
class Feedforward:
  def __init__(self, input_dim, output_dim, hidden_units, activation, l2_reg=0.0):
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

    # Define the L2 regularization parameter
    self.l2_reg = l2_reg

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
  
  def backward(self, input_data, target, loss):
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

    # Compute the gradients of the weights and biases
    dweights = []
    dbiases = []
    for i in range(len(self.weights) - 1, -1, -1):
      if i == len(self.weights) - 1:
        doutput = [loss.derivative(output[j], target[j]) for j in range(len(output))]
        dhidden = [[doutput[j] * self.weights[i][j

 def train(self, training_data, epochs, learning_rate, loss):
    # Train the model for the specified number of epochs
    for _ in range(epochs):
      # Shuffle the training data
      random.shuffle(training_data)

      # Iterate over the training data
      for input_data, target in training_data:
        # Compute the output of the network
        output = self.forward(input_data)

        # Compute the loss using the specified loss function
        error = loss.function(output, target)

        # Backpropagate the error and update the weights and biases
        self.backward(input_data, target, loss)
        for i in range(len(self.weights)):
          for j in range(len(self.weights[i])):
            for k in range(len(self.weights[i][j])):
              self.weights[i][j][k] -= learning_rate * self.dweights[i][j][k] + self.l2_reg * self.weights[i][j][k]
          for j in range(len(self.biases[i])):
            self.biases[i][j] -= learning_rate * self.dbiases[i][j] + self.l2_reg * self.biases[i][j]

    # Return the trained model
    return self
                                                 
 
                                                 
                                                 
                                                 # Define the input and output dimensions
input_dim = 2
output_dim = 1

# Define the hidden layers and units
hidden_units = [2]

# Define the activation function
activation = 'relu'

# Define the L2 regularization parameter
l2_reg = 0.1

# Define the training data
training_data = [
  ([0, 0], [0]),
  ([0, 1], [1]),
  ([1, 0], [1]),
  ([1, 1], [0])
]

# Define the number of epochs to train for
epochs = 10000

# Define the learning rate
learning_rate = 0.1

# Define the loss function to use
loss = MeanSquaredError()

# Create the feedforward layer
layer = Feedforward(input_dim, output_dim, hidden_units, activation, l2_reg)

# Train the layer
layer = layer.train(training_data, epochs, learning_rate, loss)

# Test the trained layer
print(layer.forward([0, 0]))  # Expected output: [0]
print(layer.forward([0, 1]))  # Expected output: [1]
print(layer.forward([1, 0]))  # Expected output: [1]
print(layer.forward([1, 1]))  # Expected output: [0]



import numpy as np
import torch

np.random.seed(42)

# Define the dimensions of the neural network
input_size = 3
hidden_size = 4
output_size = 2

# Initialize weights and biases for the neural network
W1_np = np.random.randn(hidden_size, input_size) * 0.01
b1_np = np.zeros((hidden_size, 1))
W2_np = np.random.randn(output_size, hidden_size) * 0.01
b2_np = np.zeros((output_size, 1))

# Define the sigmoid activation function and its derivative
X_np = np.array([
    [0.6],
    [0.3],
    [0.8]
])

Y_np = np.array([
    [1.0],
    [0.0]
])

# Perform forward propagation through the neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Perform forward propagation and print the shapes of the outputs
Z1 = np.dot(W1_np, X_np) + b1_np
A1 = sigmoid(Z1)

Z2 = np.dot(W2_np, A1) + b2_np
A2 = sigmoid(Z2)

dZ2 = (A2 - Y_np) * sigmoid_derivative(Z2)
dW2 = np.dot(dZ2, A1.T)
db2 = dZ2
dA1 = np.dot(W2_np.T, dZ2)

dZ1 = dA1 * sigmoid_derivative(Z1)
dW1 = np.dot(dZ1, X_np.T)
db1 = dZ1

# Convert the NumPy arrays to PyTorch tensors and enable gradient tracking
W1 = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
b1 = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
W2 = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
b2 = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

X = torch.tensor(X_np, dtype=torch.float64)
Y = torch.tensor(Y_np, dtype=torch.float64)

Z1_t = torch.matmul(W1, X) + b1
A1_t = torch.sigmoid(Z1_t)
Z2_t = torch.matmul(W2, A1_t) + b2
A2_t = torch.sigmoid(Z2_t)

loss = 0.5 * torch.sum((A2_t - Y) ** 2)

# Perform backward propagation to compute gradients
loss.backward()

print("Gradient comparison")
print("-------------------")

# Compare the gradients computed using NumPy and PyTorch
print("dW1:", np.allclose(
    dW1,
    W1.grad.numpy(),
    atol=1e-10
))

print("db1:", np.allclose(
    db1,
    b1.grad.numpy(),
    atol=1e-10
))

print("dW2:", np.allclose(
    dW2,
    W2.grad.numpy(),
    atol=1e-10
))

print("db2:", np.allclose(
    db2,
    b2.grad.numpy(),
    atol=1e-10
))


print("\nMaximum absolute difference")

print("dW1:",
      np.max(np.abs(dW1 - W1.grad.numpy())))

print("db1:",
      np.max(np.abs(db1 - b1.grad.numpy())))

print("dW2:",
      np.max(np.abs(dW2 - W2.grad.numpy())))

print("db2:",
      np.max(np.abs(db2 - b2.grad.numpy())))

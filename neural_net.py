import numpy as np

input_size = 3
hidden_size = 4
output_size = 2

W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



def neuron_forward(X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    dZ2 = (A2 - Y) * sigmoid_derivative(Z2)
    dW2 = np.dot(dZ2, A1.T)
    db2 = dZ2
    dA1 = np.dot(W2.T, dZ2)
    return Z1, A1, Z2, A2

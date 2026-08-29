import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

np.random.seed(42)

input_dim = 784
hidden_dim = 64
output_dim = 10

W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
b2 = np.zeros(output_dim)

lr = 0.1
epochs = 200
losses = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def backward(X, Y, a1, a2, W1, W2):
    m = X.shape[0]

    dz2 = (a2 - Y) / m

    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(dz2, W2.T)

    dz1 = da1 * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2


train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)

test_set = datasets.MNIST(
    root="./data",
    train=False,
    download=True
)

X_train = train_set.data.numpy().reshape(-1, 784).astype(np.float64)
y_train = train_set.targets.numpy()

X_test = test_set.data.numpy().reshape(-1, 784).astype(np.float64)
y_test = test_set.targets.numpy()

X_train /= 255.0
X_test /= 255.0

X_small = X_train[:2000]
y_small = y_train[:2000]

Y_small = np.eye(10)[y_small]

print("X_small:", X_small.shape)
print("Y_small:", Y_small.shape)


for epoch in range(epochs):

    z1 = np.dot(X_small, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    loss = cross_entropy(a2, Y_small)
    losses.append(loss)

    dW1, db1, dW2, db2 = backward(
        X_small,
        Y_small,
        a1,
        a2,
        W1,
        W2
    )

    W1 -= lr * dW1
    b1 -= lr * db1

    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss {loss:.4f}")


plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()
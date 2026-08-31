import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

np.random.seed(42)

input_dim = 784
hidden_dim = 64
output_dim = 10

lr = 0.4
epochs = 200

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float64)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    return -np.mean(
        np.sum(targets * np.log(predictions), axis=1)
    )


def mse_loss(predictions, targets):
    return np.mean(np.sum((predictions - targets) ** 2, axis=1))



def backward_sigmoid_ce(X, Y, z1, a1, a2, W2):
    m = X.shape[0]

    dz2 = (a2 - Y) / m

    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(dz2, W2.T)

    dz1 = da1 * sigmoid_derivative(z1)

    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2




def backward_relu_mse(X, Y, z1, a1, a2, W2):
    m = X.shape[0]

    da2 = 2 * (a2 - Y) / m

    sum_term = np.sum(da2 * a2, axis=1, keepdims=True)

    dz2 = a2 * (da2 - sum_term)


    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)


    da1 = np.dot(dz2, W2.T)

    dz1 = da1 * relu_derivative(z1)


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



W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
b2 = np.zeros(output_dim)

losses_original = []

for epoch in range(epochs):

    z1 = np.dot(X_small, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    loss = cross_entropy(a2, Y_small)
    losses_original.append(loss)

    dW1, db1, dW2, db2 = backward_sigmoid_ce(
        X_small,
        Y_small,
        z1,
        a1,
        a2,
        W2
    )

    W1 -= lr * dW1
    b1 -= lr * db1

    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 20 == 0:
        print(f"[Original] epoch {epoch}, loss {loss:.4f}")



W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
b2 = np.zeros(output_dim)

losses_stretch = []

momentum = 0.9

vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)

vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)


for epoch in range(epochs):


    z1 = np.dot(X_small, W1) + b1

    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2

    a2 = softmax(z2)


    loss = mse_loss(a2, Y_small)
    losses_stretch.append(loss)

    dW1, db1, dW2, db2 = backward_relu_mse(
        X_small,
        Y_small,
        z1,
        a1,
        a2,
        W2
    )

    vW1 = momentum * vW1 + dW1
    vb1 = momentum * vb1 + db1

    vW2 = momentum * vW2 + dW2
    vb2 = momentum * vb2 + db2

    W1 -= lr * vW1
    b1 -= lr * vb1

    W2 -= lr * vW2
    b2 -= lr * vb2

    if epoch % 20 == 0:
        print(f"[Stretch] epoch {epoch}, loss {loss:.4f}")


losses_original_norm = np.array(losses_original) / losses_original[0]
losses_stretch_norm = np.array(losses_stretch) / losses_stretch[0]

plt.plot(losses_original_norm, label="Sigmoid + Cross Entropy")
plt.plot(losses_stretch_norm, label="ReLU + MSE + Momentum")
plt.xlabel("Epoch")
plt.ylabel("Loss (relative to initial)")
plt.title("Relative Training Progress")
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison_final.png")
plt.show()
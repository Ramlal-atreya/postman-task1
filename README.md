# Postman Task 1 
# Neural Network with Manual Backpropagation

A feedforward neural network built from scratch in NumPy — forward pass, backward
pass, and gradients all derived and implemented manually, without autograd or
`.backward()`. Built for the Postman AI/ML recruitment task (Task 1).

## What's implemented

- A small 3-4-2 toy network used to work out the backprop derivation
  (`neural_net.py`)
- A correctness harness that checks manual gradients against `torch.autograd`
  on that same toy network (`test_gradients.py`)
- A 784-64-10 network trained on a reduced MNIST subset, using sigmoid
  activation and cross-entropy loss with plain SGD (`train.py`)
- A stretch-goal comparison: the same baseline vs. a ReLU + MSE + Momentum
  variant, trained side by side (`stretch.py`)

## Setup

```bash
git clone https://github.com/Ramlal-atreya/postman-task1
cd postman-task1
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy matplotlib torch torchvision
```

## How to run

**Check the gradient derivation is correct:**
```bash
python test_gradients.py
```
Prints `True`/`False` per gradient (dW1, db1, dW2, db2) comparing manual
gradients against PyTorch's, plus the maximum absolute difference for each.

**Train the baseline network on MNIST:**
```bash
python train.py
```
Downloads MNIST automatically on first run (cached in `./data`, gitignored),
trains for 200 epochs, prints loss every 20 epochs, and saves `loss_curve.png`.

**Run the stretch-goal comparison:**
```bash
python stretch.py
```
Trains both the sigmoid+CE baseline and the ReLU+MSE+momentum variant, and
saves a normalized loss comparison plot.

## Project structure
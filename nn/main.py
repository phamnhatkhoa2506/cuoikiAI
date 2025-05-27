import numpy as np
from numpy import genfromtxt
from layers import Linear
from activations import Sigmoid
from criterias import MSELoss, CrossEntropyLoss
from sequetial import Sequential
from train import train

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Fake dataset
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # binary classification

# Build model
model = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 2),
    # Sigmoid()
)

# Loss and training
loss_fn = CrossEntropyLoss()
train(model, loss_fn, X, y, epochs=100, lr=0.5)

logits = model.forward([[1.0, 0.5]])
probs = softmax(logits)
print("Probabilities:", probs)
print("Predicted class:", np.argmax(probs, axis=1)) 

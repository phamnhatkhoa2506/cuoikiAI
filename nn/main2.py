import numpy as np
from numpy import genfromtxt
from layers import Linear
from activations import Sigmoid
from criterias import MSELoss
from sequetial import Sequential
from train import train
from normalize import Normalizer

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Fake dataset
data = genfromtxt('advertising.csv', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1:]  # Features and labels

norm_x = Normalizer()
X = norm_x.fit_transform(X)

norm_y = Normalizer()
y = norm_y.fit_transform(y)

# Build model
model = Sequential(
    Linear(3, 4),
    # Sigmoid(),
    Linear(4, 1),
    # Sigmoid()
)

# Loss and training
loss_fn = MSELoss()
train(model, loss_fn, X, y, epochs=100, lr=0.5)

# Make prediction with 3 features to match the input layer
logits = model.forward([[1.0, 0.5, 0.3]])  # Added third feature
logits = norm_y.inverse_transform(logits)
print("Result:", logits)

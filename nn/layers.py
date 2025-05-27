import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros((out_features, 1))
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.input = x  # (batch_size, in_features)
        return np.dot(x, self.weight.T) + self.bias.T

    def backward(self, grad_output):
        # grad_output: (batch_size, out_features)
        self.grad_weight = np.dot(grad_output.T, self.input)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True).T
        return np.dot(grad_output, self.weight)

    def step(self, lr):
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias



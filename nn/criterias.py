import numpy as np

class MSELoss:
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.target.size

class CrossEntropyLoss:
    def forward(self, logits, target):
        self.prediction = self._softmax(logits)
        self.target = target
        m = target.shape[0]
        log_likelihood = -np.log(self.prediction[np.arange(m), target])
        return np.mean(log_likelihood)

    def backward(self):
        m = self.target.shape[0]
        if self.target.ndim == 1:
            grad = self.prediction
            grad[np.arange(m), self.target] -= 1
            return grad / m
        else:
            return (self.prediction - self.target) / m

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

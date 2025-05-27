import numpy as np

import numpy as np

class CrossEntropyLoss:
    def __init__(self, weight: np.ndarray = None, size_average: bool = True):
        """
            weight: Optional 1D array of class weights
            size_average: If True, return mean loss over batch; else return sum
        """

        self.weight = weight
        self.size_average = size_average

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Parameters:
            y_pred: np.ndarray of shape (batch_size, num_classes) — predicted probabilities (after softmax)
            y_true: np.ndarray of shape (batch_size, num_classes) — one-hot encoded ground truth
        Returns:
            Cross-entropy loss (float)
        """

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        log_probs = -np.sum(y_true * np.log(y_pred), axis=1)

        if self.weight is not None:
            class_weights = np.dot(y_true, self.weight)
            log_probs *= class_weights

        loss = log_probs.mean() if self.size_average else log_probs.sum()
        return loss


class MSELoss(object):
    def __init__(self, size_average: bool = True):
        self.size_average = size_average

    def compute_loss(self, output: np.ndarray, target: np.ndarray):
        """
            Compute the gradient of the loss with respect to the output.

            Parameters:
                output: The model's output.
                target: The ground truth labels.

            Return: The gradient of the loss.
        """

        loss = (output - target) ** 2
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        

class L1Loss(object):
    def __init__(self, size_average: bool = True):
        self.size_average = size_average

    def compute_loss(self, output: np.ndarray, target: np.ndarray):
        """
            Compute the gradient of the loss with respect to the output.

            Parameters:
                output: The model's output.
                target: The ground truth labels.

            Return: The gradient of the loss.
        """

        loss = np.abs(output - target)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        

class LogSoftmaxLoss(object):
    def __init__(self, weight: np.ndarray = None, size_average: bool = True):
        self.weight = weight
        self.size_average = size_average

    def compute_loss(self, output: np.ndarray, target: np.ndarray):
        """
            Compute the gradient of the loss with respect to the output.
            Parameters:
                output: The model's output (logits).
                target: The ground truth labels.
            
            Return: The gradient of the loss.
        """

        log_probs = self.log_softmax(output)
        loss = - np.sum(target * log_probs, axis=1)  # shape (batch,)

        if self.weight is not None:
            class_weights = np.dot(target, self.weight)
            loss *= class_weights

        return loss.mean() if self.size_average else loss.sum()
        
    def log_softmax(self, x: np.ndarray):
        """
            Compute the log softmax of the input.
            Parameters:
                x: The input array.
            Return: The log softmax of the input.
        """

        max_x = np.max(x, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True)) + max_x
        return x - logsumexp
        
if __name__ == "__main__":
    # Example usage
    output = np.array([[0.1, 0.9], [0.8, 0.2]])
    target = np.array([[0, 1], [1, 0]])
    
    ce_loss = CrossEntropyLoss()
    print("CrossEntropyLoss:", ce_loss.compute_loss(output, target))
    
    mse_loss = MSELoss()
    print("MSELoss:", mse_loss.compute_loss(output, target))
    
    l1_loss = L1Loss()
    print("L1Loss:", l1_loss.compute_loss(output, target))
    
    log_softmax_loss = LogSoftmaxLoss()
    print("LogSoftmaxLoss:", log_softmax_loss.compute_loss(output, target))
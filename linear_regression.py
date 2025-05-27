import numpy as np
import pandas as pd

class MSELoss:
    def __init__(self, size_average=True):
        self.size_average = size_average

    def forward(self, y_pred, y_true):
        loss = (y_pred - y_true) ** 2
        return loss.mean() if self.size_average else loss.sum()

    def backward(self, y_pred, y_true):
        grad = 2 * (y_pred - y_true)
        if self.size_average:
            grad /= y_pred.shape[0]
        return grad


class LinearRegression:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def backward(self, X, grad_output, learning_rate):
        batch_size = X.shape[0]
        dW = np.dot(X.T, grad_output) / batch_size
        db = np.mean(grad_output)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db


if __name__ == "__main__":
    df = pd.read_csv("Countries-exercise (1).csv")
    df.drop(columns=["name"], inplace=True)
    df_numpy = df.values

    X, y = df_numpy[:, :-1], df_numpy[:, -1].reshape(-1, 1)

    # Chuẩn hóa X và y
    mean_X, std_X = X.mean(axis=0), X.std(axis=0)
    std_X[std_X == 0] = 1  # tránh chia cho 0
    X = (X - mean_X) / std_X

    y = (y - y.mean()) / y.std()

    # Khởi tạo mô hình và loss
    model = LinearRegression(input_dim=X.shape[1])
    loss_fn = MSELoss()

    # Huấn luyện
    epochs = 1000
    learning_rate = 0.2  # nhỏ hơn

    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, y)
        grad = loss_fn.backward(y_pred, y)
        model.backward(X, grad, learning_rate)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

    print("Final Weights:", model.W.flatten())
    print("Final Bias:", model.b)

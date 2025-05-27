import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.n_classes = None
        self.n_features = None
        self.loss_history = []
        
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    def fit(self, X, y):
        n_samples, self.n_features = X.shape
        self.n_classes = y.shape[1]
        
        # Initialize weights
        self.W = np.random.randn(self.n_features, self.n_classes) * 0.01
        self.b = np.zeros((1, self.n_classes))
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass
            logits = X @ self.W + self.b
            probs = self.softmax(logits)
            
            # Compute loss
            loss = self.cross_entropy(y, probs)
            self.loss_history.append(loss)
            
            # Backward pass
            grad_logits = probs - y
            grad_W = X.T @ grad_logits / n_samples
            grad_b = np.sum(grad_logits, axis=0, keepdims=True) / n_samples
            
            # Update parameters
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def score(self, X, y):
        """Compute accuracy score on test data"""
        probs = self.predict_proba(X)
        return self.accuracy(y, probs)
    
    def plot_loss_history(self):
        """Plot the training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    # Load data from input.csv
    data = np.genfromtxt('input.csv', delimiter=',')
    X = data[:, :-1]  # Features
    labels = np.genfromtxt('input.csv', delimiter=',', dtype=str)[:, -1]  # String labels
    
    # Convert string labels to numeric
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in labels])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode y
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, epochs=500)
    model.fit(X_train, y_train)
    
    # Plot training loss
    model.plot_loss_history()
    
    # Evaluate
    test_accuracy = model.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Plot confusion matrix
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_true, y_pred, unique_labels)
    
    # Make prediction for a few samples
    print("\nSample Predictions:")
    for i in range(3):
        sample = np.array([X_test[i]])
        probs = model.predict_proba(sample)
        predicted_class = model.predict(sample)
        true_class = np.argmax(y_test[i])
        
        print(f"\nSample {i+1}:")
        print(f"True class: {unique_labels[true_class]}")
        print(f"Predicted class: {unique_labels[predicted_class[0]]}")
        print(f"Class probabilities: {probs[0]}")

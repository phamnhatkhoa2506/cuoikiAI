import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


class NeuralNetwork:
    def __init__(self, 
                 input_size: int,
                 num_classes: int,
                 learning_rate: float = 0.01,
                 num_epochs: int = 1000,
                 batch_size: int = 32,
                 random_state: Optional[int] = None) -> None:
        """
        Initialize Neural Network with logistic regression and softmax
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            learning_rate: Learning rate for gradient descent
            num_epochs: Number of training epochs
            batch_size: Size of mini-batches
            random_state: Random seed for reproducibility
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each set of scores in z
        
        Args:
            z: Input scores
            
        Returns:
            Softmax probabilities
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network
        
        Args:
            X: Input features
            
        Returns:
            tuple: (logits, probabilities)
        """
        # Compute logits
        logits = np.dot(X, self.weights) + self.bias
        
        # Compute probabilities using softmax
        probs = self.softmax(logits)
        
        return logits, probs

    def compute_loss(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_true: True labels (one-hot encoded)
            probs: Predicted probabilities
            
        Returns:
            Cross-entropy loss
        """
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(probs + 1e-15)) / m
        return loss

    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias
        
        Args:
            X: Input features
            y_true: True labels (one-hot encoded)
            probs: Predicted probabilities
            
        Returns:
            tuple: (weight_gradients, bias_gradients)
        """
        m = X.shape[0]
        
        # Compute gradients
        dw = np.dot(X.T, (probs - y_true)) / m
        db = np.sum(probs - y_true, axis=0, keepdims=True) / m
        
        return dw, db

    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels to one-hot encoding
        
        Args:
            y: Labels
            
        Returns:
            One-hot encoded labels
        """
        m = y.shape[0]
        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        return y_one_hot

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the neural network
        
        Args:
            X: Training features
            y: Training labels
        """
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Convert labels to one-hot encoding
        y_one_hot = self.one_hot_encode(y)
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            # Mini-batch training
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Forward pass
                _, probs = self.forward(X_batch)
                
                # Compute gradients
                dw, db = self.compute_gradients(X_batch, y_batch, probs)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss and accuracy for the epoch
            _, probs = self.forward(X)
            loss = self.compute_loss(y_one_hot, probs)
            predictions = np.argmax(probs, axis=1)
            accuracy = accuracy_score(y, predictions)
            
            # Store history
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        # Scale features
        X = self.scaler.transform(X)
        
        # Forward pass
        _, probs = self.forward(X)
        
        # Return predicted class
        return np.argmax(probs, axis=1)

    def plot_training_history(self) -> None:
        """Plot training loss and accuracy history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot confusion matrix for predictions
        
        Args:
            X: Input features
            y: True labels
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        tuple: (X, y, unique_labels)
    """
    # Load data
    data = np.genfromtxt(file_path, delimiter=',')
    X = data[:, :-1]
    
    # Convert string labels to numeric
    labels = np.genfromtxt(file_path, delimiter=',', dtype=str)[:, -1]
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in labels])
    
    return X, y, unique_labels


if __name__ == '__main__':
    # Load and preprocess data
    X, y, unique_labels = load_and_preprocess_data('input.csv')
    
    # Create and train model
    model = NeuralNetwork(
        input_size=X.shape[1],
        num_classes=len(unique_labels),
        learning_rate=0.01,
        num_epochs=1000,
        batch_size=32,
        random_state=42
    )
    
    # Train model
    model.fit(X, y)
    
    # Plot training history
    model.plot_training_history()
    
    # Plot confusion matrix
    model.plot_confusion_matrix(X, y)
    
    # Load and predict new data
    X_new = np.genfromtxt('output.csv', delimiter=',')
    predictions = model.predict(X_new)
    
    # Save predictions to file
    output_file = 'neural_network_predictions.txt'
    with open(output_file, 'w') as f:
        f.write("Neural Network Predictions for New Data\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Index\tFeatures\t\t\tPredicted Class\n")
        f.write("-" * 80 + "\n")
        
        for i, (features, pred) in enumerate(zip(X_new, predictions)):
            f.write(f"{i}\t{features}\t{unique_labels[pred]}\n")
    
    print(f"\nPredictions saved to {output_file}") 
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


class Activation:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu') -> None:
        """
        Initialize a neural network layer
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            activation: Activation function ('relu' or 'softmax')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        
        # Initialize cache for backpropagation
        self.cache: Dict[str, np.ndarray] = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        
        Args:
            X: Input features
            
        Returns:
            Activated output
        """
        # Linear transformation
        z = np.dot(X, self.weights) + self.bias
        
        # Store cache for backpropagation
        self.cache['X'] = X
        self.cache['z'] = z
        
        # Apply activation
        if self.activation == 'relu':
            a = Activation.relu(z)
        elif self.activation == 'softmax':
            a = Activation.softmax(z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        
        return a
    
    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer
        
        Args:
            dA: Gradient of the loss with respect to the layer output
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Gradient of the loss with respect to the layer input
        """
        X = self.cache['X']
        z = self.cache['z']
        m = X.shape[0]
        
        # Compute gradient of activation
        if self.activation == 'relu':
            dZ = dA * Activation.relu_derivative(z)
        elif self.activation == 'softmax':
            dZ = dA  # For softmax, dA is already dZ
        
        # Compute gradients
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return dX


class MultiLayerNeuralNetwork:
    def __init__(self, 
                 layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 num_epochs: int = 1000,
                 batch_size: int = 32,
                 random_state: Optional[int] = None) -> None:
        """
        Initialize Multi-Layer Neural Network
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden_size1, ..., output_size]
            learning_rate: Learning rate for gradient descent
            num_epochs: Number of training epochs
            batch_size: Size of mini-batches
            random_state: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize layers
        self.layers: List[Layer] = []
        for i in range(len(layer_sizes) - 1):
            # Use softmax activation for the last layer
            activation = 'softmax' if i == len(layer_sizes) - 2 else 'relu'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            X: Input features
            
        Returns:
            Network output
        """
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Cross-entropy loss
        """
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
        return loss

    def one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert labels to one-hot encoding
        
        Args:
            y: Labels
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels
        """
        m = y.shape[0]
        y_one_hot = np.zeros((m, num_classes))
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
        y_one_hot = self.one_hot_encode(y, self.layer_sizes[-1])
        
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
                y_pred = self.forward(X_batch)
                
                # Backward pass
                dA = y_pred - y_batch  # Initial gradient for softmax
                for layer in reversed(self.layers):
                    dA = layer.backward(dA, self.learning_rate)
            
            # Compute loss and accuracy for the epoch
            y_pred = self.forward(X)
            loss = self.compute_loss(y_one_hot, y_pred)
            predictions = np.argmax(y_pred, axis=1)
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
        y_pred = self.forward(X)
        
        # Return predicted class
        return np.argmax(y_pred, axis=1)

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
    model = MultiLayerNeuralNetwork(
        layer_sizes=[X.shape[1], 64, 32, len(unique_labels)],  # Input -> Hidden1 -> Hidden2 -> Output
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
    output_file = 'neural_network3_predictions.txt'
    with open(output_file, 'w') as f:
        f.write("Multi-Layer Neural Network Predictions for New Data\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Index\tFeatures\t\t\tPredicted Class\n")
        f.write("-" * 80 + "\n")
        
        for i, (features, pred) in enumerate(zip(X_new, predictions)):
            f.write(f"{i}\t{features}\t{unique_labels[pred]}\n")
    
    print(f"\nPredictions saved to {output_file}") 
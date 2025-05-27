import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize neural network with layer sizes
        layer_sizes: List of integers representing the number of neurons in each layer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Initialize weights with random values between -1 and 1
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            # Initialize biases with zeros
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation
        Returns activations and weighted inputs for each layer
        """
        activations = [X]  # First activation is the input
        weighted_inputs = []
        
        for i in range(self.num_layers - 1):
            # Calculate weighted input
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_inputs.append(z)
            # Use sigmoid for hidden layers, linear for output
            if i < self.num_layers - 2:
                a = self.sigmoid(z)
            else:
                a = z  # Linear activation for output layer
            activations.append(a)
        
        return activations, weighted_inputs
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                weighted_inputs: List[np.ndarray], learning_rate: float) -> None:
        """
        Backward propagation
        Updates weights and biases using gradient descent
        """
        m = X.shape[0]  # Number of training examples
        delta = activations[-1] - y  # Error in output layer
        
        for i in range(self.num_layers - 2, -1, -1):
            # Calculate gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                # Calculate error for previous layer
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(weighted_inputs[i-1])
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, 
              batch_size: int = 32) -> List[float]:
        """
        Train the neural network
        Returns list of training losses
        """
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            batches = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                activations, weighted_inputs = self.forward(X_batch)
                
                # Calculate loss
                loss = np.mean(np.square(activations[-1] - y_batch))
                total_loss += loss
                batches += 1
                
                # Backward pass
                self.backward(X_batch, y_batch, activations, weighted_inputs, learning_rate)
            
            avg_loss = total_loss / batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        activations, _ = self.forward(X)
        return activations[-1]


class DataProcessor:
    def __init__(self, csv_path: str, target_column: str):
        self.csv_path = csv_path
        self.target_column = target_column
    
    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data"""
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Select only longitude as input feature and latitude as target
        X = df[['Longitude']].values.astype(float)
        y = df[self.target_column].values.reshape(-1, 1).astype(float)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test


def plot_training_progress(losses: List[float]) -> None:
    """Plot training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def main():
    # Use the Countries-exercise dataset
    csv_path = "Countries-exercise (1).csv"
    target_column = "Latitude"  # We'll predict latitude from longitude
    
    # Initialize data processor
    processor = DataProcessor(csv_path, target_column)
    X_train, y_train, X_test, y_test = processor.load_and_preprocess()
    
    # Create neural network
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 32, 16, 1]  # Input layer, 2 hidden layers, output layer
    nn = NeuralNetwork(layer_sizes)
    
    # Train model
    print("Training neural network to predict latitude from longitude...")
    losses = nn.train(X_train, y_train, epochs=200, learning_rate=0.01)
    
    # Plot training progress
    plot_training_progress(losses)
    
    # Evaluate model
    predictions = nn.predict(X_test)
    test_loss = np.mean(np.square(predictions - y_test))
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        print(f"Actual latitude: {y_test[i][0]:.2f}, Predicted latitude: {predictions[i][0]:.2f}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable
from itertools import permutations


class DistanceMetrics:
    @staticmethod
    def euclidean(X1: np.ndarray, X2: np.ndarray) -> np.float64:
        return np.linalg.norm(X1 - X2)
    
    @staticmethod
    def manhattan(X1: np.ndarray, X2: np.ndarray) -> np.float64:
        return np.sum(np.abs(X1 - X2))
    
    @staticmethod
    def cosine(X1: np.ndarray, X2: np.ndarray) -> np.float64:
        dot_product = np.dot(X1, X2)
        norm_X1 = np.linalg.norm(X1)
        norm_X2 = np.linalg.norm(X2)
        return 1 - (dot_product / (norm_X1 * norm_X2))


class KMeans(object):
    def __init__(self, 
                 K: int, 
                 num_iters: int = 10, 
                 random_state: Optional[int] = None,
                 distance_metric: str = 'euclidean') -> None:
        """
        Initialize KMeans clustering
        
        Args:
            K: Number of clusters
            num_iters: Number of iterations
            random_state: Random seed for reproducibility
            distance_metric: Distance metric to use ('euclidean', 'manhattan', or 'cosine')
        """
        self.__K = K
        self.__num_iters = num_iters
        self.__centroid_history = []
        self.__centroids = []
        self.__indices = []
        self.__distance = 0.0
        
        # Set distance metric
        self.__distance_metric = self.__get_distance_metric(distance_metric)
        
        if random_state:
            np.random.seed(random_state)

    def __get_distance_metric(self, metric_name: str) -> Callable:
        """Get the distance metric function based on name"""
        metrics = {
            'euclidean': DistanceMetrics.euclidean,
            'manhattan': DistanceMetrics.manhattan,
            'cosine': DistanceMetrics.cosine
        }
        if metric_name not in metrics:
            raise ValueError(f"Distance metric '{metric_name}' not supported. "
                           f"Choose from: {list(metrics.keys())}")
        return metrics[metric_name]

    @property
    def centroids(self) -> np.ndarray:
        return self.__centroids

    @property
    def indices(self) -> np.ndarray:
        return self.__indices

    @property
    def centroid_history(self) -> np.ndarray:
        return self.__centroid_history

    @property
    def distance(self) -> np.float64:
        return self.__distance

    def fit(self, X: np.ndarray) -> None:
        self.__fit(X)

    def __init_centroids(self, X: np.ndarray) -> np.ndarray:
        return X[np.random.choice(X.shape[0], self.__K), :]

    def __find_closest_centroids(self, X: np.ndarray) -> np.ndarray:
        indices = np.zeros(X.shape[0], dtype=np.uint8)
        for i in range(X.shape[0]):
            distances = [self.__distance_metric(X[i], centroid) for centroid in self.__centroids]
            indices[i] = np.argmin(distances)
        return indices

    def __cluster(self, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.__K, X.shape[1]))
        for i in range(self.__K):
            cluster_points = X[indices == i]
            if cluster_points.shape[0] == 0:
                centroids[i] = X[np.random.choice(X.shape[0])]
            else:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids

    def __fit(self, X: np.ndarray) -> None:
        self.__centroids = self.__init_centroids(X)
        self.__centroid_history.append(self.__centroids.copy())
        for _ in range(self.__num_iters):
            self.__indices = self.__find_closest_centroids(X)
            self.__centroids = self.__cluster(X, self.__indices)
            self.__centroid_history.append(self.__centroids.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__find_closest_centroids(X)


def find_optimal_k(X: np.ndarray, distance_metric: str = 'euclidean') -> int:
    """
    Find optimal number of clusters using elbow method
    
    Args:
        X: Input data
        distance_metric: Distance metric to use
    """
    distances = []
    for i in range(10):
        kmn = KMeans(i + 1, distance_metric=distance_metric)
        kmn.fit(X)
        centroids = kmn.centroids

        distance = 0
        indices = kmn.indices
        for i, centroid in enumerate(centroids):
            distance += np.sum(np.linalg.norm(X[indices == i] - centroid, axis=1))

        distances.append(distance)

    plt.figure()
    plt.plot(range(1, 11), distances, marker="o")
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Total Distance')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()

    return np.argmin(distances) + 1


def plot_clusters(kmn: KMeans, X: np.ndarray):
    """Plot the clusters and centroids"""
    plt.figure()
    indices = kmn.indices
    centroids = kmn.centroids

    for i in range(kmn.centroids.shape[0]):
        plt.scatter(X[indices == i, 0], X[indices == i, 1])

    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='*', s=150)
    plt.show()


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute accuracy between K-means predictions and actual labels.
    Returns both the accuracy and the best permutation mapping.
    """
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    
    best_accuracy = 0
    best_perm = None
    for perm in permutations(range(K)):
        # Create mapping from cluster to label
        mapping = {i: perm[i] for i in range(K)}
        # Map predictions to labels
        mapped_predictions = np.array([mapping[p] for p in predictions])
        # Compute accuracy
        accuracy = np.mean(mapped_predictions == labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_perm = perm
    
    return best_accuracy, best_perm


def load_and_preprocess_data(file_path: str) -> tuple:
    """
    Load and preprocess data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        tuple: (X, y, labels, unique_labels)
    """
    # Load data
    data = np.genfromtxt(file_path, delimiter=',')
    X = data[:, :-1]
    
    # Convert string labels to numeric
    labels = np.genfromtxt(file_path, delimiter=',', dtype=str)[:, -1]
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in labels])
    
    return X, y, labels, unique_labels


def run_kmeans_experiment(X: np.ndarray, y: np.ndarray, labels: np.ndarray, 
                         unique_labels: np.ndarray, metric: str, K: int = 3, 
                         num_iters: int = 100) -> tuple:
    """
    Run KMeans clustering with specified parameters
    
    Args:
        X: Feature matrix
        y: Numeric labels
        labels: Original string labels
        unique_labels: Unique class names
        metric: Distance metric to use
        K: Number of clusters
        num_iters: Number of iterations
        
    Returns:
        tuple: (model, predictions, accuracy, best_perm)
    """
    print(f"\nUsing {metric} distance:")
    
    # Create and fit model
    kmn = KMeans(K, num_iters, distance_metric=metric)
    kmn.fit(X)
    predictions = kmn.predict(X)
    
    # Print centroids
    centroids = kmn.centroids
    for i, centroid in enumerate(centroids):
        print(f'Cluster {i}: {centroid}')
    
    # Compute accuracy and get best permutation
    accuracy, best_perm = compute_accuracy(predictions, y)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print best permutation mapping
    print("\nBest Cluster to Class Mapping:")
    for cluster, class_idx in enumerate(best_perm):
        print(f"Cluster {cluster} -> {unique_labels[class_idx]}")
    
    return kmn, predictions, accuracy, best_perm


def save_results_to_file(metric: str, kmn: KMeans, predictions: np.ndarray, 
                        labels: np.ndarray, unique_labels: np.ndarray, 
                        accuracy: float, best_perm: tuple) -> None:
    """
    Save clustering results to a file
    
    Args:
        metric: Distance metric used
        kmn: Trained KMeans model
        predictions: Cluster predictions
        labels: Original labels
        unique_labels: Unique class names
        accuracy: Model accuracy
        best_perm: Best permutation mapping
    """
    output_file = f'kmeans_predictions_{metric}.txt'
    with open(output_file, 'w') as f:
        f.write(f"KMeans Clustering Results using {metric} distance\n")
        f.write("=" * 50 + "\n\n")
        
        # Write centroids
        f.write("Cluster Centroids:\n")
        for i, centroid in enumerate(kmn.centroids):
            f.write(f'Cluster {i}: {centroid}\n')
        f.write("\n")
        
        # Write accuracy and best mapping
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        f.write("Best Cluster to Class Mapping:\n")
        for cluster, class_idx in enumerate(best_perm):
            f.write(f"Cluster {cluster} -> {unique_labels[class_idx]}\n")
        f.write("\n")
        
        # Write detailed results
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        f.write("Index\tOriginal Label\tPredicted Cluster\tMapped Class\n")
        f.write("-" * 50 + "\n")
        
        # Create mapping dictionary
        mapping = {i: unique_labels[best_perm[i]] for i in range(len(unique_labels))}
        
        for i, (true_label, pred_cluster) in enumerate(zip(labels, predictions)):
            mapped_class = mapping[pred_cluster]
            f.write(f"{i}\t{true_label}\t\tCluster {pred_cluster}\t\t{mapped_class}\n")
        
        # Write cluster statistics
        f.write("\nCluster Statistics:\n")
        f.write("-" * 50 + "\n")
        for cluster in range(len(unique_labels)):
            cluster_indices = predictions == cluster
            cluster_labels = labels[cluster_indices]
            unique, counts = np.unique(cluster_labels, return_counts=True)
            
            f.write(f"\nCluster {cluster} (mapped to {mapping[cluster]}):\n")
            for label, count in zip(unique, counts):
                f.write(f"  {label}: {count} samples\n")
    
    print(f"\nResults saved to {output_file}")


def predict_new_data(model: KMeans, X_new: np.ndarray, best_perm: tuple, 
                    unique_labels: np.ndarray) -> tuple:
    """
    Predict clusters for new data using the best permutation mapping
    
    Args:
        model: Trained KMeans model
        X_new: New data to predict
        best_perm: Best permutation mapping from training
        unique_labels: Unique class names
        
    Returns:
        tuple: (cluster_predictions, mapped_classes)
    """
    # Get cluster predictions
    cluster_predictions = model.predict(X_new)
    
    # Create mapping dictionary
    mapping = {i: unique_labels[best_perm[i]] for i in range(len(unique_labels))}
    
    # Map clusters to classes
    mapped_classes = np.array([mapping[pred] for pred in cluster_predictions])
    
    return cluster_predictions, mapped_classes


def save_predictions_to_file(X_new: np.ndarray, cluster_predictions: np.ndarray, 
                           mapped_classes: np.ndarray, metric: str) -> None:
    """
    Save predictions for new data to a file
    
    Args:
        X_new: New data
        cluster_predictions: Predicted clusters
        mapped_classes: Mapped class names
        metric: Distance metric used
    """
    output_file = f'kmeans_predictions_new_data_{metric}.txt'
    with open(output_file, 'w') as f:
        f.write(f"KMeans Predictions for New Data using {metric} distance\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Index\tFeatures\t\t\tPredicted Cluster\tMapped Class\n")
        f.write("-" * 80 + "\n")
        
        for i, (features, cluster, mapped_class) in enumerate(zip(X_new, cluster_predictions, mapped_classes)):
            f.write(f"{i}\t{features}\tCluster {cluster}\t\t{mapped_class}\n")
    
    print(f"\nNew data predictions saved to {output_file}")


def run_experiments(X: np.ndarray, y: np.ndarray, labels: np.ndarray, 
                   unique_labels: np.ndarray, X_new: np.ndarray = None) -> None:
    """
    Run KMeans experiments with different distance metrics
    
    Args:
        X: Feature matrix
        y: Numeric labels
        labels: Original string labels
        unique_labels: Unique class names
        X_new: New data to predict (optional)
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    K = 3
    num_iters = 100
    
    for metric in metrics:
        # Run experiment
        kmn, predictions, accuracy, best_perm = run_kmeans_experiment(
            X, y, labels, unique_labels, metric, K, num_iters
        )
        
        # Save results
        save_results_to_file(metric, kmn, predictions, labels, 
                           unique_labels, accuracy, best_perm)
        
        # Plot clusters
        plot_clusters(kmn, X)
        
        # Predict new data if provided
        if X_new is not None:
            cluster_predictions, mapped_classes = predict_new_data(
                kmn, X_new, best_perm, unique_labels
            )
            save_predictions_to_file(X_new, cluster_predictions, mapped_classes, metric)
    
    # Find optimal k
    find_optimal_k(X)


if __name__ == '__main__':
    # Load and preprocess data
    X, y, labels, unique_labels = load_and_preprocess_data('input.csv')
    
    # Load new data for prediction
    X_new = np.genfromtxt('output.csv', delimiter=',')
    
    # Run experiments and predict new data
    run_experiments(X, y, labels, unique_labels, X_new)
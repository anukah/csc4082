import numpy as np
from sklearn.cluster import KMeans


def load_iris_data(filepath):
    """Load iris data from text file."""
    features = []
    labels = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 5:
                    features.append([float(parts[i]) for i in range(4)])
                    labels.append(parts[4])
    
    X = np.array(features)
    species_names = sorted(list(set(labels)))
    y = np.array([species_names.index(label) for label in labels])
    
    return X, y, species_names


def run_kmeans(X, k=3, init='k-means++', random_state=42, max_iter=100):
    """Run sklearn KMeans clustering."""
    kmeans = KMeans(n_clusters=k, init=init, random_state=random_state, 
                    max_iter=max_iter, n_init=1)
    kmeans.fit(X)
    return kmeans


def evaluate_clustering(true_labels, cluster_labels, n_clusters=3):
    """Evaluate clustering accuracy using best mapping."""
    from itertools import permutations
    
    n_classes = len(np.unique(true_labels))
    confusion = np.zeros((n_clusters, n_classes), dtype=int)
    
    for i in range(len(true_labels)):
        confusion[cluster_labels[i], true_labels[i]] += 1
    
    best_accuracy = 0
    best_mapping = None
    
    for perm in permutations(range(n_classes)):
        correct = sum(confusion[i, perm[i]] for i in range(min(n_clusters, len(perm))))
        accuracy = correct / len(true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = {i: perm[i] for i in range(min(n_clusters, len(perm)))}
    
    return best_accuracy, confusion, best_mapping

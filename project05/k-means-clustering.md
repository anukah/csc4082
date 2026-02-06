# K-Means Clustering: Algorithm and Project Report

## 1. Introduction to K-Means Clustering
[cite_start]K-Means is one of the most popular and simple clustering algorithms, originally proposed over 50 years ago[cite: 23]. [cite_start]It is a form of **unsupervised learning**, meaning it organizes data into groups without having any prior labels or category definitions[cite: 20, 58].

[cite_start]The goal of K-Means is to partition $n$ objects into $K$ groups (clusters) such that objects within the same group are highly similar, while objects in different groups are distinct[cite: 61].

## 2. How K-Means Works
The algorithm operates using the concept of **centroids** (cluster centers). [cite_start]A centroid is the arithmetic mean of all the data points currently assigned to a cluster[cite: 185].

The process is iterative. It starts with a random guess of where the centers are, and then constantly refines their positions. [cite_start]It stops when the clusters "stabilize," meaning no data point wants to switch groups anymore[cite: 202].

## 3. Error Calculation: The Objective Function
K-Means evaluates the quality of a partition by calculating the **Squared Error**. [cite_start]The algorithm's mathematical goal is to minimize the sum of squared errors across all clusters[cite: 198].

[cite_start]The squared error for a single cluster is the sum of the squared Euclidean distances between every point in that cluster and the cluster's mean (centroid)[cite: 186].

### The Formula
Let $C$ be the set of clusters and $\mu_k$ be the mean of cluster $k$. The total objective function $J(C)$ is defined as:

$$J(C) = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

[cite_start]Where [cite: 196-199]:
* $K$ is the number of clusters.
* $x_i$ is a data point belonging to cluster $C_k$.
* $\mu_k$ is the centroid of cluster $C_k$.
* $||x_i - \mu_k||^2$ is the squared Euclidean distance between the point and the centroid.

**Interpretation:**
* **Lower Error:** Points are packed tightly around their centroids.
* **Minimization:** The algorithm tries to find the specific centroids $\mu_k$ that make $J(C)$ as small as possible. [cite_start]However, finding the global minimum is an NP-hard problem, so K-means only guarantees finding a "local" minimum[cite: 200].

## 4. The Algorithm Steps
[cite_start]The standard K-Means algorithm follows these four specific steps [cite: 201-204]:

1.  **Initialization:** Select an initial partition with $K$ clusters. This is typically done by picking $K$ random points from the dataset to act as the initial centroids.
2.  **Assignment Step:** Generate a new partition by assigning each data point to its closest cluster center. This is done by measuring the distance from the point to every centroid.
3.  **Update Step:** Compute the new cluster centers (centroids) by taking the average (mean) of all points currently assigned to that cluster.
4.  **Convergence:** Repeat steps 2 and 3 until the cluster membership stabilizes (the centroids stop moving).

## 5. Methods to Identify K (Number of Clusters)
[cite_start]Determining the correct number of clusters ($K$) is one of the most difficult problems in data clustering[cite: 320]. [cite_start]The squared error always decreases as you increase $K$, so you cannot simply pick the $K$ with the lowest error (otherwise $K=n$ would always be best)[cite: 200].

Here are the main methods to decide $K$:

### A. The Heuristic Approach (Elbow Method)
This is a standard visual approach. [cite_start]You run the algorithm for a range of $K$ values (e.g., 2 to 10) and plot the error (or variance) on a graph[cite: 208].
* **How it works:** You look for the "elbow" of the curve—the point where adding another cluster yields diminishing returns in error reduction.

### B. Statistical Model Selection (BIC / AIC / MDL)
These methods treat clustering as a model selection problem. [cite_start]They use criteria that balance the goodness of fit against the complexity of the model (the number of parameters)[cite: 321].
* [cite_start]**MML/MDL:** Approaches like Minimum Message Length (MML) or Minimum Description Length (MDL) start with a large number of clusters and merge them if it simplifies the description of the data [cite: 324-325].
* [cite_start]**BIC/AIC:** Variants like X-means use the Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC) to automatically optimize $K$[cite: 326].

### C. Gap Statistics
This statistical technique assumes that the optimal number of clusters is the one where the resulting partition is most resilient to random perturbations. [cite_start]It compares the error of your clustering to the error of a random distribution (noise) [cite: 327-328].

### D. Cross-Validation
Adapted from supervised learning, this method splits the data into training and validation folds. [cite_start]The likelihood of the data in the validation fold (given the clusters found in the training fold) serves as a performance indicator to find the best $K$ [cite: 425-427].

---

## 6. Project Implementation

Below is the Python code used to implement the K-Means algorithm, visualize the intermediate steps, and generate the Confusion Matrix for result validation.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

class KMeansCustom:
    def __init__(self, k=3, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # 1. Initialize Centroids (Randomly select k samples)
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization loop
        for i in range(self.max_iters):
            # Assign samples to closest centroids (Step 2)
            self.clusters = self._create_clusters(self.centroids)
            
            # Visualization for intermediate steps
            if self.plot_steps and (i == 0 or i % 3 == 0):
                self.plot_2d(i, "Intermediate")

            # Calculate new centroids (Step 3)
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check convergence (Step 4)
            if self._is_converged(centroids_old, self.centroids):
                print(f"Converged at iteration {i}")
                break
                
        if self.plot_steps:
            self.plot_2d(i, "Final")
            
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Convert list of indices into a label array
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [np.linalg.norm(sample - point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [np.linalg.norm(centroids_old[i] - centroids_new[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot_2d(self, iteration, title_prefix):
        # Create a figure with 2 subplots: Sepal View and Petal View
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = ['r', 'g', 'b']
        
        # Plot 1: Sepal Length vs Width
        for i, index in enumerate(self.clusters):
            points = self.X[index].T
            ax1.scatter(*points[:2], color=colors[i], label=f'Cluster {i}')
        for point in self.centroids:
            ax1.scatter(*point[:2], marker="x", color="black", s=200, linewidth=3)
        ax1.set_title(f"Sepal Dimensions (Iter {iteration})")
        ax1.set_xlabel("Sepal Length")
        ax1.set_ylabel("Sepal Width")

        # Plot 2: Petal Length vs Width
        for i, index in enumerate(self.clusters):
            points = self.X[index].T
            ax2.scatter(points[2], points[3], color=colors[i], label=f'Cluster {i}')
        for point in self.centroids:
            ax2.scatter(point[2], point[3], marker="x", color="black", s=200, linewidth=3)
        ax2.set_title(f"Petal Dimensions (Iter {iteration})")
        ax2.set_xlabel("Petal Length")
        ax2.set_ylabel("Petal Width")
        
        plt.suptitle(f"{title_prefix} Clustering State")
        plt.show()

# --- Helper Functions for Analysis ---

def plot_elbow_method(X, max_k=10):
    wcss = []
    # We use sklearn here just for speed in generating the curve
    from sklearn.cluster import KMeans
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Squared Error)')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, species_names):
    # Match predicted labels to true labels using mode (since cluster 0 could be any species)
    labels = np.zeros_like(y_pred)
    for i in range(3):
        mask = (y_pred == i)
        if np.any(mask):
            labels[mask] = mode(y_true[mask], keepdims=True)[0][0]
            
    cm = confusion_matrix(y_true, labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Species')
    plt.title('Confusion Matrix (Accuracy)')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    species_names = iris.target_names
    
    # 2. Elbow Method (To show why K=3 is good)
    print("Generating Elbow Method Plot...")
    plot_elbow_method(X)

    # 3. Run Custom K-Means
    print("Running K-Means Algorithm...")
    kmeans = KMeansCustom(k=3, max_iters=100, plot_steps=True)
    y_pred = kmeans.predict(X)

    # 4. Final Accuracy Check
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, species_names)
    
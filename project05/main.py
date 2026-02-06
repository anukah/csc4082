import os
import numpy as np
from kmeans import load_iris_data, run_kmeans, evaluate_clustering
from visualizations import (
    reduce_to_2d,
    plot_final_comparison,
    plot_confusion_matrix,
    plot_feature_pairs
)


def main():
    # Setup
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, 'data', 'iris.txt')
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print("\nERROR: Data file not found!")
        return
    
    # Load data
    print("\nSTEP 1: LOADING DATA")
    X, y_true, species_names = load_iris_data(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Species: {species_names}")
    
    # PCA for visualization
    X_2d, pca = reduce_to_2d(X)
    print(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Run K-Means
    print("\nSTEP 2: K-MEANS CLUSTERING")
    kmeans = run_kmeans(X, k=3, init='k-means++', random_state=42)
    
    print(f"Converged in {kmeans.n_iter_} iterations")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    
    # Evaluate
    accuracy, confusion, cluster_to_class = evaluate_clustering(y_true, kmeans.labels_)
    print(f"\nClustering accuracy: {accuracy:.1%}")
    print(f"Cluster mapping: {cluster_to_class}")
    
    # Generate visualizations
    print("\nSTEP 3: GENERATING VISUALIZATIONS")
    
    plot_final_comparison(X, kmeans.labels_, y_true, species_names, 
                         kmeans.cluster_centers_, results_dir)
    plot_confusion_matrix(confusion, species_names, cluster_to_class, 
                         accuracy, results_dir)
    plot_feature_pairs(X, kmeans.labels_, [f'Cluster {i}' for i in range(3)], 
                      results_dir, is_cluster=True)
    plot_feature_pairs(X, y_true, species_names, results_dir, is_cluster=False)
    
    # Compare initialization methods
    print("\nSTEP 4: COMPARING INITIALIZATION METHODS")
    print("-" * 50)
    print(f"{'Method':<12} {'Seed':<6} {'Accuracy':<12} {'Iterations':<12}")
    print("-" * 50)
    
    for init_method in ['random', 'k-means++']:
        for seed in range(5):
            km = run_kmeans(X, k=3, init=init_method, random_state=seed)
            acc, _, _ = evaluate_clustering(y_true, km.labels_)
            print(f"{init_method:<12} {seed:<6} {acc:.1%}{'':>5} {km.n_iter_}")
    
    print("\nEXECUTION COMPLETE!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

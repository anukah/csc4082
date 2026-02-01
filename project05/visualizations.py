import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def reduce_to_2d(X):
    """Reduce 4D data to 2D using PCA."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


def plot_final_comparison(X, cluster_labels, true_labels, species_names, 
                          centroids, output_dir):
    """Plot k-means clustering vs true species labels."""
    X_2d, pca = reduce_to_2d(X)
    centroids_2d = pca.transform(centroids)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # K-means clustering
    for i in range(3):
        mask = cluster_labels == i
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], 
                       label=f'Cluster {i}', alpha=0.7, edgecolors='white', s=50)
    axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', 
                   marker='X', s=200, edgecolors='white', linewidths=2)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('K-Means Clustering')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # True labels
    for i, species in enumerate(species_names):
        mask = true_labels == i
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], 
                       label=species, alpha=0.7, edgecolors='white', s=50)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('True Species Labels')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('K-Means Clustering vs True Species', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(confusion_matrix, species_names, cluster_to_class, 
                         accuracy, output_dir):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(species_names)))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xticklabels(species_names)
    ax.set_yticklabels([f'Cluster {i}' for i in range(confusion_matrix.shape[0])])
    ax.set_xlabel('True Species')
    ax.set_ylabel('K-Means Cluster')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            color = 'white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black'
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', 
                   color=color, fontsize=14, fontweight='bold')
    
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.1%})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_pairs(X, labels, label_names, output_dir, is_cluster=False):
    """Plot pairwise feature scatter plots."""
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for ax, (i, j) in zip(axes, pairs):
        for k, name in enumerate(label_names):
            mask = labels == k
            ax.scatter(X[mask, i], X[mask, j], c=colors[k], label=name, 
                      alpha=0.7, edgecolors='white', s=40)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    title = 'K-Means Clusters' if is_cluster else 'True Species'
    plt.suptitle(f'{title}: Feature Pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = 'cluster_feature_pairs.png' if is_cluster else 'species_feature_pairs.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

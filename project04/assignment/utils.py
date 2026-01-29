"""
Utility functions for image processing and visualization
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image_pair(path1, path2):
    """
    Load two images in grayscale
    
    Args:
        path1: Path to first image
        path2: Path to second image
    
    Returns:
        tuple: (img1, img2) - grayscale images
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Could not load images from {path1} or {path2}")
    
    return img1, img2

def create_output_directory(path="results"):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def draw_keypoints(img, keypoints, title="Keypoints"):
    """
    Draw keypoints on image
    
    Args:
        img: Input image
        keypoints: List of cv2.KeyPoint objects
        title: Title for the plot
    
    Returns:
        Image with keypoints drawn
    """
    img_with_kp = cv2.drawKeypoints(
        img, 
        keypoints, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_kp

def draw_matches_custom(img1, kp1, img2, kp2, matches, 
                       max_matches=50, title="Matches"):
    """
    Draw matches between two images
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: List of cv2.DMatch objects
        max_matches: Maximum number of matches to display
        title: Title for the plot
    
    Returns:
        Image with matches drawn
    """
    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches[:max_matches], 
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return img_matches

def save_result(img, filename, output_dir="results"):
    """Save result image"""
    create_output_directory(output_dir)
    cv2.imwrite(os.path.join(output_dir, filename), img)
    print(f"Saved: {filename}")

def display_images(images, titles, figsize=(20, 10), save_path=None):
    """
    Display multiple images in a grid
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()

def print_statistics(method_name, num_kp1, num_kp2, num_matches):
    """Print matching statistics"""
    print(f"\n{'='*50}")
    print(f"Method: {method_name}")
    print(f"{'='*50}")
    print(f"Keypoints in Image 1: {num_kp1}")
    print(f"Keypoints in Image 2: {num_kp2}")
    print(f"Number of matches: {num_matches}")
    print(f"{'='*50}\n")
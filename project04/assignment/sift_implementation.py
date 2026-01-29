"""
SIFT-based feature detection, description, and matching
"""
import cv2
import numpy as np

# ============================================================================
# CONCEPT: SIMPLE PATCH DESCRIPTOR
# ============================================================================
# This descriptor simply extracts raw pixel intensities from a local patch
# around each keypoint. It's simple but only works for translation.
# 
# Advantages:
#   - Fast to compute
#   - Easy to understand
# 
# Disadvantages:
#   - Not invariant to rotation
#   - Not invariant to scale
#   - Not invariant to illumination changes
# ============================================================================

def simple_patch_descriptor(img, keypoints, patch_size=5):
    """
    Create simple descriptor using raw pixel intensities in a patch
    
    CONCEPT:
    For each keypoint, extract a patch_size × patch_size window centered
    at the keypoint location and flatten it into a vector.
    
    Args:
        img: Input grayscale image
        keypoints: List of cv2.KeyPoint objects
        patch_size: Size of the patch (must be odd)
    
    Returns:
        valid_kps: Keypoints that had valid patches
        descriptors: Numpy array of descriptors (N × patch_size²)
    """
    half = patch_size // 2
    descriptors = []
    valid_kps = []
    h, w = img.shape
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Check if patch is within image boundaries
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        
        # Extract patch
        patch = img[y - half:y + half + 1, x - half:x + half + 1]
        
        # Flatten and normalize
        descriptor = patch.flatten().astype(np.float32)
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-7)  # Normalize
        
        descriptors.append(descriptor)
        valid_kps.append(kp)
    
    return valid_kps, np.array(descriptors)

# ============================================================================
# CONCEPT: SIFT DETECTOR
# ============================================================================
# SIFT detector finds keypoints using Difference of Gaussians (DoG)
# 
# Process:
# 1. Build scale space: Multiple octaves with increasing blur
# 2. Compute DoG: Subtract adjacent scales
# 3. Find extrema: Points that are max/min in 3×3×3 neighborhood
# 4. Refine: Sub-pixel localization and remove low contrast/edge points
# 5. Orientation: Assign dominant orientation using gradient histogram
# ============================================================================

def sift_detect_only(img, n_features=0, contrast_threshold=0.04):
    """
    Detect keypoints using SIFT detector
    
    Args:
        img: Input grayscale image
        n_features: Number of best features to retain (0 = all)
        contrast_threshold: Threshold for low contrast keypoints
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold
    )
    keypoints = sift.detect(img, None)
    return keypoints

# ============================================================================
# CONCEPT: SIFT DESCRIPTOR
# ============================================================================
# SIFT descriptor creates a 128-dimensional vector for each keypoint
# 
# Process:
# 1. Rotate patch according to keypoint orientation (rotation invariance)
# 2. Divide into 4×4 grid
# 3. For each grid cell, compute 8-bin gradient orientation histogram
# 4. Result: 4×4×8 = 128 dimensions
# 5. Normalize to unit length (illumination invariance)
# ============================================================================

def sift_detect_and_describe(img, n_features=0, contrast_threshold=0.04):
    """
    Detect keypoints and compute SIFT descriptors
    
    Args:
        img: Input grayscale image
        n_features: Number of best features to retain
        contrast_threshold: Threshold for low contrast keypoints
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: Numpy array of descriptors (N × 128)
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold
    )
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

# ============================================================================
# CONCEPT: SUM OF SQUARED DIFFERENCES (SSD) MATCHING
# ============================================================================
# For each descriptor in image 1, find the closest descriptor in image 2
# using Euclidean distance.
# 
# Formula: SSD = Σ(d1[i] - d2[i])²
# 
# This is a brute-force approach and can produce many false matches.
# ============================================================================

def match_ssd(desc1, desc2):
    """
    Match descriptors using Sum of Squared Differences
    
    CONCEPT:
    For each descriptor in desc1, compute SSD distance to all descriptors
    in desc2. The descriptor with minimum distance is the match.
    
    Args:
        desc1: Descriptors from image 1 (N1 × D)
        desc2: Descriptors from image 2 (N2 × D)
    
    Returns:
        matches: List of cv2.DMatch objects
    """
    matches = []
    
    for i, d1 in enumerate(desc1):
        # Compute SSD to all descriptors in desc2
        distances = np.sum((desc2 - d1) ** 2, axis=1)
        
        # Find minimum distance
        j = np.argmin(distances)
        
        # Create DMatch object
        # DMatch(queryIdx, trainIdx, distance)
        matches.append(cv2.DMatch(i, j, float(distances[j])))
    
    return matches

# ============================================================================
# CONCEPT: RATIO TEST MATCHING (LOWE'S METHOD)
# ============================================================================
# Improvement over SSD: Compare best match to second-best match
# 
# Idea: A good match should be significantly better than alternatives
# Accept match only if: distance(best) < ratio × distance(second_best)
# 
# Typical ratio: 0.7-0.8
# Lower ratio = fewer but more reliable matches
# Higher ratio = more matches but more false positives
# ============================================================================

def sift_match_ratio_test(des1, des2, ratio=0.75):
    """
    Match descriptors using Lowe's ratio test
    
    CONCEPT:
    Use k-nearest neighbors (k=2) to find best two matches.
    Accept match only if best match is significantly better than second best.
    
    Args:
        des1: Descriptors from image 1
        des2: Descriptors from image 2
        ratio: Ratio threshold (typically 0.7-0.8)
    
    Returns:
        good_matches: List of cv2.DMatch objects
    """
    # BFMatcher with L2 norm (Euclidean distance)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Find 2 best matches for each descriptor
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for matches in knn_matches:
        if len(matches) == 2:  # Make sure we have 2 matches
            m, n = matches
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    return good_matches

def sift_match_cross_check(des1, des2):
    """
    Match with cross-checking (mutual nearest neighbor)
    
    CONCEPT:
    A match is accepted only if descriptor A in image 1 matches to 
    descriptor B in image 2, AND descriptor B matches back to A.
    
    Args:
        des1: Descriptors from image 1
        des2: Descriptors from image 2
    
    Returns:
        matches: List of cv2.DMatch objects
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches
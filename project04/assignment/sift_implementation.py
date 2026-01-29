import cv2
import numpy as np


def simple_patch_descriptor(img, keypoints, patch_size=5):
    """
    Create simple descriptor using raw pixel intensities in a patch.
    Extracts a patch_size x patch_size window centered at keypoint and flattens it.
    
    Args:
        img: Input grayscale image
        keypoints: List of cv2.KeyPoint objects
        patch_size: Size of the patch (must be odd)
    
    Returns:
        valid_kps: Keypoints that had valid patches
        descriptors: Numpy array of descriptors (N x patch_size^2)
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
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-7)
        
        descriptors.append(descriptor)
        valid_kps.append(kp)
    
    return valid_kps, np.array(descriptors)


def sift_detect_only(img, n_features=0, contrast_threshold=0.04):
    """
    Detect keypoints using SIFT detector (Difference of Gaussians).
    
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


def sift_detect_and_describe(img, n_features=0, contrast_threshold=0.04):
    """
    Detect keypoints and compute SIFT descriptors (128-D vectors).
    
    Args:
        img: Input grayscale image
        n_features: Number of best features to retain
        contrast_threshold: Threshold for low contrast keypoints
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: Numpy array of descriptors (N x 128)
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold
    )
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_ssd(desc1, desc2):
    """
    Match descriptors using Sum of Squared Differences.
    For each descriptor in desc1, find the closest in desc2.
    
    Args:
        desc1: Descriptors from image 1 (N1 x D)
        desc2: Descriptors from image 2 (N2 x D)
    
    Returns:
        matches: List of cv2.DMatch objects
    """
    matches = []
    
    for i, d1 in enumerate(desc1):
        # Compute SSD to all descriptors in desc2
        distances = np.sum((desc2 - d1) ** 2, axis=1)
        
        # Find minimum distance
        j = np.argmin(distances)
        
        matches.append(cv2.DMatch(i, j, float(distances[j])))
    
    return matches


def sift_match_ratio_test(des1, des2, ratio=0.75):
    """
    Match descriptors using Lowe's ratio test.
    Accept match only if best match is significantly better than second best.
    
    Args:
        des1: Descriptors from image 1
        des2: Descriptors from image 2
        ratio: Ratio threshold (typically 0.7-0.8)
    
    Returns:
        good_matches: List of cv2.DMatch objects
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Find 2 best matches for each descriptor
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for matches in knn_matches:
        if len(matches) == 2:
            m, n = matches
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    return good_matches


def sift_match_cross_check(des1, des2):
    """
    Match with cross-checking (mutual nearest neighbor).
    Match accepted only if A matches B and B matches back to A.
    
    Args:
        des1: Descriptors from image 1
        des2: Descriptors from image 2
    
    Returns:
        matches: List of cv2.DMatch objects
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches
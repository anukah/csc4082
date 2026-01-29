"""
Harris corner detection with simple patch descriptors
"""
import cv2
import numpy as np

# ============================================================================
# CONCEPT: HARRIS CORNER DETECTOR
# ============================================================================
# Harris detector finds corners by analyzing how the image intensity 
# changes in different directions.
# 
# Process:
# 1. Compute image gradients (Ix, Iy)
# 2. Compute structure tensor M:
#    M = [Ix²   IxIy ]
#        [IxIy  Iy²  ]
# 3. Compute corner response: R = det(M) - k*trace(M)²
# 4. Threshold R to find corners
# 
# Parameters:
# - blockSize: Size of neighborhood for gradient computation
# - ksize: Aperture parameter for Sobel operator
# - k: Harris detector free parameter (typically 0.04-0.06)
# ============================================================================

def harris_detect(img, block_size=2, ksize=3, k=0.04, thresh=0.01):
    """
    Detect corners using Harris corner detector
    
    Args:
        img: Input grayscale image
        block_size: Neighborhood size for gradient computation
        ksize: Aperture for Sobel operator (must be odd)
        k: Harris detector parameter
        thresh: Threshold as fraction of maximum response
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
    """
    # Ensure image is float32
    img_float = img.astype(np.float32)
    
    # Compute Harris corner response
    harris_response = cv2.cornerHarris(img_float, block_size, ksize, k)
    
    # Dilate to get local maxima
    harris_response = cv2.dilate(harris_response, None)
    
    # Threshold
    threshold = thresh * harris_response.max()
    
    # Extract keypoint locations
    keypoints = []
    for y in range(harris_response.shape[0]):
        for x in range(harris_response.shape[1]):
            if harris_response[y, x] > threshold:
                # Create KeyPoint with response value
                kp = cv2.KeyPoint(
                    float(x), 
                    float(y), 
                    1,  # size
                    -1,  # angle
                    harris_response[y, x]  # response
                )
                keypoints.append(kp)
    
    return keypoints

def harris_detect_optimized(img, block_size=2, ksize=3, k=0.04, 
                           thresh=0.01, max_keypoints=500):
    """
    Harris detection with non-maximum suppression
    
    CONCEPT:
    Instead of taking all points above threshold, apply non-maximum 
    suppression to get only the strongest corners in each local region.
    
    Args:
        img: Input grayscale image
        block_size: Neighborhood size
        ksize: Aperture for Sobel
        k: Harris parameter
        thresh: Threshold as fraction of max
        max_keypoints: Maximum number of keypoints to return
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
    """
    img_float = img.astype(np.float32)
    harris_response = cv2.cornerHarris(img_float, block_size, ksize, k)
    
    # Apply non-maximum suppression
    window_size = 5
    half_window = window_size // 2
    h, w = harris_response.shape
    
    keypoints = []
    threshold = thresh * harris_response.max()
    
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            local_region = harris_response[
                y - half_window:y + half_window + 1,
                x - half_window:x + half_window + 1
            ]
            center_value = harris_response[y, x]
            
            # Check if center is local maximum and above threshold
            if center_value > threshold and center_value == local_region.max():
                kp = cv2.KeyPoint(float(x), float(y), 1, -1, center_value)
                keypoints.append(kp)
    
    # Sort by response and keep top keypoints
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    return keypoints[:max_keypoints]

# ============================================================================
# CONCEPT: DESCRIPTOR FOR HARRIS KEYPOINTS
# ============================================================================
# Harris only detects corners - it doesn't create descriptors.
# We use the same simple patch descriptor as before.
# ============================================================================

def harris_describe(img, keypoints, patch_size=7):
    """
    Create descriptors for Harris keypoints using pixel intensities
    
    This is the same as simple_patch_descriptor but included here
    for clarity in the Harris pipeline.
    
    Args:
        img: Input grayscale image
        keypoints: List of cv2.KeyPoint objects
        patch_size: Size of patch (must be odd)
    
    Returns:
        valid_kps: Keypoints with valid patches
        descriptors: Numpy array of descriptors
    """
    half = patch_size // 2
    descriptors = []
    valid_kps = []
    h, w = img.shape
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        
        patch = img[y - half:y + half + 1, x - half:x + half + 1]
        descriptor = patch.flatten().astype(np.float32)
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-7)
        
        descriptors.append(descriptor)
        valid_kps.append(kp)
    
    return valid_kps, np.array(descriptors)

# ============================================================================
# CONCEPT: MATCHING HARRIS DESCRIPTORS
# ============================================================================
# We can use either SSD matching or OpenCV's matchers
# ============================================================================

def match_harris_descriptors(desc1, desc2, method='ssd', ratio=0.75):
    """
    Match Harris descriptors
    
    Args:
        desc1: Descriptors from image 1
        desc2: Descriptors from image 2
        method: 'ssd' or 'ratio_test'
        ratio: Ratio threshold for ratio test
    
    Returns:
        matches: List of cv2.DMatch objects
    """
    if method == 'ssd':
        matches = []
        for i, d1 in enumerate(desc1):
            distances = np.sum((desc2 - d1) ** 2, axis=1)
            j = np.argmin(distances)
            matches.append(cv2.DMatch(i, j, float(distances[j])))
        return matches
    
    elif method == 'ratio_test':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for matches in knn_matches:
            if len(matches) == 2:
                m, n = matches
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        return good_matches
    
    else:
        raise ValueError(f"Unknown method: {method}")
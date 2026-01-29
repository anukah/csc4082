import cv2
import numpy as np
from sift_implementation import *
from harris_implementation import *
from utils import *


def compare_sift_ratio(img1, img2, ratios=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Compare SIFT matching with different ratio thresholds.
    Lower ratio = stricter matching, fewer but more reliable matches.
    """
    print("\nCOMPARING SIFT WITH DIFFERENT RATIO THRESHOLDS")
    
    # Detect and describe once
    kp1, des1 = sift_detect_and_describe(img1)
    kp2, des2 = sift_detect_and_describe(img2)
    
    results = []
    images = []
    titles = []
    
    for ratio in ratios:
        matches = sift_match_ratio_test(des1, des2, ratio=ratio)
        
        print(f"\nRatio: {ratio}")
        print(f"  Number of matches: {len(matches)}")
        
        img_matches = draw_matches_custom(
            img1, kp1, img2, kp2, matches, max_matches=50
        )
        
        images.append(img_matches)
        titles.append(f"Ratio={ratio}, Matches={len(matches)}")
        
        results.append({
            'ratio': ratio,
            'num_matches': len(matches),
            'matches': matches
        })
    
    display_images(images, titles, figsize=(20, 15), 
                  save_path="results/sift_ratio_comparison.png")
    
    return results


def compare_harris_window_size(img1, img2, window_sizes=[3, 5, 7, 9, 11]):
    """
    Compare Harris with different descriptor window sizes.
    Smaller window = more local, sensitive to noise.
    Larger window = more context, less discriminative.
    """
    print("\nCOMPARING HARRIS WITH DIFFERENT WINDOW SIZES")
    
    # Detect keypoints once
    kp1 = harris_detect_optimized(img1, block_size=2, ksize=3, 
                                   k=0.04, thresh=0.01, max_keypoints=500)
    kp2 = harris_detect_optimized(img2, block_size=2, ksize=3, 
                                   k=0.04, thresh=0.01, max_keypoints=500)
    
    results = []
    images = []
    titles = []
    
    for window_size in window_sizes:
        kp1_valid, des1 = harris_describe(img1, kp1, patch_size=window_size)
        kp2_valid, des2 = harris_describe(img2, kp2, patch_size=window_size)
        
        matches = match_harris_descriptors(des1, des2, method='ratio_test', 
                                          ratio=0.8)
        
        print(f"\nWindow size: {window_size}x{window_size}")
        print(f"  Valid keypoints img1: {len(kp1_valid)}")
        print(f"  Valid keypoints img2: {len(kp2_valid)}")
        print(f"  Number of matches: {len(matches)}")
        
        img_matches = draw_matches_custom(
            img1, kp1_valid, img2, kp2_valid, matches, max_matches=50
        )
        
        images.append(img_matches)
        titles.append(f"Window={window_size}x{window_size}, Matches={len(matches)}")
        
        results.append({
            'window_size': window_size,
            'num_kp1': len(kp1_valid),
            'num_kp2': len(kp2_valid),
            'num_matches': len(matches)
        })
    
    display_images(images, titles, figsize=(20, 15),
                  save_path="results/harris_window_comparison.png")
    
    return results


def compare_harris_parameters(img1, img2):
    """
    Compare Harris with different detector parameters.
    """
    print("\nCOMPARING HARRIS DETECTOR PARAMETERS")
    
    parameter_sets = [
        {'block_size': 2, 'ksize': 3, 'k': 0.04, 'thresh': 0.01, 'name': 'Default'},
        {'block_size': 3, 'ksize': 5, 'k': 0.04, 'thresh': 0.01, 'name': 'Larger block'},
        {'block_size': 2, 'ksize': 3, 'k': 0.06, 'thresh': 0.01, 'name': 'Higher k'},
        {'block_size': 2, 'ksize': 3, 'k': 0.04, 'thresh': 0.05, 'name': 'Higher threshold'},
    ]
    
    results = []
    images = []
    titles = []
    
    for params in parameter_sets:
        name = params.pop('name')
        
        kp1 = harris_detect_optimized(img1, max_keypoints=500, **params)
        kp2 = harris_detect_optimized(img2, max_keypoints=500, **params)
        
        kp1_valid, des1 = harris_describe(img1, kp1, patch_size=7)
        kp2_valid, des2 = harris_describe(img2, kp2, patch_size=7)
        
        matches = match_harris_descriptors(des1, des2, method='ratio_test')
        
        print(f"\n{name}:")
        print(f"  Parameters: {params}")
        print(f"  Keypoints: {len(kp1_valid)}, {len(kp2_valid)}")
        print(f"  Matches: {len(matches)}")
        
        img_matches = draw_matches_custom(
            img1, kp1_valid, img2, kp2_valid, matches, max_matches=50
        )
        
        images.append(img_matches)
        titles.append(f"{name}\nKP:{len(kp1_valid)},{len(kp2_valid)} M:{len(matches)}")
        
        results.append({
            'name': name,
            'params': params,
            'num_matches': len(matches)
        })
    
    display_images(images, titles, figsize=(20, 12),
                  save_path="results/harris_param_comparison.png")
    
    return results


def compare_sift_vs_harris(img1, img2):
    """
    Direct comparison between SIFT and Harris.
    SIFT: Better for scale/rotation changes. Harris: Faster, good for corners.
    """
    print("\nCOMPARING SIFT VS HARRIS")
    
    # SIFT pipeline
    print("\nRunning SIFT...")
    kp1_sift, des1_sift = sift_detect_and_describe(img1)
    kp2_sift, des2_sift = sift_detect_and_describe(img2)
    matches_sift = sift_match_ratio_test(des1_sift, des2_sift, ratio=0.75)
    
    print(f"SIFT - KP: {len(kp1_sift)}, {len(kp2_sift)}, Matches: {len(matches_sift)}")
    
    # Harris pipeline
    print("\nRunning Harris...")
    kp1_harris = harris_detect_optimized(img1, max_keypoints=500)
    kp2_harris = harris_detect_optimized(img2, max_keypoints=500)
    kp1_harris, des1_harris = harris_describe(img1, kp1_harris, patch_size=7)
    kp2_harris, des2_harris = harris_describe(img2, kp2_harris, patch_size=7)
    matches_harris = match_harris_descriptors(des1_harris, des2_harris, 
                                             method='ratio_test', ratio=0.75)
    
    print(f"Harris - KP: {len(kp1_harris)}, {len(kp2_harris)}, Matches: {len(matches_harris)}")
    
    # Visualize
    img_sift = draw_matches_custom(img1, kp1_sift, img2, kp2_sift, 
                                   matches_sift, max_matches=50)
    img_harris = draw_matches_custom(img1, kp1_harris, img2, kp2_harris, 
                                     matches_harris, max_matches=50)
    
    display_images(
        [img_sift, img_harris],
        [f"SIFT (Matches: {len(matches_sift)})", 
         f"Harris (Matches: {len(matches_harris)})"],
        figsize=(20, 10),
        save_path="results/sift_vs_harris.png"
    )
    
    return {
        'sift': {'kp1': len(kp1_sift), 'kp2': len(kp2_sift), 
                 'matches': len(matches_sift)},
        'harris': {'kp1': len(kp1_harris), 'kp2': len(kp2_harris), 
                   'matches': len(matches_harris)}
    }
import cv2
import numpy as np
import sys
import os

from utils import *
from sift_implementation import *
from harris_implementation import *
from parameter_comparison import *

def demonstrate_sift_simple_descriptor(img1, img2):
    """Step 1a: SIFT detector + Simple patch descriptor"""
    print("\nSTEP 1a: SIFT DETECTOR + SIMPLE PATCH DESCRIPTOR (5x5)")
    
    # Detect keypoints using SIFT
    kp1 = sift_detect_only(img1)
    kp2 = sift_detect_only(img2)
    
    print(f"Detected {len(kp1)} keypoints in image 1")
    print(f"Detected {len(kp2)} keypoints in image 2")
    
    # Create simple descriptors
    kp1_valid, des1 = simple_patch_descriptor(img1, kp1, patch_size=5)
    kp2_valid, des2 = simple_patch_descriptor(img2, kp2, patch_size=5)
    
    print(f"Valid descriptors: {len(kp1_valid)}, {len(kp2_valid)}")
    
    # Match using SSD
    matches = match_ssd(des1, des2)
    
    print(f"Number of matches (SSD): {len(matches)}")
    
    # Visualize
    img_kp1 = draw_keypoints(img1, kp1_valid, "Keypoints Image 1")
    img_kp2 = draw_keypoints(img2, kp2_valid, "Keypoints Image 2")
    img_matches = draw_matches_custom(img1, kp1_valid, img2, kp2_valid, 
                                     matches, max_matches=50)
    
    save_result(img_kp1, "1a_keypoints_img1.jpg")
    save_result(img_kp2, "1a_keypoints_img2.jpg")
    save_result(img_matches, "1a_matches_simple_descriptor.jpg")
    
    display_images(
        [img_kp1, img_kp2, img_matches],
        ["Keypoints Image 1", "Keypoints Image 2", 
         f"Matches (Simple Descriptor): {len(matches)}"],
        figsize=(20, 10),
        save_path="results/1a_sift_simple_descriptor.png"
    )
    
    return kp1_valid, kp2_valid, des1, des2, matches

def demonstrate_sift_full(img1, img2):
    """Step 1b: SIFT detector + SIFT descriptor"""
    print("\nSTEP 1b: SIFT DETECTOR + SIFT DESCRIPTOR (128-D)")
    
    # Detect and describe
    kp1, des1 = sift_detect_and_describe(img1)
    kp2, des2 = sift_detect_and_describe(img2)
    
    print(f"Detected {len(kp1)} keypoints in image 1")
    print(f"Detected {len(kp2)} keypoints in image 2")
    print(f"Descriptor dimensionality: {des1.shape[1]}")
    
    # Match using ratio test
    matches_ratio = sift_match_ratio_test(des1, des2, ratio=0.75)
    
    print(f"Number of matches (ratio test, ratio=0.75): {len(matches_ratio)}")
    
    # Also try cross-check matching
    matches_cross = sift_match_cross_check(des1, des2)
    
    print(f"Number of matches (cross-check): {len(matches_cross)}")
    
    # Visualize
    img_kp1 = draw_keypoints(img1, kp1)
    img_kp2 = draw_keypoints(img2, kp2)
    img_matches_ratio = draw_matches_custom(img1, kp1, img2, kp2, 
                                           matches_ratio, max_matches=50)
    img_matches_cross = draw_matches_custom(img1, kp1, img2, kp2, 
                                           matches_cross, max_matches=50)
    
    save_result(img_kp1, "1b_sift_keypoints_img1.jpg")
    save_result(img_kp2, "1b_sift_keypoints_img2.jpg")
    save_result(img_matches_ratio, "1b_matches_ratio_test.jpg")
    save_result(img_matches_cross, "1b_matches_cross_check.jpg")
    
    display_images(
        [img_kp1, img_kp2, img_matches_ratio, img_matches_cross],
        ["SIFT Keypoints Img1", "SIFT Keypoints Img2",
         f"Matches (Ratio Test): {len(matches_ratio)}",
         f"Matches (Cross-Check): {len(matches_cross)}"],
        figsize=(20, 12),
        save_path="results/1b_sift_full.png"
    )
    
    return kp1, kp2, des1, des2, matches_ratio

def demonstrate_harris(img1, img2):
    """Step 2: Harris detector + Simple descriptor"""
    print("\nSTEP 2: HARRIS CORNER DETECTOR + SIMPLE DESCRIPTOR")
    
    # Detect corners
    kp1 = harris_detect_optimized(img1, block_size=2, ksize=3, k=0.04, 
                                   thresh=0.01, max_keypoints=500)
    kp2 = harris_detect_optimized(img2, block_size=2, ksize=3, k=0.04, 
                                   thresh=0.01, max_keypoints=500)
    
    print(f"Detected {len(kp1)} corners in image 1")
    print(f"Detected {len(kp2)} corners in image 2")
    
    # Create descriptors (7x7 window)
    kp1_valid, des1 = harris_describe(img1, kp1, patch_size=7)
    kp2_valid, des2 = harris_describe(img2, kp2, patch_size=7)
    
    print(f"Valid descriptors: {len(kp1_valid)}, {len(kp2_valid)}")
    
    # Match using SSD
    matches_ssd = match_harris_descriptors(des1, des2, method='ssd')
    
    print(f"Number of matches (SSD): {len(matches_ssd)}")
    
    # Match using ratio test
    matches_ratio = match_harris_descriptors(des1, des2, method='ratio_test', 
                                            ratio=0.8)
    
    print(f"Number of matches (ratio test): {len(matches_ratio)}")
    
    # Visualize
    img_kp1 = draw_keypoints(img1, kp1_valid)
    img_kp2 = draw_keypoints(img2, kp2_valid)
    img_matches_ssd = draw_matches_custom(img1, kp1_valid, img2, kp2_valid, 
                                         matches_ssd, max_matches=50)
    img_matches_ratio = draw_matches_custom(img1, kp1_valid, img2, kp2_valid, 
                                           matches_ratio, max_matches=50)
    
    save_result(img_kp1, "2_harris_corners_img1.jpg")
    save_result(img_kp2, "2_harris_corners_img2.jpg")
    save_result(img_matches_ssd, "2_harris_matches_ssd.jpg")
    save_result(img_matches_ratio, "2_harris_matches_ratio.jpg")
    
    display_images(
        [img_kp1, img_kp2, img_matches_ssd, img_matches_ratio],
        ["Harris Corners Img1", "Harris Corners Img2",
         f"Matches (SSD): {len(matches_ssd)}",
         f"Matches (Ratio Test): {len(matches_ratio)}"],
        figsize=(20, 12),
        save_path="results/2_harris.png"
    )
    
    return kp1_valid, kp2_valid, des1, des2, matches_ratio

def run_parameter_comparisons(img1, img2):
    """Step 3: Compare different parameters"""
    print("\nSTEP 3: PARAMETER COMPARISONS")
    
    # Compare SIFT ratio thresholds
    sift_results = compare_sift_ratio(img1, img2, 
                                      ratios=[0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Compare Harris window sizes
    harris_window_results = compare_harris_window_size(img1, img2, 
                                                       window_sizes=[3, 5, 7, 9, 11])
    
    # Compare Harris detector parameters
    harris_param_results = compare_harris_parameters(img1, img2)
    
    # Direct SIFT vs Harris comparison
    comparison = compare_sift_vs_harris(img1, img2)
    
    return {
        'sift_ratio': sift_results,
        'harris_window': harris_window_results,
        'harris_params': harris_param_results,
        'sift_vs_harris': comparison
    }

def main():
    """Main execution function"""
    print("\nSIFT AND HARRIS FEATURE MATCHING ASSIGNMENT")
    
    # Create output directory
    create_output_directory("results")
    
    # Load images
    image_pairs = [
        ("images/scale_img1.jpg", "images/scale_img2.jpg"),
    ]
    
    # Check if images exist
    if not os.path.exists(image_pairs[0][0]):
        print("\nERROR: Sample images not found!")
        print("Please place your test images in the 'images/' directory")
        print("Expected files:")
        for path1, path2 in image_pairs:
            print(f"  - {path1}")
            print(f"  - {path2}")
        
        print("\nTIP: You can create test images by:")
        print("  1. Taking photos of the same object from different angles")
        print("  2. Taking photos with different lighting")
        print("  3. Taking photos at different scales")
        print("  4. Using online datasets (e.g., Oxford Visual Geometry Group)")
        
        return
    
    # Process first pair
    img1, img2 = load_image_pair(image_pairs[0][0], image_pairs[0][1])
    
    print(f"\nLoaded images:")
    print(f"  Image 1: {img1.shape}")
    print(f"  Image 2: {img2.shape}")
    
    # Run all demonstrations
    
    # Step 1a: SIFT detector + simple descriptor
    results_1a = demonstrate_sift_simple_descriptor(img1, img2)
    
    # Step 1b: SIFT detector + SIFT descriptor
    results_1b = demonstrate_sift_full(img1, img2)
    
    # Step 2: Harris detector + simple descriptor
    results_2 = demonstrate_harris(img1, img2)
    
    # Step 3: Parameter comparisons
    results_3 = run_parameter_comparisons(img1, img2)
    
    # Summary
    print("\nEXECUTION COMPLETE!")
    print("\nAll results have been saved to the 'results/' directory")
    print("\nGenerated files:")
    print("  - Individual result images (1a_*, 1b_*, 2_*)")
    print("  - Comparison plots (*_comparison.png)")
    print("  - SIFT vs Harris comparison (sift_vs_harris.png)")

if __name__ == "__main__":
    main()
"""
Create test image pairs from sample images with various transformations
This generates pairs for testing SIFT and Harris algorithms
"""
import cv2
import numpy as np
import os

def create_output_dir(path="images"):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"✅ Output directory: {path}")

def load_sample_image(path):
    """Load a sample image"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img

def save_pair(img1, img2, name, output_dir="images"):
    """Save image pair"""
    cv2.imwrite(f"{output_dir}/{name}_img1.jpg", img1)
    cv2.imwrite(f"{output_dir}/{name}_img2.jpg", img2)
    print(f"✅ Created: {name}_img1.jpg and {name}_img2.jpg")

# ============================================================================
# TRANSFORMATION 1: TRANSLATION
# ============================================================================
def create_translation_pair(img, tx=50, ty=30, name="translation"):
    """
    Create image pair with translation
    
    CONCEPT: Tests if descriptors work when object moves in image
    - SIFT: Should work well
    - Harris + simple descriptor: Should work well
    
    Args:
        img: Input image
        tx: Translation in x direction (pixels)
        ty: Translation in y direction (pixels)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating TRANSLATION pair (tx={tx}, ty={ty})")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    
    # Original image
    img1 = img.copy()
    
    # Create translated version
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 2: ROTATION
# ============================================================================
def create_rotation_pair(img, angle=30, name="rotation"):
    """
    Create image pair with rotation
    
    CONCEPT: Tests rotation invariance
    - SIFT: Should work well (rotation invariant)
    - Harris + simple descriptor: Will fail (not rotation invariant)
    
    Args:
        img: Input image
        angle: Rotation angle in degrees
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating ROTATION pair (angle={angle}°)")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Original image
    img1 = img.copy()
    
    # Create rotated version
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 3: SCALE (ZOOM)
# ============================================================================
def create_scale_pair(img, scale_factor=1.3, name="scale"):
    """
    Create image pair with scale change
    
    CONCEPT: Tests scale invariance
    - SIFT: Should work well (scale invariant)
    - Harris + simple descriptor: Will fail (not scale invariant)
    
    Args:
        img: Input image
        scale_factor: Scale factor (>1 for zoom in, <1 for zoom out)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating SCALE pair (factor={scale_factor})")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    
    # Original image
    img1 = img.copy()
    
    # Create scaled version
    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                           interpolation=cv2.INTER_LINEAR)
    
    # Crop or pad to original size
    h_scaled, w_scaled = img_scaled.shape[:2]
    
    if scale_factor > 1:  # Zoom in - crop center
        start_y = (h_scaled - h) // 2
        start_x = (w_scaled - w) // 2
        img2 = img_scaled[start_y:start_y + h, start_x:start_x + w]
    else:  # Zoom out - pad with black
        img2 = np.zeros_like(img)
        start_y = (h - h_scaled) // 2
        start_x = (w - w_scaled) // 2
        img2[start_y:start_y + h_scaled, start_x:start_x + w_scaled] = img_scaled
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 4: BRIGHTNESS CHANGE
# ============================================================================
def create_brightness_pair(img, brightness_offset=50, name="brightness"):
    """
    Create image pair with brightness change
    
    CONCEPT: Tests illumination invariance
    - SIFT: Should work reasonably well
    - Harris + simple descriptor: May struggle
    
    Args:
        img: Input image
        brightness_offset: Value to add (can be negative)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating BRIGHTNESS pair (offset={brightness_offset})")
    print(f"{'='*60}")
    
    # Original image
    img1 = img.copy()
    
    # Create brighter/darker version
    img2 = cv2.add(img, brightness_offset)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 5: ROTATION + TRANSLATION
# ============================================================================
def create_rotation_translation_pair(img, angle=20, tx=30, ty=20, 
                                    name="rotation_translation"):
    """
    Create image pair with rotation AND translation
    
    CONCEPT: Tests combined transformations (more realistic)
    - SIFT: Should work well
    - Harris + simple descriptor: Will struggle
    
    Args:
        img: Input image
        angle: Rotation angle
        tx, ty: Translation
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating ROTATION+TRANSLATION pair (angle={angle}°, tx={tx}, ty={ty})")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Original image
    img1 = img.copy()
    
    # Create transformed version
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx  # Add translation
    M[1, 2] += ty
    
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 6: SCALE + ROTATION
# ============================================================================
def create_scale_rotation_pair(img, scale=1.2, angle=25, name="scale_rotation"):
    """
    Create image pair with scale AND rotation
    
    CONCEPT: Tests if SIFT can handle multiple transformations
    - SIFT: Should work well (both invariances)
    - Harris: Will fail
    
    Args:
        img: Input image
        scale: Scale factor
        angle: Rotation angle
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating SCALE+ROTATION pair (scale={scale}, angle={angle}°)")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Original image
    img1 = img.copy()
    
    # Create transformed version
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 7: PERSPECTIVE CHANGE
# ============================================================================
def create_perspective_pair(img, strength=0.15, name="perspective"):
    """
    Create image pair with perspective transformation
    
    CONCEPT: Simulates viewing object from different angle
    - SIFT: May work with mild perspective changes
    - Harris: Will struggle
    
    Args:
        img: Input image
        strength: Strength of perspective (0-1, typical 0.1-0.3)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating PERSPECTIVE pair (strength={strength})")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    
    # Original image
    img1 = img.copy()
    
    # Define perspective transformation
    offset = int(w * strength)
    
    src_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    
    dst_points = np.float32([
        [offset, 0],
        [w - offset, 0],
        [w, h],
        [0, h]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    img2 = cv2.warpPerspective(img, M, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 8: NOISE
# ============================================================================
def create_noise_pair(img, noise_level=25, name="noise"):
    """
    Create image pair with Gaussian noise
    
    CONCEPT: Tests robustness to noise
    - SIFT: Generally robust due to descriptor normalization
    - Harris: May be sensitive
    
    Args:
        img: Input image
        noise_level: Standard deviation of Gaussian noise
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating NOISE pair (level={noise_level})")
    print(f"{'='*60}")
    
    # Original image
    img1 = img.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, img.shape)
    img2 = img.astype(np.float32) + noise
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 9: BLUR
# ============================================================================
def create_blur_pair(img, kernel_size=7, name="blur"):
    """
    Create image pair with blur
    
    CONCEPT: Simulates out-of-focus or motion blur
    - SIFT: Fairly robust
    - Harris: Corners may become less sharp
    
    Args:
        img: Input image
        kernel_size: Size of blur kernel (must be odd)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating BLUR pair (kernel={kernel_size}×{kernel_size})")
    print(f"{'='*60}")
    
    # Original image
    img1 = img.copy()
    
    # Apply Gaussian blur
    img2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# TRANSFORMATION 10: CROPPED VIEW
# ============================================================================
def create_crop_pair(img, crop_percentage=0.3, name="crop"):
    """
    Create image pair with cropped view
    
    CONCEPT: Tests partial overlap (common in real scenarios)
    - SIFT: Should work if enough overlap
    - Harris: May work with translation-only crops
    
    Args:
        img: Input image
        crop_percentage: How much to crop (0-1)
        name: Output filename prefix
    """
    print(f"\n{'='*60}")
    print(f"Creating CROP pair (crop={crop_percentage*100}%)")
    print(f"{'='*60}")
    
    h, w = img.shape[:2]
    
    # Original image
    img1 = img.copy()
    
    # Create cropped version (zoom in on center-right)
    crop_h = int(h * (1 - crop_percentage))
    crop_w = int(w * (1 - crop_percentage))
    
    start_y = (h - crop_h) // 2
    start_x = int(w * 0.3)  # Offset to right
    
    cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Resize back to original size
    img2 = cv2.resize(cropped, (w, h))
    
    save_pair(img1, img2, name)
    
    return img1, img2

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def create_all_pairs_from_image(input_image_path, output_dir="images"):
    """
    Create all test pairs from a single input image
    
    Args:
        input_image_path: Path to input sample image
        output_dir: Directory to save output pairs
    """
    print("\n" + "="*70)
    print("CREATING TEST IMAGE PAIRS FROM SAMPLE IMAGE")
    print("="*70)
    print(f"Input image: {input_image_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Load sample image
    print("\n📥 Loading sample image...")
    img = load_sample_image(input_image_path)
    print(f"✅ Loaded image: {img.shape}")
    
    # Convert to grayscale for some transformations
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Back to BGR for consistency
    else:
        img_gray = img
    
    # Create all transformation pairs
    transformations = [
        # (function, image_to_use, args, kwargs)
        (create_translation_pair, img, (), {'tx': 50, 'ty': 30, 'name': 'translation'}),
        (create_rotation_pair, img, (), {'angle': 30, 'name': 'rotation'}),
        (create_scale_pair, img, (), {'scale_factor': 1.3, 'name': 'scale'}),
        (create_brightness_pair, img, (), {'brightness_offset': 50, 'name': 'brightness'}),
        (create_rotation_translation_pair, img, (), {'angle': 20, 'tx': 30, 'ty': 20, 'name': 'rotation_translation'}),
        (create_scale_rotation_pair, img, (), {'scale': 1.2, 'angle': 25, 'name': 'scale_rotation'}),
        (create_perspective_pair, img, (), {'strength': 0.15, 'name': 'perspective'}),
        (create_noise_pair, img, (), {'noise_level': 25, 'name': 'noise'}),
        (create_blur_pair, img, (), {'kernel_size': 7, 'name': 'blur'}),
        (create_crop_pair, img, (), {'crop_percentage': 0.3, 'name': 'crop'}),
    ]
    
    # Execute all transformations
    for func, image, args, kwargs in transformations:
        try:
            func(image, *args, **kwargs)
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {e}")
    
    print("\n" + "="*70)
    print("✅ ALL TEST PAIRS CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated {len(transformations)} image pairs in '{output_dir}/'")
    print("\nPairs created:")
    print("  1. translation_img1.jpg / translation_img2.jpg")
    print("  2. rotation_img1.jpg / rotation_img2.jpg")
    print("  3. scale_img1.jpg / scale_img2.jpg")
    print("  4. brightness_img1.jpg / brightness_img2.jpg")
    print("  5. rotation_translation_img1.jpg / rotation_translation_img2.jpg")
    print("  6. scale_rotation_img1.jpg / scale_rotation_img2.jpg")
    print("  7. perspective_img1.jpg / perspective_img2.jpg")
    print("  8. noise_img1.jpg / noise_img2.jpg")
    print("  9. blur_img1.jpg / blur_img2.jpg")
    print(" 10. crop_img1.jpg / crop_img2.jpg")
    print("\n" + "="*70)

def create_pairs_from_multiple_images(image_paths, output_dir="images"):
    """
    Create test pairs from multiple input images
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save output pairs
    """
    create_output_dir(output_dir)
    
    for i, path in enumerate(image_paths):
        print(f"\n{'='*70}")
        print(f"Processing image {i+1}/{len(image_paths)}: {path}")
        print(f"{'='*70}")
        
        try:
            img = load_sample_image(path)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            # Create a few key transformations for each image
            create_translation_pair(img, name=f"{base_name}_translation")
            create_rotation_pair(img, name=f"{base_name}_rotation")
            create_scale_pair(img, name=f"{base_name}_scale")
            
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
    
    print(f"\n✅ Processed {len(image_paths)} images")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """
    Main function with command-line interface
    """
    import sys
    
    print("\n" + "="*70)
    print("TEST IMAGE PAIR GENERATOR FOR SIFT/HARRIS ASSIGNMENT")
    print("="*70)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\n📋 USAGE:")
        print("  python create_test_pairs.py <input_image_path> [output_dir]")
        print("\nEXAMPLES:")
        print("  python create_test_pairs.py sample.jpg")
        print("  python create_test_pairs.py sample.jpg test_images")
        print("  python create_test_pairs.py images/building.jpg images/pairs")
        print("\n" + "="*70)
        
        # Try to find sample images automatically
        sample_paths = [
            "sample.jpg", "sample.png",
            "img1.jpg", "image.jpg",
            "test.jpg", "test.png"
        ]
        
        found_sample = None
        for path in sample_paths:
            if os.path.exists(path):
                found_sample = path
                break
        
        if found_sample:
            print(f"\n💡 Found sample image: {found_sample}")
            response = input("Use this image? (y/n): ").strip().lower()
            if response == 'y':
                create_all_pairs_from_image(found_sample)
                return
        
        print("\n❌ No input image provided and no sample image found.")
        print("Please provide an image path as argument.")
        return
    
    # Get input path and output directory
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "images"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"\n❌ ERROR: Input image not found: {input_path}")
        return
    
    # Create all pairs
    create_all_pairs_from_image(input_path, output_dir)

if __name__ == "__main__":
    main()
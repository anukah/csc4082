import cv2
import numpy as np
import os
import sys


def create_output_dir(path="images"):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Output directory: {path}")


def load_sample_image(path):
    """Load a sample image."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img


def save_pair(img1, img2, name, output_dir="images"):
    """Save image pair."""
    cv2.imwrite(f"{output_dir}/{name}_img1.jpg", img1)
    cv2.imwrite(f"{output_dir}/{name}_img2.jpg", img2)
    print(f"Created: {name}_img1.jpg and {name}_img2.jpg")


def create_translation_pair(img, tx=50, ty=30, name="translation"):
    """Create image pair with translation."""
    print(f"\nCreating TRANSLATION pair (tx={tx}, ty={ty})")
    
    h, w = img.shape[:2]
    img1 = img.copy()
    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_rotation_pair(img, angle=30, name="rotation"):
    """Create image pair with rotation."""
    print(f"\nCreating ROTATION pair (angle={angle} deg)")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    img1 = img.copy()
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_scale_pair(img, scale_factor=1.3, name="scale"):
    """Create image pair with scale change."""
    print(f"\nCreating SCALE pair (factor={scale_factor})")
    
    h, w = img.shape[:2]
    img1 = img.copy()
    
    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                           interpolation=cv2.INTER_LINEAR)
    
    h_scaled, w_scaled = img_scaled.shape[:2]
    
    if scale_factor > 1:
        start_y = (h_scaled - h) // 2
        start_x = (w_scaled - w) // 2
        img2 = img_scaled[start_y:start_y + h, start_x:start_x + w]
    else:
        img2 = np.zeros_like(img)
        start_y = (h - h_scaled) // 2
        start_x = (w - w_scaled) // 2
        img2[start_y:start_y + h_scaled, start_x:start_x + w_scaled] = img_scaled
    
    save_pair(img1, img2, name)
    return img1, img2


def create_brightness_pair(img, brightness_offset=50, name="brightness"):
    """Create image pair with brightness change."""
    print(f"\nCreating BRIGHTNESS pair (offset={brightness_offset})")
    
    img1 = img.copy()
    img2 = cv2.add(img, brightness_offset)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    save_pair(img1, img2, name)
    return img1, img2


def create_rotation_translation_pair(img, angle=20, tx=30, ty=20, 
                                    name="rotation_translation"):
    """Create image pair with rotation AND translation."""
    print(f"\nCreating ROTATION+TRANSLATION pair (angle={angle} deg, tx={tx}, ty={ty})")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    img1 = img.copy()
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_scale_rotation_pair(img, scale=1.2, angle=25, name="scale_rotation"):
    """Create image pair with scale AND rotation."""
    print(f"\nCreating SCALE+ROTATION pair (scale={scale}, angle={angle} deg)")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    img1 = img.copy()
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img, M, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_perspective_pair(img, strength=0.15, name="perspective"):
    """Create image pair with perspective transformation."""
    print(f"\nCreating PERSPECTIVE pair (strength={strength})")
    
    h, w = img.shape[:2]
    img1 = img.copy()
    
    offset = int(w * strength)
    
    src_points = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    
    dst_points = np.float32([
        [offset, 0], [w - offset, 0], [w, h], [0, h]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    img2 = cv2.warpPerspective(img, M, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_noise_pair(img, noise_level=25, name="noise"):
    """Create image pair with Gaussian noise."""
    print(f"\nCreating NOISE pair (level={noise_level})")
    
    img1 = img.copy()
    
    noise = np.random.normal(0, noise_level, img.shape)
    img2 = img.astype(np.float32) + noise
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    save_pair(img1, img2, name)
    return img1, img2


def create_blur_pair(img, kernel_size=7, name="blur"):
    """Create image pair with blur."""
    print(f"\nCreating BLUR pair (kernel={kernel_size}x{kernel_size})")
    
    img1 = img.copy()
    img2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    save_pair(img1, img2, name)
    return img1, img2


def create_crop_pair(img, crop_percentage=0.3, name="crop"):
    """Create image pair with cropped view."""
    print(f"\nCreating CROP pair (crop={crop_percentage*100}%)")
    
    h, w = img.shape[:2]
    img1 = img.copy()
    
    crop_h = int(h * (1 - crop_percentage))
    crop_w = int(w * (1 - crop_percentage))
    
    start_y = (h - crop_h) // 2
    start_x = int(w * 0.3)
    
    cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    img2 = cv2.resize(cropped, (w, h))
    
    save_pair(img1, img2, name)
    return img1, img2


def create_all_pairs_from_image(input_image_path, output_dir="images"):
    """Create all test pairs from a single input image."""
    print("\nCREATING TEST IMAGE PAIRS FROM SAMPLE IMAGE")
    print(f"Input image: {input_image_path}")
    print(f"Output directory: {output_dir}")
    
    create_output_dir(output_dir)
    
    print("\nLoading sample image...")
    img = load_sample_image(input_image_path)
    print(f"Loaded image: {img.shape}")
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_gray = img
    
    transformations = [
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
    
    for func, image, args, kwargs in transformations:
        try:
            func(image, *args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
    
    print("\nALL TEST PAIRS CREATED SUCCESSFULLY!")
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


def create_pairs_from_multiple_images(image_paths, output_dir="images"):
    """Create test pairs from multiple input images."""
    create_output_dir(output_dir)
    
    for i, path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {path}")
        
        try:
            img = load_sample_image(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            create_translation_pair(img, name=f"{base_name}_translation")
            create_rotation_pair(img, name=f"{base_name}_rotation")
            create_scale_pair(img, name=f"{base_name}_scale")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    print(f"\nProcessed {len(image_paths)} images")


def main():
    """Main function with command-line interface."""
    print("\nTEST IMAGE PAIR GENERATOR FOR SIFT/HARRIS ASSIGNMENT")
    
    if len(sys.argv) < 2:
        print("\nUSAGE:")
        print("  python create_test_pairs.py <input_image_path> [output_dir]")
        print("\nEXAMPLES:")
        print("  python create_test_pairs.py sample.jpg")
        print("  python create_test_pairs.py sample.jpg test_images")
        print("  python create_test_pairs.py images/building.jpg images/pairs")
        
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
            print(f"\nFound sample image: {found_sample}")
            response = input("Use this image? (y/n): ").strip().lower()
            if response == 'y':
                create_all_pairs_from_image(found_sample)
                return
        
        print("\nNo input image provided and no sample image found.")
        print("Please provide an image path as argument.")
        return
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "images"
    
    if not os.path.exists(input_path):
        print(f"\nERROR: Input image not found: {input_path}")
        return
    
    create_all_pairs_from_image(input_path, output_dir)


if __name__ == "__main__":
    main()
# Skin Color Detection

## 📂 File Structure
* **`croped/`**: Contains cropped skin samples used to train the Gaussian model.
* **`image/`**: Contains the target images for detection.
* **`skin_color_detection`**: The main script/notebook implementing the algorithm.

## 🧠 Logic
1. **Color Space**: Converts images to **YCrCb** and isolates **Cr** and **Cb** channels (ignoring luminance/lighting).
2. **Training**: Computes the **Mean Vector** and **Covariance Matrix** from the `croped` sample to model skin distribution.
3. **Detection**: Calculates the **Mahalanobis Distance** for every pixel in the main image against the skin model.
4. **Masking**: Classifies pixels with a distance `< 3.0` as skin and applies morphological closing to fill gaps.

### Usage
Update `MAIN_IMAGE_PATH` and `CROP_IMAGE_PATH` in the script and run to generate the mask.
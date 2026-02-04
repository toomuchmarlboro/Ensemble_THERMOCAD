import numpy as np
import cv2
import os
from skimage.segmentation import watershed
from skimage.filters import sobel
from scipy import ndimage as ndi

def load_and_standardize(path):
    """
    Loads image based on format (TIFF vs JPG).
    Returns:
        norm_img_u8: 0-255 image for Segmentation (Visual)
        raw_data: The original data (Float for TIFF, Int for JPG) for Physics extraction
    """
    # CASE 1: DMR-IR (Radiometric TIFF)
    if path.lower().endswith(('.tif', '.tiff')):
        # Load RAW data (Temperature in Celsius)
        raw_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if raw_data is None: return None, None
        
        # DYNAMIC NORMALIZATION (Do not assume fixed range)
        # We normalize based on THIS specific patient's min/max temp.
        # This preserves the full contrast for the segmentation algorithm.
        t_min = np.nanmin(raw_data)
        t_max = np.nanmax(raw_data)
        
        # Avoid division by zero if image is flat
        if t_max - t_min == 0:
            norm_img = np.zeros_like(raw_data, dtype=np.uint8)
        else:
            norm_img = (raw_data - t_min) / (t_max - t_min)
            norm_img = (norm_img * 255).astype(np.uint8)
            
        return norm_img, raw_data

    # CASE 2: Rodriguez / HIKMICRO (Standard JPG)
    else:
        # Load Visual data (0-255)
        raw_data = cv2.imread(path)
        if raw_data is None: return None, None
        
        # Convert to Grayscale
        norm_img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
        
        return norm_img, norm_img

def run_resolution_agnostic_segmentation():
    # --- CONFIGURATION ---
    input_root = r"organized_data" 
    output_root = r"organized_data_cleaned"
    
    print(f"--- Starting Resolution-Agnostic Processing ---")
    
    processed_count = 0
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
                continue
            
            img_path = os.path.join(root, file)
            
            # 1. Smart Load (Preserves Exact Temp for TIFF)
            image_visual, image_raw = load_and_standardize(img_path)
            if image_visual is None:
                print(f"Skipping corrupt file: {file}")
                continue

            # 2. Get Dynamic Resolution (Tackles 640x480 vs 320x240)
            h, w = image_visual.shape 

            # 3. Generate Gradient Map
            elevation_map = sobel(image_visual)

            # 4. Define Resolution-Independent Markers
            markers = np.zeros_like(image_visual)
            
            # --- BACKGROUND SEEDS (Label 1) ---
            # Bottom Corners are always safe in breast thermography
            markers[-1, 0] = 1
            markers[-1, -1] = 1
            
            # DYNAMIC NECK BARRIER: Always Top 15%
            # For DMR (480px height) -> Blocks top 72px
            # For RG  (240px height) -> Blocks top 36px
            neck_limit = int(h * 0.000) 
            markers[0:neck_limit, :] = 1
            
            # --- FOREGROUND SEED (Label 2) ---
            # Body Center (Targeting the sternum area)
            center_y, center_x = int(h * 0.6), w // 2
            
            # Dynamic Radius (5% of width) ensures seed fits in both resolutions
            radius_x = max(1, int(w * 0.1)) #original is 0.05
            radius_y = max(1, int(h * 0.1)) #original is 0.05
            
            # Safety bounds check
            y1 = max(0, center_y - radius_y)
            y2 = min(h, center_y + radius_y)
            x1 = max(0, center_x - radius_x)
            x2 = min(w, center_x + radius_x)
            
            markers[y1:y2, x1:x2] = 2

            # 5. Run Watershed
            segmentation = watershed(elevation_map, markers)
            segmentation = ndi.binary_fill_holes(segmentation - 1)
            mask = segmentation.astype(np.uint8) * 255 

            # 6. Apply Mask & Save
            # Note: We save the VISUAL representation (PNG) for the CNN training.
            # If you need the RAW TEMP data for analysis, you would save 'image_raw' separately.
            final_img = cv2.bitwise_and(image_visual, image_visual, mask=mask)
            
            # Recreate folder structure
            relative_path = os.path.relpath(root, input_root)
            save_dir = os.path.join(output_root, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # Always save as PNG to prevent JPG compression artifacts
            save_path = os.path.join(save_dir, file.rsplit('.', 1)[0] + ".png")
            
            cv2.imwrite(save_path, final_img)
            processed_count += 1
            print(f"Processed: {file} | Size: {w}x{h}", end='\r')

    print(f"\n\n--- Done! Processed {processed_count} images. ---")

if __name__ == "__main__":
    run_resolution_agnostic_segmentation()
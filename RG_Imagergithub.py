import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.color import rgb2gray
from scipy import ndimage as ndi

def run_imager_watershed():
    # --- CONFIGURATION ---
    # Use one of your sorted folders
    input_folder = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data\anterior\benign"
    
    # Get list of images
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    if not image_paths:
        print("No images found.")
        return

    # Select a random image to test
    img_path = image_paths[0] 
    print(f"Testing Watershed on: {os.path.basename(img_path)}")

    # 1. Load Image
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Generate Elevation Map (The "Edges" from the tutorial)
    # The tutorial uses gradient magnitude. Sobel is the standard way to get this.
    elevation_map = sobel(image_gray)

    # 3. Generate Automated Seeds (The "Markers")
    # Instead of clicking, we generate them mathematically.
    markers = np.zeros_like(image_gray)
    
    # Background Marker: The top-left corner (and border)
    #Background: bottom-left and bottom-right 
    markers[-1, 0] = 1
    markers[-1, -1] = 1
    
    # Foreground Marker: The center pixel (and a small circle around it)
    h, w = image_gray.shape
    center_y, center_x = h // 2, w // 2
    markers[center_y-10:center_y+10, center_x-10:center_x+10] = 2 # Label 2 = Body

    # 4. Run Watershed (The "Flood")
    # It spreads label 1 and 2 until they meet at the "Mountains" (Elevation Map)
    segmentation = watershed(elevation_map, markers)
    
    # 5. Extract Result
    # Fill holes to make it solid
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    
    # 6. Apply to Image (Physics Masking)
    mask = segmentation.astype(np.uint8)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes.ravel()

    ax[0].imshow(image_gray, cmap='gray')
    ax[0].set_title("1. Original")

    ax[1].imshow(elevation_map, cmap='jet')
    ax[1].set_title("2. Edges (The Mountains)")
    
    ax[2].imshow(masked_img)
    ax[2].set_title("3. Watershed Result")

    plt.show()

if __name__ == "__main__":
    run_imager_watershed()
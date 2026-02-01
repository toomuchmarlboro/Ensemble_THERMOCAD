import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def visualize_grabcut_segmentation():
    # --- CONFIGURATION ---
    # Point this to your Curated 'Anterior' or 'Lateral' folder
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data"
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Path not found: {dataset_path}")
        return

    # Get images
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("No images found.")
        return

    # Select 2 Random Images for testing
    random_indices = random.sample(range(len(image_paths)), 2)
    
    # --- LAYOUT ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    fig.suptitle(f"GrabCut Segmentation: Iterative Energy Minimization", fontsize=20, fontweight='bold', y=0.96)

    cols = ['Original Input', 'GrabCut Mask (Energy Min.)', 'Physics Gradient']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    print(f"--- Processing {len(random_indices)} Images with GrabCut ---")

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        
        # A. Load
        # GrabCut needs a 3-channel image, even if it's grayscale conceptually
        img_bgr = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        norm_img = img_gray.astype(np.float32) / 255.0
        
        # B. GRABCUT LOGIC
        # 1. Define the Bounding Box (ROI)
        # We assume the patient is in the middle 90% of the image.
        # Everything OUTSIDE this box is considered "Definitely Background"
        h, w = img_bgr.shape[:2]
        margin_w = int(w * 0.025) # 2.5% margin
        margin_h = int(h * 0.025) # 2.5% margin
        rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
        
        # 2. Internal masks for GrabCut
        mask = np.zeros(img_bgr.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64) # Internal memory for Background GMM
        fgdModel = np.zeros((1,65), np.float64) # Internal memory for Foreground GMM
        
        # 3. Run GrabCut (5 Iterations)
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 4. Interpret Result
        # Mask values: 0=Def BG, 1=Def FG, 2=Prob BG, 3=Prob FG
        # We take all (1 and 3) as the body.
        final_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        
        # 5. Cleanup (Optional Morphological Closing)
        # GrabCut is usually clean, but small holes might appear
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask * 255, cv2.MORPH_CLOSE, kernel)
        
        # Keep Largest Contour only (Safety)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(final_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
        tissue_mask_bool = clean_mask > 0
        
        # C. Physics Feature
        grad_y, grad_x = np.gradient(norm_img)
        heat_flux = np.sqrt(grad_x**2 + grad_y**2)
        heat_flux_masked = heat_flux * tissue_mask_bool
        
        # D. PLOTTING
        # Col 1: Original
        axes[i, 0].imshow(norm_img, cmap='inferno')
        axes[i, 0].axis('off')
        
        # Col 2: GrabCut Mask
        axes[i, 1].imshow(tissue_mask_bool, cmap='gray')
        axes[i, 1].set_title("Iterative Graph Cut")
        axes[i, 1].axis('off')

        # Col 3: Physics
        im3 = axes[i, 3-1].imshow(heat_flux_masked, cmap='jet') # Index corrected
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046).set_label('Gradient')

    print("Displaying GrabCut results...")
    plt.show()

if __name__ == "__main__":
    visualize_grabcut_segmentation()
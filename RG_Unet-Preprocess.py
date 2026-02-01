import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from rembg import remove

def visualize_u2net_segmentation():
    # --- CONFIGURATION ---
    # Point to your 'Anterior' or 'Lateral' folder
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data"
    
    if not os.path.exists(dataset_path):
        print("Path not found.")
        return

    # Get images
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Select 2 Random Images
    random_indices = random.sample(range(len(image_paths)), 2)
    
    # --- LAYOUT ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    fig.suptitle(f"U²-Net Segmentation: Structural Salient Object Detection", fontsize=20, fontweight='bold', y=0.96)

    cols = ['Original Input', 'U²-Net Mask (AI)', 'Physics Gradient']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    print(f"--- Processing {len(random_indices)} Images with U²-Net ---")

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        
        # A. Load Image (OpenCV loads as BGR)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # rembg expects RGB
        
        # B. THE AI SEGMENTATION (One line of code)
        # 'remove' returns the image with background as transparent (Alpha channel)
        # We perform post_process_mask=True to smooth edges
        result_rgba = remove(img_rgb, alpha_matting=True, alpha_matting_foreground_threshold=240)
        
        # C. Extract the Mask from the Alpha Channel
        # The 4th channel (Alpha) is our binary mask
        alpha_channel = result_rgba[:, :, 3]
        
        # Normalize mask to 0-1 bool
        tissue_mask_bool = alpha_channel > 10
        
        # D. Physics Feature Calculation
        # Convert original to float norm for physics
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        norm_img = img_gray.astype(np.float32) / 255.0
        
        grad_y, grad_x = np.gradient(norm_img)
        heat_flux = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply the AI Mask to the Physics
        heat_flux_masked = heat_flux * tissue_mask_bool
        
        # E. PLOTTING
        # Col 1: Original
        axes[i, 0].imshow(norm_img, cmap='inferno')
        axes[i, 0].axis('off')
        
        # Col 2: U2-Net Mask
        axes[i, 1].imshow(tissue_mask_bool, cmap='gray')
        axes[i, 1].set_title("Deep Learned ROI")
        axes[i, 1].axis('off')

        # Col 3: Physics
        im3 = axes[i, 2].imshow(heat_flux_masked, cmap='jet')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046).set_label('Gradient')

    print("Displaying U²-Net results...")
    plt.show()

if __name__ == "__main__":
    visualize_u2net_segmentation()
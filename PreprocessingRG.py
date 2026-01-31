import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def analyze_rodriguez_pure_physics():
    # --- 1. Configuration ---
    # UPDATE THIS PATH to your local Rodriguez dataset folder
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\data"
    
    print(f"--- 1. Searching for images in: {dataset_path} ---")
    
    image_paths = []
    labels = []
    
    # Collect images
    for class_name in ['Benign', 'Malignant']:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(class_name)
    
    if not image_paths:
        print("ERROR: No images found! Please check the path.")
        return

    # Select 2 Random Images
    total_images = len(image_paths)
    random_indices = random.sample(range(total_images), 2)
    
    # --- 2. Layout Setup ---
    # 2 Rows, 2 Columns (Simplified as requested)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f"Pure Physics Analysis (No Temperature Assumptions)", fontsize=18, fontweight='bold', y=0.96)

    cols = ['Normalized Radiometric Input (0-1)', 'Physics Gradient (Structural Flow)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    print(f"--- 2. Processing {len(random_indices)} Images ---")

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        label_str = labels[idx].upper()
        
        # --- A. Load & Normalize (No Temp Scaling) ---
        # 1. Read Raw Grayscale (0-255)
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Normalize to 0.0 - 1.0
        # This preserves the RELATIVE heat distribution exactly.
        # 0.0 = Coldest point in image, 1.0 = Hottest point in image.
        norm_img = raw_img / 255.0
        
        # --- B. Physics Gradient ---
        # We calculate the change in intensity. 
        # Since input is 0-1, the gradient represents "Percent Heat Change per Pixel"
        grad_y, grad_x = np.gradient(norm_img)
        heat_flux = np.sqrt(grad_x**2 + grad_y**2)
        
        # --- C. Plotting ---
        
        # Label
        axes[i, 0].set_ylabel(f"Patient {idx}\n{label_str}", 
                              color='red' if 'MALIGNANT' in label_str else 'orange',
                              fontsize=14, fontweight='bold', labelpad=20)

        # Col 1: Normalized Input
        im1 = axes[i, 0].imshow(norm_img, cmap='gray') # Using gray to show raw input clearly
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04).set_label('Normalized Intensity')

        # Col 2: Physics Gradient
        # We use 'jet' here to highlight the high-gradient regions (Vessels/Edges)
        im2 = axes[i, 1].imshow(heat_flux, cmap='jet')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04).set_label('Gradient Magnitude')

    print("Displaying pure physics window...")
    plt.show()

if __name__ == "__main__":
    analyze_rodriguez_pure_physics()
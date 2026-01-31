import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def visualize_kmeans_segmentation():
    # --- 1. Configuration ---
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\data"
    
    image_paths = []
    labels = []
    for class_name in ['Benign', 'Malignant']:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(class_name)
    
    if not image_paths: return

    random_indices = random.sample(range(len(image_paths)), 2)
    
    # --- 2. Layout ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    fig.suptitle(f"Rodriguez Dataset: K-Means Clustering Segmentation", fontsize=20, fontweight='bold', y=0.96)

    cols = ['Radiometric Input', 'K-Means Mask (Clustering)', 'Metabolic Feature']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    print(f"--- Processing {len(random_indices)} Images with K-Means ---")

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        label_str = labels[idx].upper()
        
        # A. Load
        raw_img_u8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        norm_img = raw_img_u8.astype(np.float32) / 255.0
        
        # B. K-Means Segmentation (The Nuclear Option)
        # 1. Flatten the image to a 1D array of pixels
        pixel_values = raw_img_u8.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        # 2. Define criteria (Stop when epsilon <= 1.0 or 10 iterations)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # 3. Run K-Means with K=2 (Background vs Body)
        k = 2
        _, labels_arr, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 4. Convert back to image shape
        centers = np.uint8(centers)
        segmented_data = labels_arr.reshape(raw_img_u8.shape)
        
        # 5. Determine which cluster is the body
        # The body is always the HOTTER (Brighter) cluster.
        # centers[0] is mean brightness of cluster 0, centers[1] is cluster 1.
        if centers[1] > centers[0]:
            body_cluster_idx = 1
        else:
            body_cluster_idx = 0
            
        # Create mask: 1 where pixel belongs to body cluster, 0 otherwise
        kmeans_mask = (segmented_data == body_cluster_idx).astype(np.uint8) * 255
        
        # 6. Cleanup (Morphological Closing to fill small holes inside body)
        kernel = np.ones((5,5), np.uint8)
        kmeans_mask = cv2.morphologyEx(kmeans_mask, cv2.MORPH_CLOSE, kernel)
        
        # 7. Safety Check: If mask is empty, fallback to full image
        if np.sum(kmeans_mask) < 100: 
            kmeans_mask = np.ones_like(kmeans_mask) * 255
            
        tissue_mask_bool = kmeans_mask > 0
        
        # C. Physics Feature
        grad_y, grad_x = np.gradient(norm_img)
        heat_flux = np.sqrt(grad_x**2 + grad_y**2)
        heat_flux_masked = heat_flux * tissue_mask_bool
        
        # D. Plotting
        axes[i, 0].set_ylabel(f"Patient {idx}\n{label_str}", 
                              color='red' if 'MALIGNANT' in label_str else 'orange',
                              fontsize=14, fontweight='bold', labelpad=20)

        # Col 1: Input
        im1 = axes[i, 0].imshow(norm_img, cmap='inferno')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04).set_label('Norm. Intensity')

        # Col 2: K-Means Mask
        axes[i, 1].imshow(tissue_mask_bool, cmap='gray')
        axes[i, 1].set_title(f"Cluster {body_cluster_idx} (Foreground)")
        axes[i, 1].axis('off')

        # Col 3: Physics
        im3 = axes[i, 2].imshow(heat_flux_masked, cmap='jet')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04).set_label('Gradient')

    print("Displaying K-Means Segmentation...")
    plt.show()

if __name__ == "__main__":
    visualize_kmeans_segmentation()
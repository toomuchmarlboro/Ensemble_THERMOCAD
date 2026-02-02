import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import skfuzzy as fuzz  # The C-Means Library
import random

def visualize_cmeans_segmentation():
    # --- CONFIGURATION ---
    # Use your reorganized/sorted folder if you want, or the raw one
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data"
    
    # We will pick random images to test
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("Dataset not found. Check path.")
        return

    # Select 2 Random Images
    random_indices = random.sample(range(len(image_paths)), 2)
    
    # --- LAYOUT ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    fig.suptitle(f"Fuzzy C-Means (FCM): Soft Clustering for Edge Preservation", fontsize=20, fontweight='bold', y=0.96)

    cols = ['Original Input', 'FCM Membership (Soft)', 'Final Hard Mask', 'Physics Gradient']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    print(f"--- Processing {len(random_indices)} Images with Fuzzy C-Means ---")

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        
        # A. Load & Preprocess
        raw_img_u8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        norm_img = raw_img_u8.astype(np.float32) / 255.0
        
        # B. FUZZY C-MEANS LOGIC (include neighbor differences via gradient)
        # Compute gradient magnitude (neighbor differences) and normalize
        grad_y, grad_x = np.gradient(norm_img)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        if grad_mag.max() > 0:
            grad_mag_n = grad_mag / grad_mag.max()
        else:
            grad_mag_n = grad_mag

        # 1. Build feature matrix: intensity + gradient magnitude (shape: 2, N_pixels)
        intensity = norm_img.ravel()
        features = np.vstack((intensity, grad_mag_n.ravel()))

        # 2. Run C-Means on combined features
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            features, c=2, m=10.0, error=0.020, maxiter=10000, init=None
        )

        print(f"FCM centers (intensity, grad): {cntr}, fpc: {fpc:.4f}")

        # 3. Determine which cluster is the BODY (use intensity component of centers)
        body_cluster_idx = np.argmax(cntr[:, 0])

        # 4. Extract Membership Map (The "Soft" Mask)
        # u is (2, N_pixels). We grab the row corresponding to the body.
        soft_membership = u[body_cluster_idx, :].reshape(raw_img_u8.shape)

        # 5. Create Binary Mask using Otsu on soft membership (adaptive)
        soft_u8 = (soft_membership * 255).astype('uint8')
        _, hard_mask = cv2.threshold(soft_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Cleanup (Morphological Closing)
        kernel = np.ones((2,2), np.uint8)
        hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_CLOSE, kernel)

        # Keep Largest Contour (Remove artifacts)
        contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(hard_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        tissue_mask_bool = clean_mask > 0

        # C. Physics Feature (Gradient)
        # Use previously-computed grad_mag as heat_flux for visualization
        heat_flux = grad_mag
        heat_flux_masked = heat_flux * tissue_mask_bool
        
        # D. PLOTTING
        # Col 1: Original
        axes[i, 0].imshow(norm_img, cmap='inferno')
        axes[i, 0].axis('off')
        
        # Col 2: Soft Membership (THE COOL PART)
        im2 = axes[i, 1].imshow(soft_membership, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title("Prob. of being Body")
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046).set_label('Probability')

        # Col 3: Hard Mask
        axes[i, 2].imshow(tissue_mask_bool, cmap='gray')
        axes[i, 2].set_title("Final ROI")
        axes[i, 2].axis('off')

        # Col 4: Physics
        im4 = axes[i, 3].imshow(heat_flux_masked, cmap='jet')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.046).set_label('Gradient')

    print("Displaying Fuzzy C-Means results...")
    plt.show()

if __name__ == "__main__":
    visualize_cmeans_segmentation()
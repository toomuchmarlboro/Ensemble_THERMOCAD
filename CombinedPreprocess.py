import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from datasets import load_dataset
from scipy.ndimage import gaussian_filter

def unified_physics_pipeline():
    print("--- 1. Loading Heterogeneous Datasets ---")
    
    # ==========================================
    # DATASET A: DMR-IR (High-Quality TIFF)
    # ==========================================
    print("-> Loading DMR-IR (Radiometric)...")
    ds_dmr = load_dataset("SemilleroCV/DMR-IR", split="train", streaming=False)
    dmr_sample = ds_dmr[random.randint(0, len(ds_dmr)-1)]
    
    # 1. Load Raw Data
    dmr_raw = np.array(dmr_sample['image'])
    
    # 2. Normalize (Universal Scale 0.0 - 1.0)
    dmr_norm = (dmr_raw - np.min(dmr_raw)) / (np.max(dmr_raw) - np.min(dmr_raw))
    
    # 3. Segmentation (Simple Threshold works for clean lab data)
    # We assume background is the coldest 15% of pixels
    dmr_mask = dmr_norm > 0.15 
    
    # 4. Physics Feature (Gradient)
    dy, dx = np.gradient(dmr_norm)
    dmr_grad = np.sqrt(dx**2 + dy**2) * dmr_mask

    # ==========================================
    # DATASET B: Rodriguez (Low-Quality JPG)
    # ==========================================
    print("-> Loading Rodriguez (Grayscale)...")
    rod_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\data"
    
    rod_files = []
    for root, _, files in os.walk(rod_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                rod_files.append(os.path.join(root, file))
    
    if not rod_files:
        print("Error: Rodriguez dataset path not found.")
        return

    rod_raw_u8 = cv2.imread(random.choice(rod_files), cv2.IMREAD_GRAYSCALE)
    
    # 1. Normalize (Universal Scale 0.0 - 1.0)
    rod_norm = rod_raw_u8.astype(np.float32) / 255.0
    
    # 2. Segmentation (K-Means Clustering for noisy backgrounds)
    # Flatten image
    pixels = rod_raw_u8.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Run K-Means (K=2)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Auto-detect 'Hot' Cluster (The Body)
    body_cluster = 1 if centers[1] > centers[0] else 0
    rod_mask_raw = (labels.reshape(rod_raw_u8.shape) == body_cluster).astype(np.uint8)
    
    # Cleanup holes
    rod_mask = cv2.morphologyEx(rod_mask_raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)) > 0
    
    # 3. Physics Feature (Gradient)
    rdy, rdx = np.gradient(rod_norm)
    rod_grad = np.sqrt(rdx**2 + rdy**2) * rod_mask

    # ==========================================
    # VISUALIZATION (Side-by-Side Comparison)
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    fig.suptitle("Unified Physics Pipeline: Domain Invariance Achievement", fontsize=20, fontweight='bold', y=0.96)

    cols = ['Universal Input (Normalized)', 'Adaptive Segmentation', 'Physics Feature (Gradient)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)

    # --- Row 1: DMR-IR ---
    axes[0, 0].set_ylabel("DMR-IR Dataset\n(Radiometric TIFF)", fontsize=14, fontweight='bold', labelpad=15)
    
    im1 = axes[0, 0].imshow(dmr_norm, cmap='inferno')
    axes[0, 0].axis('off'); plt.colorbar(im1, ax=axes[0, 0], fraction=0.046).set_label('Norm. Intensity')
    
    axes[0, 1].imshow(dmr_mask, cmap='gray')
    axes[0, 1].set_title("Method: Thresholding (Clean)")
    axes[0, 1].axis('off')
    
    im3 = axes[0, 2].imshow(dmr_grad, cmap='jet')
    axes[0, 2].axis('off'); plt.colorbar(im3, ax=axes[0, 2], fraction=0.046).set_label('Gradient Magnitude')

    # --- Row 2: Rodriguez ---
    axes[1, 0].set_ylabel("Rodriguez Dataset\n(Standard JPG)", fontsize=14, fontweight='bold', labelpad=15)
    
    im4 = axes[1, 0].imshow(rod_norm, cmap='inferno')
    axes[1, 0].axis('off'); plt.colorbar(im4, ax=axes[1, 0], fraction=0.046).set_label('Norm. Intensity')
    
    axes[1, 1].imshow(rod_mask, cmap='gray')
    axes[1, 1].set_title("Method: K-Means (Noisy)")
    axes[1, 1].axis('off')
    
    im6 = axes[1, 2].imshow(rod_grad, cmap='jet')
    axes[1, 2].axis('off'); plt.colorbar(im6, ax=axes[1, 2], fraction=0.046).set_label('Gradient Magnitude')

    print("Pipeline Complete. Displaying unified results...")
    plt.show()

if __name__ == "__main__":
    unified_physics_pipeline()
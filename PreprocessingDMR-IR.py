import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import random

def visualize_clean_physics_2_samples():
    print("--- 1. Loading Dataset ---")
    ds = load_dataset("SemilleroCV/DMR-IR", split="train", streaming=False)
    
    # Select 2 unique random indices
    total_images = len(ds)
    random_indices = random.sample(range(total_images), 2)
    
    # --- 2. Layout Setup ---
    # 2 Rows, 3 Columns. Adjusted figsize for a clean, non-cramped look.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Adjust spacing: More height between rows (hspace), more width between cols (wspace)
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f"Physics-Informed Analysis (Random 2 Patients)", fontsize=22, fontweight='bold', y=0.96)

    # Column Headers
    cols = ['Radiometric Heat Matrix (°C)', 'Tissue Mask (ROI)', 'Metabolic Proxy (Gradient)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=50)

    print(f"--- 3. Processing Patients: {random_indices} ---")
    
    for i, idx in enumerate(random_indices):
        sample = ds[idx]
        
        # Extract Data
        label_int = sample['label'] 
        label_str = "ABNORMAL" if label_int == 1 else "NORMAL"
        temp_matrix = np.array(sample['image']) # This IS the heat matrix
        
        # Physics Logic
        tissue_mask = temp_matrix > 26.0
        grad_y, grad_x = np.gradient(temp_matrix)
        heat_flux = np.sqrt(grad_x**2 + grad_y**2)
        heat_flux_masked = heat_flux * tissue_mask
        
        # --- Plotting ---
        
        # Row Label (Patient ID) on the Left
        axes[i, 0].set_ylabel(f"Patient {idx}\n{label_str}", 
                              color='red' if label_int == 1 else 'green',
                              fontsize=14, fontweight='bold', labelpad=20)

        # 1. Radiometric Heat Matrix (The Raw Data)
        im1 = axes[i, 0].imshow(temp_matrix, cmap='inferno')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        # Colorbar specific to this matrix
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar1.set_label('Temp (°C)', fontsize=10)
        cbar1.ax.tick_params(labelsize=10)

        # 2. ROI Mask
        axes[i, 1].imshow(tissue_mask, cmap='gray')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # 3. Metabolic Proxy (Physics Feature)
        im3 = axes[i, 2].imshow(heat_flux_masked, cmap='jet')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        # Colorbar
        cbar3 = plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar3.set_label('Gradient Magnitude', fontsize=10)
        cbar3.ax.tick_params(labelsize=10)

    print("Displaying clean window...")
    plt.show()

if __name__ == "__main__":
    visualize_clean_physics_2_samples()
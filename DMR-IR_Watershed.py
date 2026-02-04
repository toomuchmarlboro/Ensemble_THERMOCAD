import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset
from skimage.segmentation import watershed
from skimage.filters import sobel
from scipy import ndimage as ndi

def process_dmr_normal_abnormal_with_plot():
    # --- CONFIGURATION ---
    output_root = r"DatasetDMR-IR_Watershed"
    
    # 1. View Mapping (Strict Filtering: 0, 1, 3 only)
    view_map = {
        0: 'Anterior',  # Frontal
        1: 'Oblright',  # Right 45
        3: 'Oblleft'    # Left 45
    }
    
    # 2. LABEL MAPPING
    label_map = {
        0: 'Normal',    
        1: 'Abnormal'  
    }

    print("--- 1. Loading DMR-IR Dataset ---")
    try:
        ds = load_dataset("SemilleroCV/DMR-IR", split="train", streaming=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"--- 2. Processing {len(ds)} images into '{output_root}' ---")
    
    # UPDATED: Nested dictionary to track Class Balance
    counts = {
        'Anterior': {'Normal': 0, 'Abnormal': 0},
        'Oblright': {'Normal': 0, 'Abnormal': 0},
        'Oblleft':  {'Normal': 0, 'Abnormal': 0},
        'Skipped': 0
    }
    
    for i, sample in enumerate(ds):
        try:
            # --- A. METADATA EXTRACTION ---
            view_idx = sample['view']
            label_idx = sample['label']
            
            # Filter Views
            if view_idx not in view_map:
                counts['Skipped'] += 1
                continue 
            
            # Map Metadata to Folder Names
            folder_view = view_map[view_idx]
            folder_label = label_map[label_idx]
            
            # --- B. IMAGE NORMALIZATION ---
            raw_temp = np.array(sample['image'])
            
            # Dynamic Min-Max Scaling
            t_min = np.nanmin(raw_temp)
            t_max = np.nanmax(raw_temp)
            
            if t_max - t_min == 0:
                norm_img = np.zeros_like(raw_temp, dtype=np.uint8)
            else:
                norm_img = (raw_temp - t_min) / (t_max - t_min)
                norm_img = (norm_img * 255).astype(np.uint8)
            
            # --- C. SEGMENTATION ---
            h, w = norm_img.shape
            elevation_map = sobel(norm_img)
            markers = np.zeros_like(norm_img, dtype=int)
            
            # Background Seeds
            markers[-1, 0] = 1
            markers[-1, -1] = 1
            
            # Neck Barrier (kept at 0.00)
            neck_limit = int(h * 0.00)
            markers[0:neck_limit, :] = 1
            
            # Foreground Seed
            center_y, center_x = int(h * 0.6), w // 2
            rad_x = max(1, int(w * 0.05))
            rad_y = max(1, int(h * 0.05))
            
            y1 = max(0, center_y - rad_y)
            y2 = min(h, center_y + rad_y)
            x1 = max(0, center_x - rad_x)
            x2 = min(w, center_x + rad_x)
            markers[y1:y2, x1:x2] = 2
            
            # Watershed
            segmentation = watershed(elevation_map, markers)
            segmentation = ndi.binary_fill_holes(segmentation - 1)
            mask = segmentation.astype(np.uint8) * 255
            
            # Apply Mask
            final_img = cv2.bitwise_and(norm_img, norm_img, mask=mask)
            
            # --- D. SAVE ---
            save_dir = os.path.join(output_root, folder_view, folder_label)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"dmr_{i}_{folder_view}.png"
            save_path = os.path.join(save_dir, filename)
            
            cv2.imwrite(save_path, final_img)
            
            # Update Counts
            counts[folder_view][folder_label] += 1
            
            if i % 50 == 0: print(f"Processed {i}...", end='\r')

        except Exception as e:
            counts['Skipped'] += 1

    print("\n\n--- Processing Complete ---")
    print(f"Images saved to: {os.path.abspath(output_root)}")
    print(f"Skipped Files: {counts['Skipped']}")

    # --- E. PLOTTING DISTRIBUTION ---
    print("--- Generating Distribution Plot ---")
    
    views = ['Anterior', 'Oblleft', 'Oblright']
    normal_vals = [counts[v]['Normal'] for v in views]
    abnormal_vals = [counts[v]['Abnormal'] for v in views]

    x = np.arange(len(views))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bars
    rects1 = ax.bar(x - width/2, normal_vals, width, label='Normal', color='green', alpha=0.7)
    rects2 = ax.bar(x + width/2, abnormal_vals, width, label='Abnormal', color='red', alpha=0.7)

    # Add text labels, title, and custom x-axis tick labels
    ax.set_ylabel('Number of Images')
    ax.set_title('DMR-IR Dataset Distribution by View and Class')
    ax.set_xticks(x)
    ax.set_xticklabels(views)
    ax.legend()

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    
    # Save the plot so you can put it in your thesis!
    plot_path = os.path.join(output_root, "dataset_distribution.png")
    plt.savefig(plot_path)
    print(f"Distribution plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    process_dmr_normal_abnormal_with_plot()
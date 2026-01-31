import os
import shutil
from pathlib import Path

def reorganize_dataset_by_view_and_diagnosis():
    """
    Reorganizes the dataset from:
        data/Benign/IIR00XX/IIR00XX_anterior.jpg
        data/Malignant/IIR00XX/IIR00XX_oblleft.jpg
    
    To:
        organized_data/anterior/benign/IIR00XX_anterior.jpg
        organized_data/anterior/malignant/IIR00XX_anterior.jpg
        organized_data/oblleft/benign/IIR00XX_oblleft.jpg
        organized_data/oblleft/malignant/IIR00XX_oblleft.jpg
        organized_data/oblright/benign/IIR00XX_oblright.jpg
        organized_data/oblright/malignant/IIR00XX_oblright.jpg
    """
    
    # --- CONFIGURATION ---
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\data"
    output_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data"
    
    # View angle keywords
    views = {
        'anterior': ['_anterior'],
        'oblleft': ['_oblleft'],
        'oblright': ['_oblright'],
    }
    
    diagnosis_types = ['Benign', 'Malignant']
    
    # Create output directory structure
    print(f"Creating output directory structure at: {output_path}")
    
    for view in views.keys():
        for diagnosis in diagnosis_types:
            view_diagnosis_path = os.path.join(output_path, view, diagnosis.lower())
            os.makedirs(view_diagnosis_path, exist_ok=True)
            print(f"  Created: {view}/{diagnosis.lower()}")
    
    print(f"\n--- Starting Dataset Reorganization ---\n")
    
    # Counter for tracking
    total_copied = 0
    total_skipped = 0
    
    # Iterate through Benign and Malignant folders
    for diagnosis in diagnosis_types:
        diagnosis_path = os.path.join(dataset_path, diagnosis)
        
        if not os.path.exists(diagnosis_path):
            print(f"WARNING: {diagnosis_path} not found. Skipping...")
            continue
        
        print(f"\nProcessing {diagnosis} cases...")
        
        # Iterate through patient folders
        for patient_folder in os.listdir(diagnosis_path):
            patient_path = os.path.join(diagnosis_path, patient_folder)
            
            if not os.path.isdir(patient_path):
                continue
            
            # Iterate through images in patient folder
            for image_file in os.listdir(patient_path):
                if not image_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    continue
                
                image_path = os.path.join(patient_path, image_file)
                
                # Identify view type
                image_name_lower = image_file.lower()
                view_found = None
                
                for view, keywords in views.items():
                    if any(k in image_name_lower for k in keywords):
                        view_found = view
                        break
                
                if view_found is None:
                    print(f"  SKIPPED (view unknown): {image_file}")
                    total_skipped += 1
                    continue
                
                # Build destination path
                destination_dir = os.path.join(output_path, view_found, diagnosis.lower())
                destination_path = os.path.join(destination_dir, image_file)
                
                # Copy file
                try:
                    shutil.copy2(image_path, destination_path)
                    total_copied += 1
                    print(f"  Copied: {image_file} -> {view_found}/{diagnosis.lower()}/")
                except Exception as e:
                    print(f"  ERROR copying {image_file}: {e}")
                    total_skipped += 1
    
    # Summary
    print(f"\n--- Reorganization Complete ---")
    print(f"Total files copied: {total_copied}")
    print(f"Total files skipped: {total_skipped}")
    print(f"\nNew organized dataset is at: {output_path}")
    print("\nDirectory structure:")
    for view in views.keys():
        print(f"  {view}/")
        for diagnosis in diagnosis_types:
            print(f"    {diagnosis.lower()}/")

if __name__ == "__main__":
    reorganize_dataset_by_view_and_diagnosis()

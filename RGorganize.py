import os

def audit_dataset_angles():
    # --- CONFIGURATION ---
    # Path to your Rodriguez Dataset
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\data"
    
    # Keywords to detect angles in filenames
    # You may need to tweak these based on your specific filenames
    angles = {
        'Anterior (Front)': ['anterior'],
        'Oblique Left (45°)': ['_oblleft'],
        'Oblique Right (45°)': ['oblright'],
    }

    print(f"--- Auditing Dataset Structure: {dataset_path} ---")
    
    found_counts = {k: 0 for k in angles.keys()}
    found_counts['Unsure/Mixed'] = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            fname = file.lower()
            identified = False
            
            # Check against our angle dictionary
            for angle_name, keywords in angles.items():
                if any(k in fname for k in keywords):
                    found_counts[angle_name] += 1
                    identified = True
                    break
            
            if not identified:
                found_counts['Unsure/Mixed'] += 1
                # Optional: Print unsure filenames to debug
                # print(f"Unsure file: {file}")

    print("\n--- Audit Results ---")
    for angle, count in found_counts.items():
        print(f"{angle}: {count} images")
        
    if found_counts['Unsure/Mixed'] > 0:
        print("\nWARNING: Many images have unclear names. We may need to inspect the 'Unsure' ones manually.")
    else:
        print("\nSUCCESS: All images can be automatically separated.")

if __name__ == "__main__":
    audit_dataset_angles()
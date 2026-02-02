import os
import cv2
import numpy as np
from rembg import remove, new_session

def run_u2net_cleaner_fixed():
    # --- CONFIGURATION ---
    input_root = r"organized_data" 
    output_root = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\RemBG_TrialData"
    
    # Setup session (Using 'u2net' is standard/balanced)
    # We specify 'cpu' explicitly to stop it from hunting for missing CUDA drivers
    session = new_session("u2net", providers=['CPUExecutionProvider'])

    print(f"--- Starting U²-Net Background Removal (CPU Mode) ---")
    print(f"Input:  {os.path.abspath(input_root)}")
    print(f"Output: {os.path.abspath(output_root)}")

    success_count = 0
    error_count = 0

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img_path = os.path.join(root, file)
            
            # Recreate folder structure
            relative_path = os.path.relpath(root, input_root)
            save_dir = os.path.join(output_root, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save as PNG to keep quality
            save_path = os.path.join(save_dir, file.replace(".jpg", ".png"))
            
            try:
                # A. Load Image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {file}")
                    continue

                # B. Remove Background (The AI Step)
                # alpha_matting=True keeps the fuzzy edge for thermal accuracy
                output = remove(img, session=session, alpha_matting=True, alpha_matting_foreground_threshold=240)
                
                # C. Post-Process (Fixing the Alpha Blending Crash)
                # 1. Split channels
                b, g, r, a = cv2.split(output)
                
                # 2. Merge BGR to make foreground
                foreground = cv2.merge((b, g, r)).astype(float)
                
                # 3. Create Mask (Expand dimensions from (H,W) to (H,W,1))
                # This fixes the "Sizes do not match" error!
                alpha = a.astype(float) / 255.0
                alpha = np.expand_dims(alpha, axis=-1) 
                
                # 4. Create Black Background
                bg = np.zeros_like(foreground)
                
                # 5. Blend using NumPy (Robust Math)
                # Result = (Image * Mask) + (Black * Inverse_Mask)
                final_img = (foreground * alpha) + (bg * (1.0 - alpha))
                final_img = final_img.astype(np.uint8)
                
                # D. Save
                cv2.imwrite(save_path, final_img)
                success_count += 1
                
                # Simple progress indicator
                print(f"Processed: {file}", end='\r')
                
            except Exception as e:
                print(f"\nError processing {file}: {e}")
                error_count += 1

    print(f"\n\n--- Processing Complete ---")
    print(f"Successful: {success_count}")
    print(f"Failed:     {error_count}")
    print("Ready for CNN Training.")

if __name__ == "__main__":
    run_u2net_cleaner_fixed()
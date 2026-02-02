"""
FCM_GrabCut.py

Combine Fuzzy C-Means (intensity + gradient) with GrabCut refinement.
- Runs FCM on features [intensity, grad_mag]
- If FCM is unreliable (low fpc) or if `use_grabcut=True`, initialize GrabCut from FCM soft map
- Plot results for visual inspection

Usage: run directly. Adjust parameters in the call at bottom if needed.
"""

import os
import random
import numpy as np
import cv2
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def fcm_grabcut_segmentation(dataset_path: str,
                             num_images: int = 2,
                             use_grabcut_on_low_fpc: bool = True,
                             fpc_threshold: float = 0.75,
                             m: float = 2.0):
    # Collect image paths
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("No images found at:", dataset_path)
        return

    num_images = min(num_images, len(image_paths))
    chosen = random.sample(range(len(image_paths)), num_images)

    cols = ['Original', 'Soft Membership', 'FCM ROI', 'GrabCut ROI', 'Gradient']
    fig, axes = plt.subplots(num_images, len(cols), figsize=(5 * len(cols), 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(chosen):
        img_path = image_paths[idx]
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print("Failed to read", img_path)
            continue

        norm = raw.astype(np.float32) / 255.0

        # Gradient magnitude (neighbor differences)
        gy, gx = np.gradient(norm)
        grad = np.sqrt(gx**2 + gy**2)
        grad_n = grad / (grad.max() + 1e-9)

        # Build features (2 x N)
        features = np.vstack((norm.ravel(), grad_n.ravel()))

        # Fuzzy C-Means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            features, c=2, m=m, error=0.02, maxiter=5000, init=None
        )
        print(f"Image: {os.path.basename(img_path)}  |  centers: {cntr}  |  fpc: {fpc:.4f}")

        # Choose cluster whose intensity center is larger
        body_idx = np.argmax(cntr[:, 0])
        soft = u[body_idx, :].reshape(norm.shape)

        # FCM hard mask via Otsu on soft map
        soft_u8 = (soft * 255).astype('uint8')
        _, fcm_mask = cv2.threshold(soft_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fcm_mask = cv2.morphologyEx(fcm_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Prepare initial GrabCut mask from soft membership
        gc_init_mask = np.full(norm.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        gc_init_mask[soft > 0.85] = cv2.GC_FGD      # sure foreground
        gc_init_mask[soft < 0.15] = cv2.GC_BGD      # sure background
        gc_init_mask[(soft >= 0.6) & (soft <= 0.85)] = cv2.GC_PR_FGD
        gc_init_mask[(soft > 0.15) & (soft <= 0.4)] = cv2.GC_PR_BGD

        do_grabcut = (use_grabcut_on_low_fpc and fpc < fpc_threshold) or (not use_grabcut_on_low_fpc)
        grabcut_mask_final = np.zeros_like(fcm_mask)

        if do_grabcut:
            # GrabCut expects 3-channel image
            img_bgr = cv2.cvtColor((norm * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
            mask_gc = gc_init_mask.copy()
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(img_bgr, mask_gc, None, bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_MASK)
                grabcut_mask_final = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                grabcut_mask_final = cv2.morphologyEx(grabcut_mask_final, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            except Exception as e:
                print("GrabCut failed:", e)
                grabcut_mask_final = np.zeros_like(fcm_mask)

        # Keep largest contour for both masks to remove small artifacts
        def keep_largest(mask):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = np.zeros_like(mask)
            if cnts:
                lc = max(cnts, key=cv2.contourArea)
                cv2.drawContours(out, [lc], -1, 255, thickness=cv2.FILLED)
            return out

        fcm_clean = keep_largest(fcm_mask)
        gc_clean = keep_largest(grabcut_mask_final) if do_grabcut else np.zeros_like(fcm_mask)

        # Visualization
        axes[i, 0].imshow(norm, cmap='inferno')
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original')

        im1 = axes[i, 1].imshow(soft, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Soft membership')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046).set_label('Prob')

        axes[i, 2].imshow(fcm_clean, cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('FCM ROI')

        axes[i, 3].imshow(gc_clean, cmap='gray')
        axes[i, 3].axis('off')
        axes[i, 3].set_title('GrabCut ROI')

        im2 = axes[i, 4].imshow(grad * (fcm_clean > 0), cmap='jet')
        axes[i, 4].axis('off')
        axes[i, 4].set_title('Gradient (masked)')
        plt.colorbar(im2, ax=axes[i, 4], fraction=0.046).set_label('Grad')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data"
    fcm_grabcut_segmentation(dataset_path, num_images=2, use_grabcut_on_low_fpc=True, fpc_threshold=0.75)

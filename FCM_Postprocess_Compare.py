"""
FCM_Postprocess_Compare.py

Run Fuzzy C-Means (intensity + gradient) and compare post-processing methods for converting
FCM soft membership maps into closed ROI masks. This script implements and compares:

  A) Hysteresis thresholding + morphological closing + hole-fill + keep-largest-component
  C) Edge-guided fill (Canny on the fuzzy map -> dilate -> close -> fill)
  A+C) Union of A and C, followed by closing and largest-component filtering

Quick tweak guide (short):
  - FCM: change `m` (fuzziness, default 2.0), change `error` in `fuzz.cluster.cmeans` to control convergence
  - Hysteresis (A): change `t_low` (include weaker pixels) and `t_high` (sure-foreground threshold), `close_kernel` controls gap closing
  - Edge-fill (C): change `canny_sigma` (edge sensitivity), `dilate_kernel` (connect edges), `close_kernel` (closing size)
  - Combined (A+C): union helps fill gaps when one method misses; you can also try intersection if noisy

How to use:
  - Run directly: `python FCM_Postprocess_Compare.py` (defaults to sampling N images)
  - For experiments, call `fcm_postprocess_compare(dataset_path, num_images=5, m=2.0)` from a notebook or CLI wrapper.

Notes:
  - Monitor printed FCM `centers` and `fpc` (fuzzy partition coefficient). Low `fpc` indicates poor cluster separation.
  - If results are poor, try preprocessing (CLAHE, Gaussian blur) or add more features (x,y coords) to FCM.
"""

import os
import random
import argparse
import numpy as np
import cv2
import skfuzzy as fuzz
import matplotlib
# Allow a headless mode by setting environment var HEADLESS=1 before import
if os.environ.get('HEADLESS', '0') == '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def keep_largest_component(mask_uint8):
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_uint8)
    if cnts:
        lc = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [lc], -1, 255, thickness=cv2.FILLED)
    return out


def hysteresis_fill(soft, t_low=0.4, t_high=0.7, close_kernel=11):
    """Convert soft membership map into a closed mask using hysteresis + morphological ops.

    Args:
        soft (np.ndarray): Float map [0..1] from FCM (probability of being foreground).
        t_low (float): Lower threshold for 'possible' foreground. Decrease to include more weak pixels.
        t_high (float): Upper threshold for 'sure' foreground. Increase to be more conservative.
        close_kernel (int): Kernel size for morphological closing to bridge gaps; larger = stronger closing.

    Returns:
        np.ndarray: uint8 binary mask (0/255) representing the cleaned ROI.

    Tips to tweak:
        - If mask has holes, increase `t_high` slightly or increase `close_kernel`.
        - If mask leaks into background, raise `t_high` or increase `t_low` to be stricter.
        - For very noisy soft maps, consider applying Gaussian blur to `soft` before hysteresis.
    """
    # soft: float image [0..1]
    h, w = soft.shape
    sure_fg = (soft > t_high).astype('uint8')
    mask_allow = (soft > t_low).astype('uint8')

    # morphological reconstruction by dilation within mask_allow
    marker = sure_fg.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    prev = None
    for _ in range(500):
        dil = cv2.dilate(marker, kernel)
        dil = dil & mask_allow
        if np.array_equal(dil, marker):
            break
        marker = dil

    out = marker.copy().astype('uint8') * 255
    # Close gaps and fill holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    # Fill holes using contour fill
    out = keep_largest_component(out)
    return out


def edge_guided_fill(soft, canny_sigma=0.33, dilate_kernel=5, close_kernel=15):
    """Use edges detected on the soft map to create a closed ROI.

    Args:
        soft (np.ndarray): Float map [0..1] from FCM.
        canny_sigma (float): Scale for Canny thresholds around the median intensity of the soft map.
        dilate_kernel (int): Size of kernel to dilate edges so they connect better.
        close_kernel (int): Size of kernel to close dilated edges into a contour.

    Returns:
        np.ndarray: uint8 binary mask (0/255) representing the filled region.

    Tips:
        - If edges are too weak, lower `canny_sigma` or blur `soft` slightly to reduce noise.
        - If resulting closed contour is too small, decrease `dilate_kernel` or `close_kernel`.
        - If edges are fragmented, increasing `dilate_kernel` can help connect them before closing.
    """
    # Convert soft to uint8 for Canny
    soft_u8 = (soft * 255).astype('uint8')
    # Automatic thresholds based on median (robust)
    v = np.median(soft_u8)
    lower = int(max(0, (1.0 - canny_sigma) * v))
    upper = int(min(255, (1.0 + canny_sigma) * v))

    edges = cv2.Canny(soft_u8, lower, upper)
    # Dilate edges to make them more likely to close
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    edges_d = cv2.dilate(edges, kernel_d, iterations=2)

    # Close edges and fill
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, k_close)

    # Sometimes edges are sparse — fall back to thresholded soft map if no contours
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(soft_u8)
    if cnts:
        lc = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [lc], -1, 255, thickness=cv2.FILLED)
        out = keep_largest_component(out)
    else:
        # fallback: threshold soft map and close
        fall = (soft > 0.5).astype('uint8') * 255
        out = cv2.morphologyEx(fall, cv2.MORPH_CLOSE, k_close)
        out = keep_largest_component(out)

    return out


def fcm_postprocess_compare(dataset_path: str,
                            num_images: int = 4,
                            m: float = 2.0,
                            t_low: float = 0.005,
                            t_high: float = 0.950,
                            hysteresis_close: int = 66,
                            canny_sigma: float = 0.80,
                            dilate_kernel: int = 15,
                            edge_close: int = 50,
                            combine_close: int = 50,
                            cmeans_error: float = 0.005,
                            cmeans_maxiter: int = 5000,
                            random_seed: int = None,
                            process_all: bool = False,
                            save_masks: bool = False,
                            out_dir: str = None,
                            masked_out: str = None):
    # Optionally set random seed for reproducible sampling
    if random_seed is not None:
        random.seed(random_seed)

    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("No images found at:", dataset_path)
        return

    # Choose images to process: all or a random sample
    if process_all:
        chosen = image_paths
    else:
        chosen = random.sample(image_paths, min(num_images, len(image_paths)))

    cols = ['Original', 'Soft Membership', 'Hysteresis (A)', 'Edge-fill (C)', 'Combined (A+C)', 'Overlay A', 'Overlay C', 'Overlay A+C']
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(5 * len(cols), 4 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_path in enumerate(chosen):
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print("Failed to read", img_path)
            continue
        norm = raw.astype(np.float32) / 255.0

        # features: intensity + grad magnitude
        gy, gx = np.gradient(norm)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad_n = grad / (grad.max() + 1e-9)
        features = np.vstack((norm.ravel(), grad_n.ravel()))

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            features, c=2, m=m, error=cmeans_error, maxiter=cmeans_maxiter, init=None
        )
        print(f"{os.path.basename(img_path)} | centers: {cntr} | fpc: {fpc:.4f}")

        body_idx = np.argmax(cntr[:, 0])
        soft = u[body_idx, :].reshape(norm.shape)

        # Method A: hysteresis (parameters controllable)
        mask_a = hysteresis_fill(soft, t_low=t_low, t_high=t_high, close_kernel=hysteresis_close)

        # Method C: edge-guided fill (parameters controllable)
        mask_c = edge_guided_fill(soft, canny_sigma=canny_sigma, dilate_kernel=dilate_kernel, close_kernel=edge_close)

        # Method A+C: union then refine
        def combine_masks(mask1, mask2, close_kernel=11):
            comb = cv2.bitwise_or(mask1, mask2)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k)
            comb = keep_largest_component(comb)
            return comb

        mask_ac = combine_masks(mask_a, mask_c, close_kernel=combine_close)

        # Optionally save masks to out_dir for later analysis. When processing the whole
        # dataset we preserve subfolder structure relative to dataset_path.
        if save_masks and out_dir:
            rel = os.path.relpath(img_path, dataset_path)
            rel_dir = os.path.dirname(rel)
            save_dir = os.path.join(out_dir, rel_dir) if rel_dir != '' else out_dir
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(save_dir, f"{base}_soft.png"), (soft*255).astype('uint8'))
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_a.png"), mask_a)
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_c.png"), mask_c)
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_ac.png"), mask_ac)

        # Optionally save masked original images (apply combined mask to the original)
        if save_masks and masked_out:
            rel = os.path.relpath(img_path, dataset_path)
            rel_dir = os.path.dirname(rel)
            save_masked_dir = os.path.join(masked_out, rel_dir) if rel_dir != '' else masked_out
            os.makedirs(save_masked_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(img_path))[0]
            # Mask original grayscale image
            masked_img = cv2.bitwise_and((raw).astype('uint8'), (raw).astype('uint8'), mask=mask_ac)
            cv2.imwrite(os.path.join(save_masked_dir, f"{base}_masked.png"), masked_img)

        # Overlays
        def overlay_mask(img_gray, mask, color=(1.0, 0, 0)):
            img_rgb = cv2.cvtColor((img_gray * 255).astype('uint8'), cv2.COLOR_GRAY2RGB) / 255.0
            overlay = img_rgb.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.3 + np.array(color) * 0.7)
            return overlay

        ov_a = overlay_mask(norm, mask_a)
        ov_c = overlay_mask(norm, mask_c, color=(0, 1.0, 0))
        ov_ac = overlay_mask(norm, mask_ac, color=(1.0, 0.5, 0))

        # Plotting
        axes[i, 0].imshow(norm, cmap='inferno')
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original')

        im1 = axes[i, 1].imshow(soft, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Soft membership')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046).set_label('Prob')

        axes[i, 2].imshow(mask_a, cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Hysteresis (A)')

        axes[i, 3].imshow(mask_c, cmap='gray')
        axes[i, 3].axis('off')
        axes[i, 3].set_title('Edge-fill (C)')

        axes[i, 4].imshow(mask_ac, cmap='gray')
        axes[i, 4].axis('off')
        axes[i, 4].set_title('Combined (A+C)')

        axes[i, 5].imshow(ov_a)
        axes[i, 5].axis('off')
        axes[i, 5].set_title('Overlay A (red)')

        axes[i, 6].imshow(ov_c)
        axes[i, 6].axis('off')
        axes[i, 6].set_title('Overlay C (green)')

        axes[i, 7].imshow(ov_ac)
        axes[i, 7].axis('off')
        axes[i, 7].set_title('Overlay A+C (orange)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCM postprocess comparison: hysteresis vs edge-fill')
    parser.add_argument('--dataset', '-d', default=r"C:\\Users\\LENOVO THINKPAD T14\\Documents\\PROPOSAL TA\\files\\Rodriguez-Guerrero Dataset\\Breast Thermography\\Preprocessing\\organized_data", help='Dataset root path')
    parser.add_argument('--num', '-n', type=int, default=4, help='Number of random images to sample')
    parser.add_argument('--m', type=float, default=2.0, help='FCM fuzziness parameter')
    parser.add_argument('--tlow', type=float, default=0.4, help='Hysteresis lower threshold')
    parser.add_argument('--thigh', type=float, default=0.75, help='Hysteresis upper threshold')
    parser.add_argument('--hclose', type=int, default=11, help='Hysteresis close kernel')
    parser.add_argument('--canny_sigma', type=float, default=0.33, help='Edge detection sigma')
    parser.add_argument('--dilate', type=int, default=5, help='Edge dilate kernel')
    parser.add_argument('--eclose', type=int, default=15, help='Edge close kernel')
    parser.add_argument('--aclose', type=int, default=11, help='Combined close kernel')
    parser.add_argument('--err', type=float, default=0.02, help='FCM error threshold')
    parser.add_argument('--maxiter', type=int, default=5000, help='FCM max iterations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible sampling')
    parser.add_argument('--all', action='store_true', dest='process_all', help='Process all images in the dataset')
    parser.add_argument('--save', action='store_true', help='Save masks to output directory')
    parser.add_argument('--out', type=str, default='FCM_results', help='Output directory to save masks')
    parser.add_argument('--masked_out', type=str, default=None, help='Directory to save masked images (applies combined mask)')
    args = parser.parse_args()

    fcm_postprocess_compare(
        args.dataset,
        num_images=args.num,
        m=args.m,
        t_low=args.tlow,
        t_high=args.thigh,
        hysteresis_close=args.hclose,
        canny_sigma=args.canny_sigma,
        dilate_kernel=args.dilate,
        edge_close=args.eclose,
        combine_close=args.aclose,
        cmeans_error=args.err,
        cmeans_maxiter=args.maxiter,
        random_seed=args.seed,
        process_all=args.process_all,
        save_masks=args.save,
        out_dir=args.out,
        masked_out=args.masked_out
    )

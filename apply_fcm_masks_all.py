"""
apply_fcm_masks_all.py

Run FCM postprocessing over the entire dataset and save masked images and masks
using the parameters you supply. This is a thin wrapper around
`fcm_postprocess_compare()` implemented in `FCM_Postprocess_Compare.py`.

Example:
  python apply_fcm_masks_all.py --dataset "<path/to/organized_data>" \
      --out "<path/to/save_masks>" --masked_out "<path/to/save_masked_images>" \
      --tlow 0.4 --thigh 0.75 --num 5 --m 2.0

Note: this will process ALL images when run (it calls with --all). Be patient — it
may take a few minutes depending on your parameters and dataset size.
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import os
from FCM_Postprocess_Compare import fcm_postprocess_compare


def main():
    parser = argparse.ArgumentParser(description='Batch apply FCM masks to entire dataset')
    parser.add_argument('--dataset', '-d', required=False, default=r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\organized_data", help='Dataset root path (organized_data)')
    parser.add_argument('--out', required=False, default=r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\FCM_results", help='Directory to save masks (soft, mask_a, mask_c, mask_ac)')
    parser.add_argument('--masked_out', required=False, default=r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\Preprocessing\RemBG_TrialData", help='Directory to save masked original images')

    # FCM & postprocess parameters (expose common ones)
    parser.add_argument('--m', type=float, default=2.0, help='FCM fuzziness coefficient')
    parser.add_argument('--num', type=int, default=5, help='(ignored) sample count; script processes all images')
    parser.add_argument('--tlow', type=float, default=0.4, help='Hysteresis lower threshold')
    parser.add_argument('--thigh', type=float, default=0.75, help='Hysteresis upper threshold')
    parser.add_argument('--hclose', type=int, default=11, help='Hysteresis closing kernel')
    parser.add_argument('--canny_sigma', type=float, default=0.33, help='Edge Canny sigma')
    parser.add_argument('--dilate', type=int, default=5, help='Edge dilate kernel')
    parser.add_argument('--eclose', type=int, default=15, help='Edge close kernel')
    parser.add_argument('--aclose', type=int, default=11, help='Combined close kernel')
    parser.add_argument('--err', type=float, default=0.02, help='FCM error threshold')
    parser.add_argument('--maxiter', type=int, default=5000, help='FCM max iterations')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed (not used since processing all)')

    args = parser.parse_args()

    # Confirm outputs
    print(f"Dataset: {args.dataset}")
    print(f"Saving masks to: {args.out}")
    print(f"Saving masked originals to: {args.masked_out}")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.masked_out, exist_ok=True)

    # Run processing on entire dataset
    fcm_postprocess_compare(
        args.dataset,
        num_images=0,  # ignored when process_all=True
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
        process_all=True,
        save_masks=True,
        out_dir=args.out,
        masked_out=args.masked_out
    )


if __name__ == '__main__':
    main()

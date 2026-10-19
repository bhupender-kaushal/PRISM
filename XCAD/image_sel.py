import os
import argparse
import numpy as np
import cv2
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate Randomized Frozen Cohort for XCAD/DRIVE Diffusion Dataset")
    parser.add_argument('--dataroot', type=str, default='/bhupender/datasets/Angiogram/XCAD/',
                        help="Root directory of the dataset (e.g., /path/to/XCAD)")
    parser.add_argument('--threshold', type=float, default=0.055, 
                        help="Minimum area ratio the ground truth must cover (0.15 = 15%, 0.20 = 20%)")
    parser.add_argument('--num_images', type=int, default=50, 
                        help="Number of images to randomly select")
    parser.add_argument('--output_file', type=str, default='img_sel.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    mask_dir = os.path.join(args.dataroot, 'test', 'masks')
    if not os.path.exists(mask_dir):
        print(f"FATAL: Masks directory not found at {mask_dir}")
        return

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.gif'))]
    print(f"Scanning {len(mask_files)} masks for topological density >= {args.threshold*100}%...")

    qualified_images = []

    for mask_name in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_name)
        
        # Match dataloader behavior (grayscale reading)
        mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_cv is None:
            continue
            
        # Binarize and calculate density
        binary_mask = (mask_cv > 0).astype(np.uint8)
        total_pixels = binary_mask.size
        foreground_pixels = np.sum(binary_mask)
        density_ratio = foreground_pixels / total_pixels
        
        if density_ratio >= args.threshold:
            # Strip extension to match the evaluation script's case_name
            base_name = os.path.splitext(mask_name)[0]
            qualified_images.append(base_name)

    if not qualified_images:
        print(f"FATAL: No images met the {args.threshold*100}% threshold. Vascular masks are thin; try lowering to 0.05 or 0.08.")
        return

    print(f"Successfully qualified {len(qualified_images)} images.")

    # Randomize to prevent sequential selection bias
    random.shuffle(qualified_images)
    selected_cohort = qualified_images[:args.num_images]

    if len(selected_cohort) < args.num_images:
        print(f"WARNING: Requested {args.num_images} images, but only {len(selected_cohort)} qualified. Proceeding with available.")

    with open(args.output_file, 'w') as f:
        for name in selected_cohort:
            f.write(f"{name}\n")

    print(f"Success! Cohort saved to {os.path.abspath(args.output_file)}")

if __name__ == "__main__":
    main()
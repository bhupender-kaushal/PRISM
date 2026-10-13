import os
import argparse
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate Randomized Frozen Cohort for ISIC Skin Lesion Dataset")
    parser.add_argument('--dataroot_val', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test',
                        help="Root directory containing 'images' and 'masks' subfolders")
    parser.add_argument('--threshold', type=float, default=0.15, 
                        help="Minimum area ratio the lesion must cover (0.15 = 15%)")
    parser.add_argument('--num_images', type=int, default=50, 
                        help="Number of images to randomly select")
    parser.add_argument('--output_file', type=str, default='img_sel.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    mask_dir = os.path.join(args.dataroot_val, 'masks')
    if not os.path.exists(mask_dir):
        print(f"FATAL: Masks directory not found at {mask_dir}")
        return

    # Grab all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Scanning {len(mask_files)} masks for topological density >= {args.threshold*100}%...")

    qualified_images = []

    for mask_name in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        
        # Binarize
        binary_mask = (mask_np > 0).astype(np.uint8)
        
        total_pixels = binary_mask.size
        lesion_pixels = np.sum(binary_mask)
        density_ratio = lesion_pixels / total_pixels
        
        if density_ratio >= args.threshold:
            # Save the exact base name (without extension) to match the MSDUNet dataloader output
            base_name = os.path.splitext(mask_name)[0].rsplit('_', 1)[0]
            qualified_images.append(base_name)

    if not qualified_images:
        print("FATAL: No images met the threshold. Lower the threshold.")
        return

    print(f"Successfully qualified {len(qualified_images)} images.")

    # Randomize to ensure a diverse distribution of lesion shapes and sizes
    random.shuffle(qualified_images)
    selected_cohort = qualified_images[:args.num_images]

    if len(selected_cohort) < args.num_images:
        print(f"WARNING: Only {len(selected_cohort)} qualified. Proceeding with available.")

    with open(args.output_file, 'w') as f:
        for name in selected_cohort:
            f.write(f"{name}\n")

    print(f"Success! Cohort saved to {os.path.abspath(args.output_file)}")

if __name__ == "__main__":
    main()
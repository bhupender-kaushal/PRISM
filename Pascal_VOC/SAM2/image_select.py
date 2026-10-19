import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate a Frozen Image Cohort based on GT Area for SAM Evaluation")
    
    # Hardcoded to your exact paths and class
    parser.add_argument('--data_dir', type=str, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final', 
                        help="Base path to the dataset")
    parser.add_argument('--target_class', type=str, default='person', 
                        help="Target class folder")
    parser.add_argument('--threshold', type=float, default=0.30, 
                        help="Minimum percentage of image area the GT must cover (0.30 = 30%)")
    parser.add_argument('--num_images', type=int, default=10, 
                        help="Number of top images to select for the frozen cohort")
    parser.add_argument('--output_file', type=str, default="img_sel.txt", 
                        help="Output text file containing the selected image names")
    args = parser.parse_args()

    # Construct the full paths to match your SAM scripts
    class_dir = os.path.join(args.data_dir, args.target_class)
    image_dir = os.path.join(class_dir, 'images')
    mask_dir = os.path.join(class_dir, 'masks')

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"FATAL: Could not find 'images' or 'masks' directory inside {class_dir}")
        return

    # Grab image filenames
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    qualified_candidates = []

    print(f"Scanning {len(image_filenames)} images in '{class_dir}'...")
    print(f"Looking for GT pixel density >= {args.threshold * 100}%...")

    for img_name in tqdm(image_filenames):
        # Match mask extension logic from your GenericSegmentationDataset loader
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        # Load mask and compute density
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        
        # Binarize mask (foreground > 0)
        binary_mask = (mask_np > 0).astype(np.uint8)
        
        total_pixels = binary_mask.size
        foreground_pixels = np.sum(binary_mask)
        density_ratio = foreground_pixels / total_pixels
        
        # Qualify if it meets the user threshold (>= 30%)
        if density_ratio >= args.threshold:
            qualified_candidates.append({
                'img_name': img_name,
                'density': density_ratio
            })

    # Sort by largest density to ensure the most robust foregrounds are selected first
    qualified_candidates = sorted(qualified_candidates, key=lambda x: x['density'], reverse=True)
    selected_cohort = qualified_candidates[:args.num_images]

    if len(selected_cohort) < args.num_images:
        print(f"\nWARNING: Only found {len(selected_cohort)} images meeting the {args.threshold*100}% threshold.")
        print("Consider lowering the threshold to 0.20 or 0.10 if you need exactly 50 images.")
    else:
        print(f"\nSuccessfully qualified {len(selected_cohort)} images.")

    # Write the frozen cohort to the output file
    with open(args.output_file, 'w') as f:
        for item in selected_cohort:
            f.write(f"{item['img_name']}\n")

    print(f"Frozen cohort saved to: {os.path.abspath(args.output_file)}")

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np
import h5py, sys
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate Randomized Frozen Cohort using Intra-Volume Relative Area")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--target_class_index", type=int, default=8)
    parser.add_argument("--area_threshold_percentage", type=float, default=0.90, 
                        help="Multiplier for the maximum slice area within a volume (0.90 = 90%)")
    parser.add_argument("--num_images", type=int, default=100, 
                        help="Number of images to randomly select from the qualified pool")
    parser.add_argument("--output_file", type=str, default="img_sel.txt")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducible shuffling")
    
    args = parser.parse_args()
    random.seed(args.seed)

    test_vol_file = os.path.join(args.list_dir, "test_vol.txt")
    if not os.path.exists(test_vol_file):
        print(f"FATAL: test_vol.txt not found at {test_vol_file}")
        sys.exit(1)

    with open(test_vol_file, 'r') as f:
        volume_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Scanning {len(volume_names)} volumes for Class {args.target_class_index} using relative area threshold >= {args.area_threshold_percentage*100}% of volume max...")
    
    qualified_slices = []

    for vol_name in tqdm(volume_names, desc="Scanning Volumes"):
        filepath = os.path.join(args.volume_path, f"{vol_name}.npy.h5")
        if not os.path.exists(filepath):
            continue
            
        with h5py.File(filepath, 'r') as data:
            labels = data['label'][:]
            num_slices = labels.shape[0]
            
            slices_in_volume = []
            
            # First pass: map the areas of the target class in all slices
            for slice_idx in range(num_slices):
                label_slice = labels[slice_idx, :, :]
                if np.any(label_slice == args.target_class_index):
                    area = np.sum(label_slice == args.target_class_index)
                    slices_in_volume.append({
                        'case_name': f"{vol_name}_slice_{slice_idx}",
                        'area': area
                    })
            
            # Second pass: qualify slices based on the maximum area in this specific volume
            if slices_in_volume:
                max_area_in_volume = max(s['area'] for s in slices_in_volume)
                area_threshold = max_area_in_volume * args.area_threshold_percentage
                
                for s in slices_in_volume:
                    if s['area'] >= area_threshold:
                        qualified_slices.append(s['case_name'])

    if not qualified_slices:
        print("FATAL: Found 0 qualified slices. Check your target_class_index and volume paths.")
        return

    print(f"\nSuccessfully qualified {len(qualified_slices)} slices across the dataset.")

    # Randomize to eliminate serial slice bias
    random.shuffle(qualified_slices)

    # Extract the target number of images
    selected_cohort = qualified_slices[:args.num_images]

    if len(selected_cohort) < args.num_images:
        print(f"WARNING: Requested {args.num_images} images, but only found {len(selected_cohort)} qualified ones.")

    # Save to text file
    with open(args.output_file, 'w') as f:
        for case_name in selected_cohort:
            f.write(f"{case_name}\n")

    print(f"Success! {len(selected_cohort)} randomized slice names saved to {os.path.abspath(args.output_file)}")

if __name__ == "__main__":
    main()
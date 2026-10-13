import os
import time
import argparse
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import plotly.express as px

from skimage import color
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid

def get_strict_context_maps(segments, gt_mask, ic_threshold=0.5, oc_threshold=0.0):
    """
    Applies the Strict Criterion: Evaluates WHOLE superpixels based on their overlap 
    with the Ground Truth mask.
    """
    unique_labels = np.unique(segments)
    gt_bool = (gt_mask > 0)
    
    in_context_map = np.zeros_like(segments)
    out_of_context_map = np.zeros_like(segments)
    
    for sp_id in unique_labels:
        if sp_id == 0: continue 
        
        sp_mask = (segments == sp_id)
        sp_area = np.sum(sp_mask)
        if sp_area == 0: continue
        
        overlap_ratio = np.sum(sp_mask & gt_bool) / sp_area
        
        if overlap_ratio >= ic_threshold:
            in_context_map[sp_mask] = sp_id
        elif overlap_ratio <= oc_threshold:
            out_of_context_map[sp_mask] = sp_id
            
    return in_context_map, out_of_context_map

def save_boundary_image(image, segments, save_path):
    """Overlays superpixel boundaries on the image and saves it."""
    if segments is not None:
        boundary_img = mark_boundaries(image, segments, color=(1, 1, 0), mode='thick')
        boundary_img_uint8 = (np.clip(boundary_img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(boundary_img_uint8).save(save_path)
    else:
        Image.fromarray((image * 255).astype(np.uint8)).save(save_path)

def run_superpixel_algorithms(image_rgb, target_segments=500):
    """Executes 4 superpixel algorithms and returns their segment maps and execution times."""
    results = {}
    
    # 1. SLIC
    start_time = time.perf_counter()
    labels_slic = slic(image_rgb, n_segments=target_segments, compactness=10, start_label=1)
    slic_time = time.perf_counter() - start_time
    results['SLIC'] = {'segments': labels_slic, 'time': slic_time}

    # 2. Felzenszwalb
    start_time = time.perf_counter()
    labels_felz = felzenszwalb(image_rgb, scale=100, sigma=0.5, min_size=50)
    felz_time = time.perf_counter() - start_time
    results['Felzenszwalb'] = {'segments': labels_felz, 'time': felz_time}

    # 3. Quickshift
    start_time = time.perf_counter()
    labels_quick = quickshift(image_rgb, kernel_size=3, max_dist=6, ratio=0.5)
    quick_time = time.perf_counter() - start_time
    results['Quickshift'] = {'segments': labels_quick, 'time': quick_time}

    # 4. Compact Watershed
    start_time = time.perf_counter()
    gradient = sobel(color.rgb2gray(image_rgb))
    grid = regular_grid(image_rgb.shape[:2], n_points=target_segments)
    markers = np.zeros(image_rgb.shape[:2], dtype=int)
    
    # Safely reshape the arange array to match the N-dimensional grid space
    target_shape = markers[grid].shape
    markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
    
    labels_water = watershed(gradient, markers, compactness=0.001)
    water_time = time.perf_counter() - start_time
    results['Watershed'] = {'segments': labels_water, 'time': water_time}

    return results
def find_synapse_slices(volume_path, target_class, num_images):
    """Scans the Synapse h5py volumes by auto-detecting all .h5 files in the directory."""
    from glob import glob
    
    # Auto-detect all .h5 files in the volume_path
    filepaths = glob(os.path.join(volume_path, "*.h5"))
    
    if not filepaths:
        print(f"FATAL: No .h5 files found in {volume_path}")
        print("Please check your --volume_path argument.")
        return []
        
    candidate_slices = []
    print(f"Auto-detected {len(filepaths)} volume files. Scanning for target class {target_class}...")
    
    files_found = 0
    for filepath in filepaths:
        # Extract base name (e.g., 'case0001' from 'case0001.npy.h5')
        vol_name = os.path.basename(filepath).split('.')[0] 
        files_found += 1
        
        try:
            with h5py.File(filepath, 'r') as data:
                image_vol = data['image'][:]
                label_vol = data['label'][:]
                
                for slice_idx in range(label_vol.shape[0]):
                    label_slice = label_vol[slice_idx]
                    if np.any(label_slice == target_class):
                        area = np.sum(label_slice == target_class)
                        candidate_slices.append({
                            'case_name': f"{vol_name}_slice_{slice_idx}",
                            'image': image_vol[slice_idx],
                            'label': label_slice,
                            'area': area
                        })
        except Exception as e:
            print(f"  [Error] Could not read {filepath}: {e}")

    print(f"\n--- SCAN SUMMARY ---")
    print(f"Volumes successfully scanned: {files_found}/{len(filepaths)}")
    print(f"Total slices found containing class {target_class}: {len(candidate_slices)}")
    print(f"--------------------\n")

    if not candidate_slices:
        if files_found > 0:
            print(f"ERROR: The volumes were opened successfully, but Class {target_class} does not exist in any of them.")
            print(f"Try changing --target_class_index to a different integer (e.g., 1, 2, or 3).")
        return []
                    
    # Sort by the largest area of the target class
    candidate_slices.sort(key=lambda x: x['area'], reverse=True)
    return candidate_slices[:num_images]

def main():
    parser = argparse.ArgumentParser(description="Superpixel Benchmarking for Synapse h5py Dataset")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--output_dir", type=str, default="./superpixel_benchmark")
    
    parser.add_argument("--num_images", type=int, default=20, help="Number of slices to process.")
    parser.add_argument("--target_class_index", type=int, default=8, help="Class index to use as Ground Truth.")
    
    # Strict Threshold Controls
    parser.add_argument('--ic_threshold', type=float, default=0.5, help="Min overlap percentage for In-Context.")
    parser.add_argument('--oc_threshold', type=float, default=0.0, help="Max overlap percentage for Out-of-Context.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract the best 2D slices from the 3D volumes
    # top_slices = find_synapse_slices(args.volume_path, args.list_dir, "test_vol", args.target_class_index, args.num_images)
    # Extract the best 2D slices from the 3D volumes via auto-detection
    top_slices = find_synapse_slices(args.volume_path, args.target_class_index, args.num_images)
    
    if not top_slices:
        print("No valid slices found containing the target class.")
        return

    performance_metrics = []

    for slice_data in top_slices:
        base_name = slice_data['case_name']
        raw_img = slice_data['image']
        raw_label = slice_data['label']

        print(f"\nProcessing Slice: {base_name} (Class Area: {slice_data['area']} px)")
        
        # Normalize the 2D medical image to [0, 1]
        img_01 = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        
        # Convert to RGB (required for Quickshift and Felzenszwalb to function properly)
        img_rgb = color.gray2rgb(img_01)
        
        # Create Binary Ground Truth mask for the specific target class
        gt_mask_bin = (raw_label == args.target_class_index).astype(np.uint8)

        img_output_dir = os.path.join(args.output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Run Algorithms targeting ~500 segments
        sp_results = run_superpixel_algorithms(img_rgb, target_segments=500)

        for algo_name, data in sp_results.items():
            segments = data['segments']
            exec_time = data['time']
            num_segments = len(np.unique(segments))

            print(f"  -> {algo_name}: {num_segments} segments generated in {exec_time:.4f}s")
            
            performance_metrics.append({
                "Image": base_name,
                "Algorithm": algo_name,
                "Time_Seconds": round(exec_time, 4),
                "Num_Segments": num_segments
            })

            # Strictly split context using area overlap thresholds
            in_context_map, out_of_context_map = get_strict_context_maps(
                segments, 
                gt_mask_bin, 
                ic_threshold=args.ic_threshold, 
                oc_threshold=args.oc_threshold
            )
            
            algo_out_dir = os.path.join(img_output_dir, algo_name.lower())
            os.makedirs(algo_out_dir, exist_ok=True)

            save_boundary_image(img_rgb, segments, os.path.join(algo_out_dir, f"{base_name}_all_superpixels.png"))
            save_boundary_image(img_rgb, in_context_map, os.path.join(algo_out_dir, f"{base_name}_in_context.png"))
            save_boundary_image(img_rgb, out_of_context_map, os.path.join(algo_out_dir, f"{base_name}_out_of_context.png"))
            save_boundary_image(img_rgb, None, os.path.join(algo_out_dir, f"{base_name}_original.png"))

    # Generate Interactive Plotly Chart
    if performance_metrics:
        df_perf = pd.DataFrame(performance_metrics)
        print("\nAverage Execution Times:")
        print(df_perf.groupby("Algorithm")["Time_Seconds"].mean().reset_index())

if __name__ == "__main__":
    main()
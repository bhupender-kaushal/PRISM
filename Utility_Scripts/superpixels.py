import os
import time
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
# import plotly.express as px

from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid

def get_strict_context_maps(segments, gt_mask, ic_threshold=0.5, oc_threshold=0.0):
    """
    Applies the Strict Criterion: Evaluates WHOLE superpixels based on their overlap 
    with the Ground Truth mask.
    - In-Context: >= 50% (default) of the superpixel area is inside the GT.
    - Out-of-Context: <= 0% (default) of the superpixel area is inside the GT.
    """
    unique_labels = np.unique(segments)
    gt_bool = (gt_mask > 0)
    
    in_context_map = np.zeros_like(segments)
    out_of_context_map = np.zeros_like(segments)
    
    for sp_id in unique_labels:
        if sp_id == 0: continue # Skip background if 0 is used
        
        sp_mask = (segments == sp_id)
        sp_area = np.sum(sp_mask)
        
        if sp_area == 0: continue
        
        # Calculate how much of this specific superpixel lies within the GT
        overlap_ratio = np.sum(sp_mask & gt_bool) / sp_area
        
        # Strictly assign the entire superpixel geometry to the correct map
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


def run_superpixel_algorithms(image, target_segments=500):
    """Executes 4 superpixel algorithms and returns their segment maps and execution times."""
    results = {}
    
    # 1. SLIC
    # SLIC natively accepts the target number of segments
    start_time = time.perf_counter()
    labels_slic = slic(image, n_segments=target_segments, compactness=20, start_label=1)
    slic_time = time.perf_counter() - start_time
    results['SLIC'] = {'segments': labels_slic, 'time': slic_time}

    # 2. Felzenszwalb
    # Note: Felzenszwalb uses 'scale' to determine segment size, not a hard segment count.
    start_time = time.perf_counter()
    labels_felz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    felz_time = time.perf_counter() - start_time
    results['Felzenszwalb'] = {'segments': labels_felz, 'time': felz_time}

    # 3. Quickshift
    # Note: Quickshift uses 'kernel_size' and 'max_dist', not a hard segment count.
    start_time = time.perf_counter()
    labels_quick = quickshift(image, kernel_size=3, max_dist=5, ratio=0.5)
    quick_time = time.perf_counter() - start_time
    results['Quickshift'] = {'segments': labels_quick, 'time': quick_time}

    # 4. Compact Watershed
    # Watershed uses a grid of markers, which we set using the target_segments
    start_time = time.perf_counter()
    gradient = sobel(color.rgb2gray(image))
    grid = regular_grid(image.shape[:2], n_points=target_segments)
    markers = np.zeros(image.shape[:2], dtype=int)

    target_shape = markers[grid].shape
    markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
    
    labels_water = watershed(gradient, markers, compactness=0.001)
    water_time = time.perf_counter() - start_time
    results['Watershed'] = {'segments': labels_water, 'time': water_time}

    return results

def main():
    parser = argparse.ArgumentParser(description="Superpixel Algorithm Benchmarking and Extraction")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final')
    parser.add_argument('--output_dir', type=str, default='./superpixel_benchmark1', help="Directory to save extracted maps.")
    parser.add_argument('--num_images', type=int, default=500, help="Number of images to process.")
    
    # Strict Threshold Controls
    parser.add_argument('--ic_threshold', type=float, default=1.0, help="Min overlap percentage to be considered In-Context.")
    parser.add_argument('--oc_threshold', type=float, default=0.0, help="Max overlap percentage to be considered Out-of-Context.")
    parser.add_argument('--target_class', type=str, default='person')
    parser.add_argument('--input_file', type=str, default='/bhupender/2025/Mouse_other/img_sel.txt', help="Optional file containing list of specific images to process.")
    
    args = parser.parse_args()



    sel_path = args.input_file if hasattr(args, 'input_file') else None
    selected_files = None
    if os.path.exists(sel_path):
        with open(sel_path, 'r') as f:
            selected_files = {ln.strip() for ln in f if ln.strip()}
 



    image_dir = os.path.join(args.data_dir,args.target_class, 'images') 
    mask_dir = os.path.join(args.data_dir,args.target_class, 'masks')

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("Error: data_dir must contain 'images' and 'masks' subdirectories.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = sorted(glob(os.path.join(image_dir, '*.*')))[:args.num_images]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    performance_metrics = []

    for img_path in image_paths:

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if selected_files is not None and base_name+'.jpg' not in selected_files:
            print(f"Skipping {base_name}: not in img_sel.txt selection.")
            continue


        mask_path = os.path.join(mask_dir, f"{base_name}.png") 

        if not os.path.exists(mask_path):
            print(f"Skipping {base_name}: Mask not found.")
            continue

        print(f"\nProcessing Image: {base_name}")
        
        # Load and normalize image to RGB [0, 1]
        raw_img = io.imread(img_path)
        if raw_img.ndim == 2:
            raw_img = color.gray2rgb(raw_img)
        elif raw_img.shape[2] == 4:
            raw_img = color.rgba2rgb(raw_img)
        
        img_01 = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        
        # Load mask
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        if gt_mask.shape != img_01.shape[:2]:
            gt_mask = np.array(Image.fromarray(gt_mask).resize(img_01.shape[1::-1], Image.NEAREST))
        gt_mask_bin = (gt_mask > 127).astype(np.uint8)

        img_output_dir = os.path.join(args.output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Run Algorithms
        sp_results = run_superpixel_algorithms(img_01)

        for algo_name, data in sp_results.items():
            segments = data['segments']
            exec_time = data['time']
            num_segments = len(np.unique(segments))

            print(f"  -> {algo_name}: {num_segments} segments generated in {exec_time:.4f}s")
            
            # Log metrics
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

            save_boundary_image(img_01, segments, os.path.join(algo_out_dir, f"{base_name}_all_superpixels.png"))
            save_boundary_image(img_01, in_context_map, os.path.join(algo_out_dir, f"{base_name}_in_context.png"))
            save_boundary_image(img_01, out_of_context_map, os.path.join(algo_out_dir, f"{base_name}_out_of_context.png"))
            save_boundary_image(img_01, None, os.path.join(algo_out_dir, f"{base_name}_original.png"))

    # Generate Interactive Plotly Chart
    if performance_metrics:
        df_perf = pd.DataFrame(performance_metrics)
        print("\nAverage Execution Times:")
        print(df_perf.groupby("Algorithm")["Time_Seconds"].mean().reset_index())

        print("\nGenerating Interactive Plot...")
        
        # Interactive grouped bar chart
        # fig = px.bar(
        #     df_perf, 
        #     x="Image", 
        #     y="Time_Seconds", 
        #     color="Algorithm", 
        #     barmode="group",
        #     hover_data=["Num_Segments"],
        #     title="Superpixel Algorithm Benchmarking: Execution Time",
        #     labels={"Time_Seconds": "Execution Time (Seconds)"},
        #     color_discrete_sequence=px.colors.qualitative.Prism
        # )
        

if __name__ == "__main__":
    main()
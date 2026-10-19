import os
import time
import argparse
import logging
import random
import numpy as np
import pandas as pd
# import plotly.express as px
from PIL import Image
from tqdm import tqdm

from skimage import color
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid
from skimage.color import rgb2gray, gray2rgb

import torch

# Direct imports for your actual diffusion modules
import data as Data
import core.logger as Logger

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
    gradient = sobel(rgb2gray(image_rgb))
    grid = regular_grid(image_rgb.shape[:2], n_points=target_segments)
    markers = np.zeros(image_rgb.shape[:2], dtype=int)
    
    # Safely reshape the arange array to match the N-dimensional grid space
    target_shape = markers[grid].shape
    markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
    
    labels_water = watershed(gradient, markers, compactness=0.001)
    water_time = time.perf_counter() - start_time
    results['Watershed'] = {'segments': labels_water, 'time': water_time}

    return results

def main():
    parser = argparse.ArgumentParser(description="Superpixel Benchmarking for Diffusion Dataloader")
    
    # Diffusion Specific Logger / Config Args
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/bhupender/datasets/Angiogram/XCAD/')
    parser.add_argument('-d', '--debug', action='store_true')

    # Benchmarking Specific Args
    parser.add_argument("--output_dir", type=str, default="./superpixel_benchmark")
    parser.add_argument("--num_images", type=int, default=20, help="Number of slices to process.")
    parser.add_argument("--num_superpixels", type=int, default=500)
    
    # Strict Threshold Controls
    parser.add_argument('--ic_threshold', type=float, default=0.5, help="Min overlap percentage for In-Context.")
    parser.add_argument('--oc_threshold', type=float, default=0.0, help="Max overlap percentage for Out-of-Context.")
    
    args = parser.parse_args()

    # Parse options exactly as done in your diffusion script
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir:
        if 'datasets' in opt and 'val' in opt['datasets']: opt['datasets']['val']['dataroot'] = args.data_dir
        elif 'datasets' in opt and 'test' in opt['datasets']: opt['datasets']['test']['dataroot'] = args.data_dir
        else: opt['datasets'] = {'test': {'name': 'xcad_test', 'mode': args.phase, 'dataroot': args.data_dir}}

    opt['phase'] = args.phase
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = logging.getLogger('base')
    logger.setLevel(logging.INFO)

    logger.info("Loading Diffusion Dataset...")
    dataset_opt = opt['datasets']['test']
    val_set = Data.create_dataset_xcad(dataset_opt, 'test')
    val_loader = Data.create_dataloader(val_set, dataset_opt, 'test')

    performance_metrics = []

    for idata, val_data in enumerate(tqdm(val_loader, desc="Processing Images")):
        if idata >= args.num_images: break
        
        # Extract name using the exact logic from the diffusion script
        case_name = os.path.basename(val_data['P'][0]).replace('.png', '')
        print(f"\nProcessing Image: {case_name}")
        
        # Safely extract and scale the [-1, 1] tensor back to a [0, 1] numpy array
        orig_image_np_raw = val_data['A'][0].cpu().numpy()
        if orig_image_np_raw.ndim == 3:
            orig_image_np = rgb2gray(orig_image_np_raw.transpose(1, 2, 0)) if orig_image_np_raw.shape[0] == 3 else orig_image_np_raw.squeeze()
        else:
            orig_image_np = orig_image_np_raw
            
        orig_image_01 = np.clip((orig_image_np + 1.0) / 2.0, 0.0, 1.0)
        img_rgb = gray2rgb(orig_image_01) if orig_image_01.ndim == 2 else orig_image_01

        # Safely extract and scale the GT mask tensor to binary uint8
        gt_mask_raw = val_data['F'][0].squeeze().cpu().numpy()
        gt_mask_bin = ((gt_mask_raw + 1.0) / 2.0 > 0.5).astype(np.uint8)

        img_output_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Run Algorithms targeting ~500 segments
        sp_results = run_superpixel_algorithms(img_rgb, target_segments=args.num_superpixels)

        for algo_name, data in sp_results.items():
            segments = data['segments']
            exec_time = data['time']
            num_segments = len(np.unique(segments))

            print(f"  -> {algo_name}: {num_segments} segments generated in {exec_time:.4f}s")
            
            performance_metrics.append({
                "Image": case_name,
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

            save_boundary_image(img_rgb, segments, os.path.join(algo_out_dir, f"{case_name}_all_superpixels.png"))
            save_boundary_image(img_rgb, in_context_map, os.path.join(algo_out_dir, f"{case_name}_in_context.png"))
            save_boundary_image(img_rgb, out_of_context_map, os.path.join(algo_out_dir, f"{case_name}_out_of_context.png"))
            save_boundary_image(img_rgb, None, os.path.join(algo_out_dir, f"{case_name}_orig.png"))


    # Generate Interactive Plotly Chart
    if performance_metrics:
        df_perf = pd.DataFrame(performance_metrics)
        print("\nAverage Execution Times:")
        print(df_perf.groupby("Algorithm")["Time_Seconds"].mean().reset_index())

        print("\nGenerating Interactive Plot...")
        
        # fig = px.bar(
        #     df_perf, 
        #     x="Image", 
        #     y="Time_Seconds", 
        #     color="Algorithm", 
        #     barmode="group",
        #     hover_data=["Num_Segments"],
        #     title="Diffusion Dataset Superpixel Benchmarking: Execution Time vs Segment Count",
        #     labels={"Time_Seconds": "Execution Time (Seconds)"},
        #     color_discrete_sequence=px.colors.qualitative.Prism
        # )
        
        # fig.show()

if __name__ == "__main__":
    main()
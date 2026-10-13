import os
import sys
import time
import argparse
import logging
import random
import json
import math
from math import exp
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb, rgb2gray
from skimage.metrics import structural_similarity
from skimage import color
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import scipy.stats as stats

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)

# =================================================================================
# 1. HELPER & ANALYSIS FUNCTIONS
# =================================================================================
def calculate_dice_score(mask1, mask2, label=1):
    if mask1.shape != mask2.shape:
        mask1 = np.resize(mask1, mask2.shape)
    mask1_bool = (mask1 == label)
    mask2_bool = (mask2 == label)
    intersection = np.sum(mask1_bool & mask2_bool)
    sum_of_masks = np.sum(mask1_bool) + np.sum(mask2_bool)
    if sum_of_masks == 0: return 100.0
    return ((2. * intersection) / sum_of_masks) * 100

def calculate_iou_in_roi(pred_mask, gt_mask, roi_mask):
    roi_mask = roi_mask.astype(bool)
    pred_roi = pred_mask[roi_mask]
    gt_roi = gt_mask[roi_mask]
    intersection = np.logical_and(pred_roi, gt_roi).sum()
    union = np.logical_or(pred_roi, gt_roi).sum()
    return (intersection / union)*100 if union > 0 else 0.0

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    unique_labels = np.unique(superpixels_segments)
    unique_labels = unique_labels[unique_labels != 0]
    gt_bool = (gt_mask > 0)
    ic_sps, oc_sps = [], []
    for sp_id in unique_labels:
        sp_mask = (superpixels_segments == sp_id)
        sp_area = np.sum(sp_mask)
        if sp_area == 0: continue
        gamma = np.sum(sp_mask & gt_bool) / sp_area
        if gamma >= ic_threshold: ic_sps.append(sp_id)
        elif gamma <= oc_threshold: oc_sps.append(sp_id)
    return ic_sps, oc_sps

def perturb_image_slice(image_01, segments, sp_to_perturb, noise_type, noise_level):
    perturbed_image = np.copy(image_01)
    
    region_mask = np.isin(segments, sp_to_perturb)
    if not np.any(region_mask):
        return perturbed_image

    if noise_type == 'gaussian':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] += noise[region_mask]
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, 0.05 * noise_level)
        num_pixels = np.sum(region_mask)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            rows, cols = np.where(region_mask)
            indices = np.random.choice(len(rows), num_noise, replace=False)
            perturbed_image[rows[indices], cols[indices]] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        sigma = max(0.05, 3.2 - noise_level)
        region = perturbed_image[region_mask]
        noisy_region = np.random.poisson(region * 300 * sigma) / (300 * sigma)
        perturbed_image[region_mask] = noisy_region
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1 + noise[region_mask])
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=10 * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

# --- 2D SPECIFIC SOTA GLOBAL PERTURBATION ---
def perturb_global_image_2d(image_01, noise_type, noise_level):
    perturbed_image = np.copy(image_01)
    
    if noise_type == 'gaussian':
        perturbed_image += np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
    elif noise_type == 'salt_pepper' or noise_type == 's&p':
        noise_char_value = min(0.5, 0.05 * noise_level)
        num_pixels = perturbed_image.size
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            rows = np.random.choice(perturbed_image.shape[0], num_noise)
            cols = np.random.choice(perturbed_image.shape[1], num_noise)
            perturbed_image[rows, cols] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        sigma = max(0.05, 3.2 - noise_level)
        perturbed_image = np.random.poisson(perturbed_image * 300 * sigma) / (300  * sigma)
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image *= (1 + noise)
    elif noise_type == 'gaussian_blur':
        perturbed_image = gaussian_filter(image_01, sigma=10 * noise_level)
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))

    return np.clip(perturbed_image, 0.0, 1.0)

def run_ranking_analysis(
    analysis_type, model_diffusion, device, batch_data, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, current_image_output_dir, sota_max_valid_levels, noise_step=0.01
):
    logger = logging.getLogger('base')
    analysis_results = {}
    lambda_param = 1.0 
    beta_param = 1.0
    
    # --- DYNAMIC SAFE ID CALCULATION ---
    if not superpixels_to_measure: return {}
    max_id = max(segments.max(), max(superpixels_to_measure)) + 1
    valid_sp_to_measure = [sp for sp in superpixels_to_measure if sp < max_id]
    
    bg_gt = (target_class_gt == 0)
    bg_pred_init = (initial_pred_mask_bool == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0
    
    intersect_mask_init = (initial_pred_mask_bool == 1) & (target_class_gt == 1)
    union_mask_init = (initial_pred_mask_bool == 1) | (target_class_gt == 1)
    
    intersect_counts_init = np.bincount(segments[intersect_mask_init].ravel(), minlength=max_id)
    union_counts_init = np.bincount(segments[union_mask_init].ravel(), minlength=max_id)
    
    initial_sp_iou_data = []
    for sp_id in valid_sp_to_measure:
        u_val = union_counts_init[sp_id]
        i_val = intersect_counts_init[sp_id]
        iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
        initial_sp_iou_data.append({"sp_id": int(sp_id), "iou": round(iou_val, 2)})

    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): sp_data['rank'] = rank
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        level_data_list = [{"noise_level": 0.0, "image_path": "original", "superpixels": sorted_initial_sp_data, "instant_spillover_ratio": 0.0}]
        sp_iou_history = {sp_id: [] for sp_id in valid_sp_to_measure}

        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0

        # --- OPTIMIZATION: Dynamically cap PRISM inference to SOTA's boundary ---
        max_level = sota_max_valid_levels.get(noise_type, 0.0)
        
        if max_level < noise_step:
            noise_levels_arr = [] 
        else:
            noise_levels_arr = np.arange(noise_step, max_level + 1e-5, noise_step)
        
        for step_idx, noise_level in enumerate(noise_levels_arr): 
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, superpixels_to_perturb, noise_type, noise_level)
            
            # --- SSIM FREEZE LOGIC ---
            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            if ssim_score < 0.70:
                logger.info(f"[PRISM {analysis_type}] SSIM break at {noise_level:.2f}. Halting.")
                break
                
            if perturbed_img_gray.ndim == 2:
                perturbed_tensor = torch.from_numpy(perturbed_img_gray).unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                perturbed_tensor = torch.from_numpy(perturbed_img_gray).permute(2,0,1).unsqueeze(0).float().to(device)
            
            perturbed_tensor_m11 = (perturbed_tensor * 2.0) - 1.0

            val_data_perturbed = batch_data.copy()
            val_data_perturbed['A'] = perturbed_tensor_m11
            model_diffusion.feed_data(val_data_perturbed)
            model_diffusion.test()
            
            perturbed_predseg_raw = model_diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
            perturbed_pred_mask_bool = ((perturbed_predseg_raw + 1.0) / 2.0 > 0.5)

            bg_pred_pert = (perturbed_pred_mask_bool == 0)
            intersection_bg_pert = np.sum(bg_gt & bg_pred_pert)
            union_bg_pert = np.sum(bg_gt | bg_pred_pert)
            iou_bg_pert = (intersection_bg_pert / union_bg_pert * 100.0) if union_bg_pert > 0 else 100.0
            
            delta_iou_bg = iou_bg_base - iou_bg_pert
            if delta_iou_bg > 0:
                spillover_ratio_at_j = delta_iou_bg / 100.0
                cumulative_spillover_ratio += spillover_ratio_at_j
                conditional_bleed_count += 1
            else:
                spillover_ratio_at_j = 0.0

            intersect_mask_pert = (initial_pred_mask_bool == 1) & (perturbed_pred_mask_bool == 1)
            union_mask_pert = (initial_pred_mask_bool == 1) | (perturbed_pred_mask_bool == 1)
            
            intersect_counts_pert = np.bincount(segments[intersect_mask_pert].ravel(), minlength=max_id)
            union_counts_pert = np.bincount(segments[union_mask_pert].ravel(), minlength=max_id)
            
            current_level_sp_data = []
            for sp_id in valid_sp_to_measure:
                u_val = union_counts_pert[sp_id]
                i_val = intersect_counts_pert[sp_id]
                iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
                current_level_sp_data.append({"sp_id": int(sp_id), "iou": round(iou_val, 2)})

            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)          
            for sp_id in valid_sp_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
            
            level_data_list.append({
                "noise_level": round(noise_level, 2), 
                "instant_spillover_ratio": round(spillover_ratio_at_j, 4),
                "superpixels": sorted_superpixels
            })
        
        avg_spillover = cumulative_spillover_ratio / conditional_bleed_count if conditional_bleed_count > 0 else 0.0
        spillover_discount = math.exp(-beta_param * avg_spillover)
        
        analysis_results[noise_type] = {
            "spillover_metrics": { 
                "conditional_average_spillover": round(avg_spillover, 4),
                "bleed_count": conditional_bleed_count,
                "global_spillover_discount": round(spillover_discount, 4)
            },
            "level_data": level_data_list
        }
    return analysis_results

# =================================================================================
# 2. MAIN EXECUTION (TWO-PHASE RANK ABLATION ARCHITECTURE)
# =================================================================================

def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Reliability Inference + Rank Ablation")
    
    # Diffusion Specific Logger / Config Args
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)

    # Two-Phase Core Args
    parser.add_argument("--output_dir", type=str, default="./sota_prism_ablation_results")
    parser.add_argument("--num_images", type=int, default=50, help="Strictly process exactly 50 qualified images for ablation study.")
    parser.add_argument("--noise_step", type=float, default=0.01, help="Step size for noise loop.")
    parser.add_argument("--noise_max", type=float, default=3.0, help="Maximum noise boundary.")
    
    parser.add_argument("--num_superpixels", type=int, default=500)
    # parser.add_argument("--noise_types", nargs='+', default=['gaussian_blur'])
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--img_path', type=str, default='/bhupender/2025/cdarl_mouse/img_sel.txt', help="Path to the frozen images file generated")

    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir:
        if 'datasets' in opt and 'val' in opt['datasets']: opt['datasets']['val']['dataroot'] = args.data_dir
        elif 'datasets' in opt and 'test' in opt['datasets']: opt['datasets']['test']['dataroot'] = args.data_dir
        else: opt['datasets'] = {'test': {'name': 'xcad_test', 'mode': args.phase, 'dataroot': args.data_dir}}

    opt['phase'] = args.phase
    if args.gpu_ids: opt['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
    elif opt['gpu_ids'] is None: opt['gpu_ids'] = []

    if 'path' not in opt: opt['path'] = {}
    if 'log' not in opt['path']: opt['path']['log'] = './logs'
    if 'results' not in opt['path']: opt['path']['results'] = './results'

    final_log_dir = os.path.dirname(opt['path']['log']) if os.path.splitext(opt['path']['log'])[1] else opt['path']['log']
    os.makedirs(final_log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = logging.DEBUG if args.debug else logging.INFO
    Logger.setup_logger(None, final_log_dir, opt['phase'], level=log_level, screen=True)
    logger = logging.getLogger('base')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading Diffusion Dataset...")
    dataset_opt = opt['datasets']['test']
    val_set = Data.create_dataset_xcad(dataset_opt, 'test')
    val_loader = Data.create_dataloader(val_set, dataset_opt, 'test')

    logger.info("Loading Diffusion model...")
    diffusion = Model.create_model(opt)
    diffusion.netG.to(device)

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify EXACTLY {args.num_images} Images for Ablation...")
    qualified_images = []

    cohort_file = args.img_path
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing. Run generate script first.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])

    for idata, val_data in enumerate(tqdm(val_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break

        raw_filename = os.path.basename(val_data['P'][0])
        case_name = os.path.splitext(raw_filename)[0]
        
        if case_name not in frozen_cohort:
            continue
        
        orig_image_np_raw = val_data['A'][0].cpu().numpy()
        if orig_image_np_raw.ndim == 3:
            orig_image_np = rgb2gray(orig_image_np_raw.transpose(1, 2, 0)) if orig_image_np_raw.shape[0] == 3 else orig_image_np_raw.squeeze()
        else:
            orig_image_np = orig_image_np_raw
            
        orig_image_01 = np.clip((orig_image_np + 1.0) / 2.0, 0.0, 1.0)

        diffusion.feed_data(val_data)
        diffusion.test()
        
        predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
        initial_pred_mask_bool = ((predseg_raw + 1.0) / 2.0 > 0.5)

        gt_mask_raw = val_data['F'][0].squeeze().cpu().numpy()
        target_class_gt = ((gt_mask_raw + 1.0) / 2.0 > 0.5)
        full_image_roi = np.ones_like(target_class_gt, dtype=bool)

        segments = slic(orig_image_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=0.85, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'case_name': case_name, 'val_data': val_data, 'orig_image_01': orig_image_01, 
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")

    # =========================================================================
    # PHASE 2: SOTA (First) -> PRISM (Truncated) SWEEPS
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing SOTA (Upper Bound) & PRISM Noise Analysis ({args.noise_step} to {args.noise_max})...")
    
    all_images_analysis = {}
    
    for q_data in tqdm(qualified_images, desc="Processing Image Cohort"):
        case_name = q_data['case_name']
        val_data = q_data['val_data']
        orig_image_01 = q_data['orig_image_01']
        target_class_gt = q_data['target_class_gt']
        segments = q_data['segments']
        initial_pred_mask_bool = q_data['initial_pred_mask_bool']
        qualified_ic_sp, oc_sps_all = q_data['qualified_ic_sp'], q_data['oc_sps_all']

        final_analysis = {"image_info": {"name": case_name}}

        # --- 1. SOTA Pipeline (Run FIRST to Establish Truncation Boundary) ---
        full_roi = np.ones_like(target_class_gt, dtype=bool)
        Af_c = calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, full_roi) / 100.0
        if Af_c == 0: Af_c = 1e-6 

        sota_results = {}
        sota_max_valid_levels = {}
        
        for noise_type in args.noise_types:
            sota_results[noise_type] = []
            max_level_for_this_noise = 0.0
            
            for noise_level in np.arange(args.noise_step, args.noise_max + 1e-5, args.noise_step):

                pert_img = perturb_global_image_2d(orig_image_01, noise_type, noise_level)

                ssim_score = structural_similarity(orig_image_01, pert_img, data_range=1.0)
                
                # --- SSIM BREAK LOGIC ---
                if ssim_score >= 0.70:
                    max_level_for_this_noise = noise_level
                    
                    if pert_img.ndim == 2:
                        pert_tensor = torch.from_numpy(pert_img).unsqueeze(0).unsqueeze(0).float().to(device)
                    else:
                        pert_tensor = torch.from_numpy(pert_img).permute(2,0,1).unsqueeze(0).float().to(device)
                    
                    pert_tensor_m11 = (pert_tensor * 2.0) - 1.0
                    val_data_pert = val_data.copy()
                    val_data_pert['A'] = pert_tensor_m11
                    
                    diffusion.feed_data(val_data_pert)
                    diffusion.test()
                    
                    pert_predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
                    pert_pred_mask_bool = ((pert_predseg_raw + 1.0) / 2.0 > 0.5)
                else:
                    logger.info(f"[SOTA {case_name}] SSIM break at {noise_level:.2f} for noise type '{noise_type}'. Halting further perturbation.")     
                    break 
                
                
                Af_ps = calculate_iou_in_roi(pert_pred_mask_bool, target_class_gt, full_roi) / 100.0
                gamma_r = Af_ps / Af_c  
                sota_results[noise_type].append({'noise_level': round(noise_level, 2), 'gamma_r': gamma_r})
                
            sota_max_valid_levels[noise_type] = max_level_for_this_noise
            
        final_analysis['sota_analysis'] = sota_results

        # --- 2. PRISM Pipeline (Dynamically Capped via sota_max_valid_levels) ---
        res_ic = run_ranking_analysis(
            'in_context', diffusion, device, val_data, orig_image_01, segments, target_class_gt, 
            qualified_ic_sp, qualified_ic_sp, initial_pred_mask_bool,
            args.noise_types, '', sota_max_valid_levels, noise_step=args.noise_step
        )
        final_analysis['in_context_analysis'] = res_ic
        
        num_oc_to_perturb = int(0.35 * len(oc_sps_all))
        out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
        res_oc = run_ranking_analysis(
            'out_of_context', diffusion, device, val_data, orig_image_01, segments, target_class_gt, 
            out_of_context_sp_to_perturb, qualified_ic_sp, initial_pred_mask_bool,
            args.noise_types, '', sota_max_valid_levels, noise_step=args.noise_step
        )
        final_analysis['out_of_context_analysis'] = res_oc
        
        all_images_analysis[case_name] = final_analysis

    # =========================================================================
    # PHASE 3: RANK ENTROPY ABLATION (STRICT COMMON GROUND)
    # =========================================================================
    logger.info("\n=== [PHASE 3] Rank Entropy Ablation: Strict Common Ground Evaluation ===")
    
    entropy_results = []
    img_names_ordered = [q['case_name'] for q in qualified_images]
    
    for noise_type in args.noise_types:
        min_valid_steps = 999 
        
        for img_name in img_names_ordered:
            data = all_images_analysis[img_name]
            sota_len = len(data['sota_analysis'][noise_type])
            ic_len = len(data['in_context_analysis'][noise_type]['level_data']) - 1
            oc_len = len(data['out_of_context_analysis'][noise_type]['level_data']) - 1
            min_valid_steps = min(min_valid_steps, sota_len, ic_len, oc_len)
            
        logger.info(f"[{noise_type.upper()}] Common Ground strictly established at {min_valid_steps} perturbation steps.")
        if min_valid_steps == 0: continue
            
        sota_rank_matrix = {img: [] for img in img_names_ordered}
        prism_rank_matrix = {img: [] for img in img_names_ordered}
        
        for step_idx in range(min_valid_steps):
            sota_level_scores = []
            prism_level_scores = []
            
            for img_name in img_names_ordered:
                data = all_images_analysis[img_name]
                
                sota_gamma = data['sota_analysis'][noise_type][step_idx]['gamma_r']
                sota_level_scores.append(sota_gamma)
                
                ic_lvl_data = data['in_context_analysis'][noise_type]['level_data'][step_idx + 1]
                oc_lvl_data = data['out_of_context_analysis'][noise_type]['level_data'][step_idx + 1]
                
                ic_iou = np.mean([sp['iou'] for sp in ic_lvl_data['superpixels']]) / 100.0 if ic_lvl_data['superpixels'] else 0.0
                oc_iou = np.mean([sp['iou'] for sp in oc_lvl_data['superpixels']]) / 100.0 if oc_lvl_data['superpixels'] else 0.0
                
                R_ic = ic_iou * math.exp(-1.0 * ic_lvl_data['instant_spillover_ratio'])
                R_oc = oc_iou * math.exp(-1.0 * oc_lvl_data['instant_spillover_ratio'])
                
                R_prism = (2 * R_ic * R_oc) / (R_ic + R_oc) if (R_ic + R_oc) > 0 else 0.0
                prism_level_scores.append(R_prism)
            
            jitter_s = np.random.normal(0, 1e-9, len(sota_level_scores))
            sota_ranks = stats.rankdata(-(np.array(sota_level_scores) + jitter_s), method='ordinal')
            
            jitter_p = np.random.normal(0, 1e-9, len(prism_level_scores))
            prism_ranks = stats.rankdata(-(np.array(prism_level_scores) + jitter_p), method='ordinal')
            
            for i, img_name in enumerate(img_names_ordered):
                sota_rank_matrix[img_name].append(sota_ranks[i])
                prism_rank_matrix[img_name].append(prism_ranks[i])
                
        for img_name in img_names_ordered:
            s_counts = np.bincount(sota_rank_matrix[img_name], minlength=len(img_names_ordered)+1)[1:]
            p_counts = np.bincount(prism_rank_matrix[img_name], minlength=len(img_names_ordered)+1)[1:]
            
            H_sota = stats.entropy(s_counts, base=2)
            H_prism = stats.entropy(p_counts, base=2)
            
            entropy_results.append({'Noise Type': noise_type, 'Image': img_name, 'SOTA Entropy': H_sota, 'PRISM Entropy': H_prism})

    # =========================================================================
    # PHASE 4: DISCRIMINATIVE RESOLUTION (Levene's Variance Test) & JSON EXPORT
    # =========================================================================
    df_entropy = pd.DataFrame(entropy_results)
    
    logger.info("\n" + "="*70)
    logger.info("FORMAL STATISTICAL PROOFS (LEVENE'S VARIANCE TEST)")
    logger.info("="*70)
    
    global_summary_data = []
    
    for noise_type in args.noise_types:
        if df_entropy.empty or noise_type not in df_entropy['Noise Type'].unique():
            logger.warning(f"Skipping statistical proof for {noise_type.upper()}: No valid common ground data found.")
            continue

        df_sub = df_entropy[df_entropy['Noise Type'] == noise_type].copy()
        sota_entropies = df_sub['SOTA Entropy'].values
        prism_entropies = df_sub['PRISM Entropy'].values
        
        var_sota = np.var(sota_entropies, ddof=1)
        var_prism = np.var(prism_entropies, ddof=1)
        
        if np.allclose(sota_entropies, prism_entropies):
            levene_stat, p_val = 0.0, 1.0
        else:
            levene_stat, p_val = stats.levene(sota_entropies, prism_entropies, center='median')
        
        is_significant = (p_val < 0.05) and (var_prism > var_sota)
        sig_marker = "***" if (p_val < 0.001 and is_significant) else "**" if (p_val < 0.01 and is_significant) else "*" if is_significant else "ns"
        
        logger.info(f"\n[STATISTICAL PROOF] {noise_type.upper()} NOISE:")
        logger.info(f"    -> PRISM Variance: {var_prism:.4f} | SOTA Variance: {var_sota:.4f}")
        logger.info(f"    -> Levene's Test: W = {levene_stat:.4f}, p = {p_val:.4e}")
        
        if is_significant:
            logger.info("    -> Conclusion: Null rejected. PRISM provides significantly higher discriminative spread.")
        elif p_val < 0.05 and var_prism <= var_sota:
            logger.info("    -> Conclusion: Null rejected, but SOTA has higher variance (unexpected).")
        else:
            logger.info("    -> Conclusion: Failed to reject null. Variances are not significantly different.")
            
        global_summary_data.append({
            'Noise_Type': noise_type.upper(),
            'SOTA_Var': float(var_sota), 'PRISM_Var': float(var_prism),
            'levene_statistic': float(levene_stat), 'p_value': float(p_val),
            'significance_marker': sig_marker
        })
            
        plt.figure(figsize=(14, 6))
        x = np.arange(len(df_sub))
        width = 0.35
        
        plt.bar(x - width/2, df_sub['SOTA Entropy'], width, label=f'SOTA (Var: {var_sota:.2f})', color='salmon', edgecolor='black')
        plt.bar(x + width/2, df_sub['PRISM Entropy'], width, label=f'PRISM (Var: {var_prism:.2f})', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Cohort Image Identifiers', fontsize=14)
        plt.ylabel('Rank Entropy ($H_i$) [bits]', fontsize=14)
        plot_title = f'Discriminative Rank Stability Ablation: {noise_type.upper()} Noise\nHigher Variance = Better Discriminative Spread | Levene p = {p_val:.4e} ({sig_marker})'
        plt.title(plot_title, fontsize=16, fontweight='bold')
        
        plt.xticks(x, [f"Img {i+1}" for i in range(len(df_sub))], fontsize=12, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"Ablation_RankEntropy_{noise_type}.png"), dpi=300)
        plt.close()
        
    if global_summary_data:
        df_summary = pd.DataFrame(global_summary_data)
        plt.figure(figsize=(12, 7))
        x_summary = np.arange(len(df_summary))
        width_summary = 0.35
        
        plt.bar(x_summary - width_summary/2, df_summary['SOTA_Var'], width_summary, label=r'SOTA Variance', color='salmon', edgecolor='black')
        plt.bar(x_summary + width_summary/2, df_summary['PRISM_Var'], width_summary, label=r'PRISM Variance', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Perturbation Types', fontsize=14, fontweight='bold')
        plt.ylabel('Entropy Variance ($\sigma^2$)', fontsize=14, fontweight='bold')
        plt.title('Global Discriminative Resolution Across Perturbations\nHigher Variance = Superior Ability to Distinguish Model Performance', fontsize=16, fontweight='bold')
        plt.xticks(x_summary, df_summary['Noise_Type'], fontsize=12, rotation=15)
        
        for idx, row in df_summary.iterrows():
            max_y = max(row['SOTA_Var'], row['PRISM_Var'])
            plt.text(idx, max_y + (0.05 * max_y), f"{row['significance_marker']} (L)", ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
        
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "Ablation_Variance_ALL_SUMMARY.png"), dpi=300)
        plt.close()

    json_export_data = {"statistical_summary": global_summary_data, "raw_image_entropies": entropy_results}
    with open(os.path.join(args.output_dir, "Ablation_Statistical_Results.json"), 'w') as f:
        json.dump(json_export_data, f, indent=4, cls=NpEncoder)
        
    logger.info(f"\n--- Ablation Finished. Visuals and JSON Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    main()
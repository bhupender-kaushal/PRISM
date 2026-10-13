import os
import argparse
import json
import logging
import random
import math
from abc import ABC, abstractmethod
from math import exp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import h5py

from scipy.ndimage import gaussian_filter, zoom
from scipy.ndimage import center_of_mass, label
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid
from sklearn.decomposition import PCA
import scipy.stats as stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Directly import the model from your project structure
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# =================================================================================
# 1. DATASET LOADER
# =================================================================================
class Synapse_dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, list_dir, split, img_size):
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vol_name = self.sample_list[idx].strip('\n')
        filepath = os.path.join(self.data_dir, f"{vol_name}.npy.h5")
        data = h5py.File(filepath, 'r')
        image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label, 'case_name': vol_name}
        return sample

# =================================================================================
# 2. HELPER & ANALYSIS FUNCTIONS 
# =================================================================================
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
    elif noise_type == 'salt_pepper' or noise_type == 's&p':
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
        noisy_region = np.random.poisson(region * 100   * sigma) / (100 * sigma)
        perturbed_image[region_mask] = noisy_region
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1 + noise[region_mask])
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=10 * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.05 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.05* noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

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
        perturbed_image = np.random.poisson(perturbed_image * 100   * sigma) / (100  * sigma)
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image *= (1 + noise)
    elif noise_type == 'gaussian_blur':
        perturbed_image = gaussian_filter(image_01, sigma=10 * noise_level)
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.05 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.05* noise_level, perturbed_image.shape)
        perturbed_image = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))

    return np.clip(perturbed_image, 0.0, 1.0)

def run_inference_on_slice(model, image_slice, patch_size, device='cuda'):
    h, w = image_slice.shape
    slice_rescaled = zoom(image_slice, (patch_size / h, patch_size / w), order=3) if (h != patch_size or w != patch_size) else image_slice
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = x_transforms(slice_rescaled).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    if h != patch_size or w != patch_size:
        prediction = zoom(prediction, (h / patch_size, w / patch_size), order=0)
    return prediction, input_tensor

def run_ranking_analysis(
    analysis_type, model, device, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, patch_size, target_class_index, sota_max_valid_levels, 
    noise_step=0.1
):
    logger = logging.getLogger('base')
    analysis_results = {}
    lambda_param = 1.0 
    beta_param = 1.0
    
    # --- DYNAMIC SAFE ID CALCULATION ---
    # Fixes the bincount IndexError by accommodating all IDs actually in play
    if not superpixels_to_measure: return {}
    max_id = max(segments.max(), max(superpixels_to_measure)) + 1
    
    # Filter ghost IDs
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
            
            perturbed_pred_mask_full, _ = run_inference_on_slice(model, perturbed_img_gray, patch_size, device)
            perturbed_pred_mask_bool = (perturbed_pred_mask_full == target_class_index)

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

def find_samples_with_class(dataset, target_class, area_threshold_percentage=0.9):
    candidate_slices = []
    logging.info(f"Scanning volumes to find slices with area >= {area_threshold_percentage*100}% of max area for class {target_class}...")

    for vol_idx in tqdm(range(len(dataset)), desc="Scanning Volumes"):
        sample = dataset[vol_idx]
        slices_in_volume = []
        for slice_idx in range(sample['label'].shape[0]):
            label_slice = sample['label'][slice_idx]
            if np.any(label_slice == target_class):
                area = np.sum(label_slice == target_class)
                slices_in_volume.append({'vol_idx': vol_idx, 'slice_idx': slice_idx, 'area': area})
        if slices_in_volume:
            max_area_in_volume = max(s['area'] for s in slices_in_volume)
            area_threshold = max_area_in_volume * area_threshold_percentage
            for s in slices_in_volume:
                if s['area'] >= area_threshold: candidate_slices.append(s)
            
    return sorted(candidate_slices, key=lambda x: x['area'], reverse=True)

# =================================================================================
# 3. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Loading MaxViT model...")
    net = MaxViT_deformableLKAFormer(num_classes=args.num_classes).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    logger.info("Loading Synapse dataset...")
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    candidate_samples = find_samples_with_class(db_test, args.target_class_index)
    if not candidate_samples: sys.exit()
    random.shuffle(candidate_samples)


    cohort_file = args.img_cohort
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing. Run generate_synapse_cohort.py first.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])
    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify EXACTLY {args.num_images} Images for Ablation...")
    qualified_images = []
    
    for sample_info in tqdm(candidate_samples, desc="Qualifying Slices"):
        if len(qualified_images) >= args.num_images: break

        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        volume_data = db_test[vol_idx]
        image_slice, label_slice = volume_data['image'][slice_idx, :, :], volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"
        
        if case_name not in frozen_cohort: continue
        
        image_slice_01 = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        initial_pred_mask, input_tensor = run_inference_on_slice(net, image_slice_01, args.img_size, device)
        
        target_class_gt = (label_slice == args.target_class_index)
        initial_pred_mask_bool = (initial_pred_mask == args.target_class_index)
        full_image_roi = np.ones_like(target_class_gt, dtype=bool)
        
        # if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, full_image_roi) < 70: continue

        segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=1.0, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'case_name': case_name, 'image_slice_01': image_slice_01, 
            'target_class_gt': target_class_gt, 'segments': segments, 'initial_pred_mask_bool': initial_pred_mask_bool,
            'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")

    # =========================================================================
    # PHASE 2: SOTA (First) -> PRISM (Truncated) SWEEPS
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing SOTA (Upper Bound) & PRISM Noise Analysis ({args.noise_step} to {args.noise_max})...")
    
    all_images_analysis = {}
    
    for q_data in tqdm(qualified_images, desc="Processing Image Cohort"):
        case_name = q_data['case_name']
        image_slice_01 = q_data['image_slice_01']
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
                pert_img = perturb_global_image_2d(image_slice_01, noise_type, noise_level)
                ssim_score = structural_similarity(image_slice_01, pert_img, data_range=1.0)
                
                if ssim_score >= 0.70:
                    max_level_for_this_noise = noise_level
                    perturbed_pred_mask_full, _ = run_inference_on_slice(net, pert_img, args.img_size, device)
                    perturbed_pred_mask_bool = (perturbed_pred_mask_full == args.target_class_index)
                else:
                    logger.info(f"[SOTA {case_name}] SSIM break at {noise_level:.2f} for noise type '{noise_type}'. Halting inference further.")
                    break 
                
                Af_ps = calculate_iou_in_roi(perturbed_pred_mask_bool, target_class_gt, full_roi) / 100.0
                gamma_r = Af_ps / Af_c  
                sota_results[noise_type].append({'noise_level': round(noise_level, 2), 'gamma_r': gamma_r})
            
            sota_max_valid_levels[noise_type] = max_level_for_this_noise
                
        final_analysis['sota_analysis'] = sota_results

        # --- 2. PRISM Pipeline (Dynamically Capped via sota_max_valid_levels) ---
        res_ic = run_ranking_analysis(
            'in_context', net, device, image_slice_01, segments, target_class_gt, 
            qualified_ic_sp, qualified_ic_sp, initial_pred_mask_bool,
            args.noise_types, args.img_size, args.target_class_index, sota_max_valid_levels, 
            noise_step=args.noise_step
        )
        final_analysis['in_context_analysis'] = res_ic
        
        num_oc_to_perturb = int(0.70 * len(oc_sps_all))
        out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
        res_oc = run_ranking_analysis(
            'out_of_context', net, device, image_slice_01, segments, target_class_gt, 
            out_of_context_sp_to_perturb, qualified_ic_sp, initial_pred_mask_bool,
            args.noise_types, args.img_size, args.target_class_index, sota_max_valid_levels, 
            noise_step=args.noise_step
        )
        final_analysis['out_of_context_analysis'] = res_oc

        all_images_analysis[case_name] = final_analysis

    # =========================================================================
    # PHASE 3: ABLATION STUDY - STRICT COMMON GROUND
    # =========================================================================
    logger.info("\n=== [PHASE 3] Rank Entropy Ablation: Strict Common Ground Evaluation ===")
    
    entropy_results = []
    img_names_ordered = [q['case_name'] for q in qualified_images]
    
    for noise_type in args.noise_types:
        min_valid_steps = 999 
        
        for img_name in img_names_ordered:
            data_img = all_images_analysis[img_name]
            sota_len = len(data_img['sota_analysis'][noise_type])
            ic_len = len(data_img['in_context_analysis'][noise_type]['level_data']) - 1
            oc_len = len(data_img['out_of_context_analysis'][noise_type]['level_data']) - 1
            min_valid_steps = min(min_valid_steps, sota_len, ic_len, oc_len)
            
        logger.info(f"[{noise_type.upper()}] Common Ground strictly established at {min_valid_steps} steps.")
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
    # PHASE 4: ABLATION VISUALIZATION, STATISTICAL PROOF & JSON EXPORT
    # =========================================================================
    df_entropy = pd.DataFrame(entropy_results)
    
    logger.info("\n" + "="*70)
    logger.info("FORMAL STATISTICAL PROOFS (SHAPIRO-WILK, T-TEST, WILCOXON)")
    logger.info("="*70)
    
    global_summary_data = []
    
    for noise_type in args.noise_types:
        if df_entropy.empty or noise_type not in df_entropy['Noise Type'].unique():
            continue

        df_sub = df_entropy[df_entropy['Noise Type'] == noise_type].copy()
        sota_entropies = df_sub['SOTA Entropy'].values
        prism_entropies = df_sub['PRISM Entropy'].values
        
        differences = sota_entropies - prism_entropies
        if np.allclose(sota_entropies, prism_entropies) or len(differences) < 3:
            shapiro_stat, shapiro_p = 1.0, 1.0
            is_normal = True
        else:
            shapiro_stat, shapiro_p = stats.shapiro(differences)
            is_normal = shapiro_p > 0.05
            
        if np.allclose(sota_entropies, prism_entropies):
            t_stat, t_p_val = 0.0, 1.0 
        else:
            t_stat, t_p_val = stats.ttest_rel(sota_entropies, prism_entropies, alternative='greater')
            
        if np.allclose(sota_entropies, prism_entropies):
            w_stat, w_p_val, cles, r_rb = 0.0, 1.0, 0.0, 0.0 
        else:
            w_stat, w_p_val = stats.wilcoxon(sota_entropies, prism_entropies, alternative='greater')
            cles = np.mean(sota_entropies > prism_entropies)
            diff_nz = differences[differences != 0]
            if len(diff_nz) == 0:
                r_rb = 0.0
            else:
                ranks = stats.rankdata(np.abs(diff_nz))
                w_plus = np.sum(ranks[diff_nz > 0])
                w_minus = np.sum(ranks[diff_nz < 0])
                r_rb = (w_plus - w_minus) / (w_plus + w_minus)
            
        mean_sota, std_sota = np.mean(sota_entropies), np.std(sota_entropies)
        mean_prism, std_prism = np.mean(prism_entropies), np.std(prism_entropies)
        
        logger.info(f"\n[STATISTICAL PROOF] {noise_type.upper()} NOISE:")
        logger.info(f"    -> PRISM Mean Entropy: {mean_prism:.4f} bits | SOTA Mean Entropy: {mean_sota:.4f} bits")
        logger.info(f"    -> Shapiro-Wilk (Normality): W = {shapiro_stat:.4f}, p = {shapiro_p:.4e} (Normal: {is_normal})")
        logger.info(f"    -> Paired t-test: t = {t_stat:.4f}, p = {t_p_val:.4e}")
        logger.info(f"    -> Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {w_p_val:.4e}")
        
        if is_normal:
            final_p_val = t_p_val
            test_used = "Paired t-test"
        else:
            final_p_val = w_p_val
            test_used = "Wilcoxon Signed-Rank"
            
        logger.info(f"    -> Selected Test: {test_used}")
        
        if final_p_val < 0.05:
            sig_marker = "***" if final_p_val < 0.001 else "**" if final_p_val < 0.01 else "*"
        else:
            sig_marker = "ns"
            
        global_summary_data.append({
            'Noise_Type': noise_type.upper(),
            'SOTA_Mean_Entropy': float(mean_sota), 'SOTA_Std_Entropy': float(std_sota),
            'PRISM_Mean_Entropy': float(mean_prism), 'PRISM_Std_Entropy': float(std_prism),
            'shapiro_p': float(shapiro_p), 'is_normal': is_normal,
            't_statistic': float(t_stat), 't_p_value': float(t_p_val),
            'w_statistic': float(w_stat), 'w_p_value': float(w_p_val),
            'selected_test': test_used, 'final_p_value': float(final_p_val),
            'CLES': float(cles), 'Rank_Biserial': float(r_rb), 'significance_marker': sig_marker
        })
            
        plt.figure(figsize=(14, 6))
        x = np.arange(len(df_sub))
        width = 0.35
        
        plt.bar(x - width/2, df_sub['SOTA Entropy'], width, label=r'SOTA (Global $\gamma_r$)', color='salmon', edgecolor='black')
        plt.bar(x + width/2, df_sub['PRISM Entropy'], width, label=r'PRISM (Regional $R$)', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Cohort Image Identifiers', fontsize=14)
        plt.ylabel('Rank Entropy ($H_i$) [bits]', fontsize=14)
        plot_title = f'Discriminative Rank Stability Ablation: {noise_type.upper()} Noise\nLower Entropy = Stable Ranking | SOTA vs PRISM: p = {final_p_val:.4e} ({sig_marker}) [{test_used}]'
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
        
        plt.bar(x_summary - width_summary/2, df_summary['SOTA_Mean_Entropy'], width_summary, yerr=df_summary['SOTA_Std_Entropy'], capsize=5, label=r'SOTA (Global $\gamma_r$)', color='salmon', edgecolor='black')
        plt.bar(x_summary + width_summary/2, df_summary['PRISM_Mean_Entropy'], width_summary, yerr=df_summary['PRISM_Std_Entropy'], capsize=5, label=r'PRISM (Regional $R$)', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Perturbation Types', fontsize=14, fontweight='bold')
        plt.ylabel(r'Average Rank Entropy ($\bar{H}$) [bits]', fontsize=14, fontweight='bold')
        plt.title('Global Rank Stability Comparison Across All Perturbations\nLower Average Entropy = Superior Rank Stability', fontsize=16, fontweight='bold')
        plt.xticks(x_summary, df_summary['Noise_Type'], fontsize=12, rotation=15)
        
        for idx, row in df_summary.iterrows():
            max_y = max(row['SOTA_Mean_Entropy'] + row['SOTA_Std_Entropy'], row['PRISM_Mean_Entropy'] + row['PRISM_Std_Entropy'])
            test_abbr = "T" if row['selected_test'] == "Paired t-test" else "W"
            plt.text(idx, max_y + 0.1, f"{row['significance_marker']} ({test_abbr})", ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
        
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "Ablation_RankEntropy_ALL_SUMMARY.png"), dpi=300)
        plt.close()

    json_export_data = {"statistical_summary": global_summary_data, "raw_image_entropies": entropy_results}
    with open(os.path.join(args.output_dir, "Ablation_Statistical_Results.json"), 'w') as f:
        json.dump(json_export_data, f, indent=4, cls=NpEncoder)
        
    logger.info(f"\n--- Ablation Finished. Entropy Data, Visuals, Summary Graph, JSON Tables, and T-Test Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxViT Model Reliability Inference + Rank Ablation")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--model_path", type=str, default="./MaxViT_deform_LKA.pth")
    parser.add_argument("--output_dir", type=str, default="./sota_prism_ablation_results")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int,default=8)
    
    parser.add_argument("--num_images", type=int, default=50, help="Strictly process exactly 50 qualified images for ablation study.")
    parser.add_argument("--noise_step", type=float, default=0.010, help="Step size for noise loop.")
    parser.add_argument("--noise_max", type=float, default=3.0, help="Max noise level.")
    
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument("--img_cohort", type=str, default="/bhupender/2025/Mouse_other/Synapse/img_sel.txt", help="pre compiled image list.")
    args = parser.parse_args()
    main(args)
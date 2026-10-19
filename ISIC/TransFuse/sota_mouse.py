import os
import json
import argparse
import logging
import random
import sys
import sys
import time
import itertools
import math
from math import exp, ceil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid

from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import scipy.stats as stats
from skimage.color import gray2rgb, rgb2gray
import torch.utils.data as data

from lib.TransFuse import TransFuse_S 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# =================================================================================
# 1. DATASET LOADER 
# =================================================================================
class PngDataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.transform(image)
        gt = self.gt_transform(gt)
        name = os.path.basename(self.images[index]).split('.')[0]
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

# =================================================================================
# 2. HELPER & ANALYSIS FUNCTIONS 
# =================================================================================
def calculate_dice_score(mask1, mask2, label=1):
    if mask1.shape != mask2.shape:
        pil_img = Image.fromarray(mask1.astype(np.uint8))
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img)
    m1 = (mask1 == label).astype(int)
    m2 = (mask2 == label).astype(int)
    intersect = float(np.sum(m1 * m2))
    sum_of_masks = float(np.sum(m1) + np.sum(m2))
    if sum_of_masks == 0: return 100.0
    return ((2. * intersect) / sum_of_masks) * 100

def calculate_iou_score(mask1, mask2, roi_mask, label=1):
    roi_mask = roi_mask.astype(bool)
    if mask1.shape != mask2.shape:
        pil_img = Image.fromarray(mask1.astype(np.uint8))
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img)
    if mask1.shape != roi_mask.shape:
        pil_img_roi = Image.fromarray(roi_mask.astype(np.uint8))
        resized_img_roi = pil_img_roi.resize(mask1.shape[::-1], Image.NEAREST)
        roi_mask = np.array(resized_img_roi).astype(bool)
    mask1_roi, mask2_roi = mask1[roi_mask], mask2[roi_mask]
    m1_roi, m2_roi = (mask1_roi == label), (mask2_roi == label)
    intersect = np.sum(m1_roi & m2_roi)
    union = np.sum(m1_roi | m2_roi)
    if union == 0: return 100.0
    return (float(intersect) / float(union)) * 100

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    if superpixels_segments.shape != gt_mask.shape:
        pil_img = Image.fromarray(gt_mask.astype(np.uint8))
        resized_img = pil_img.resize(superpixels_segments.shape[::-1], Image.NEAREST)
        gt_mask = np.array(resized_img)
        
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

# --- PRISM Regional Perturbation (Color-Safe) ---
def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0
    
    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask): return perturbed_image, noise_char_value
        
    region_mask_3c = np.stack([region_mask]*3, axis=-1)

    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] += noise[region_mask_3c]
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, 0.05 * noise_level)
        rows, cols = np.where(region_mask)
        num_pixels_in_region = len(rows)
        num_noise = int(noise_char_value * num_pixels_in_region)
        if num_noise > 0:
            indices = np.random.choice(num_pixels_in_region, num_noise, replace=False)
            salt_or_pepper = np.random.choice([0.0, 1.0], num_noise)
            perturbed_image[rows[indices], cols[indices], :] = salt_or_pepper[:, np.newaxis]
            
    elif noise_type == 'poisson':
        intensity_scale = max(0.05, 3.0 - noise_level)
        for c in range(perturbed_image.shape[2]):
            channel_view = perturbed_image[:, :, c]
            channel_mask = region_mask_3c[:, :, c]
            region_pixels = channel_view[channel_mask]
            raw_noise = np.random.poisson(np.clip(region_pixels, 0, 1) * 500 * intensity_scale) / (500 * intensity_scale)
            perturbed_image[:, :, c][channel_mask] = np.clip(raw_noise, 0, 1)
            
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] *= (1 + noise[region_mask_3c])
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = 50 * noise_level
        blurred_image = np.zeros_like(original_image_scaled)
        for c in range(3):
            blurred_image[:, :, c] = gaussian_filter(original_image_scaled[:, :, c], sigma=noise_char_value)
        perturbed_image[region_mask_3c] = blurred_image[region_mask_3c]
        
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        region = perturbed_image[region_mask_3c]
        n1 = np.random.normal(0, noise_char_value, region.shape)
        n2 = np.random.normal(0, noise_char_value, region.shape)
        noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
        perturbed_image[region_mask_3c] = noisy_region

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

# --- SOTA Global Perturbation (Color-Safe) ---
def perturb_global_image(original_image_scaled_01, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled_01)
    
    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image += noise
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.3, 0.03 * noise_level)
        mask = np.random.rand(*perturbed_image.shape[:2]) < noise_char_value
        val = np.random.choice([0.0, 1.0], np.sum(mask))
        for c in range(3):
            perturbed_image[mask, c] = val
            
    elif noise_type == 'poisson':
        intensity_scale = max(0.05, 3.2 - noise_level)
        noisy_img = np.zeros_like(perturbed_image, dtype=np.float32)
        for c in range(perturbed_image.shape[2]):
            channel_data = np.clip(perturbed_image[:, :, c], 0, 1)
            scaled_counts = channel_data * 500 * intensity_scale
            noisy_img[:, :, c] = np.random.poisson(scaled_counts) / (500 * intensity_scale)
        perturbed_image = np.clip(noisy_img, 0, 1)
            
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image *= (1 + noise)
        
    elif noise_type == 'gaussian_blur':
        sigma_val =  (noise_level * 50)
        for c in range(3):
            perturbed_image[:, :, c] = gaussian_filter(perturbed_image[:, :, c], sigma=sigma_val)
            
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        n1 = np.random.normal(0, noise_char_value, perturbed_image.shape)
        n2 = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))

    return np.clip(perturbed_image, 0.0, 1.0)


def run_ranking_analysis(
    analysis_type, args, device, model,
    orig_image_np_01, pred_mask, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, sota_max_valid_levels, current_image_output_dir,
    ssim_data_collector, save_visuals=False
):
    logger = logging.getLogger('base')
    analysis_results = {}
    
    lambda_param = 1.0  
    beta_param = 1.0
    max_id = segments_slic.max() + 1 

    bg_gt = (gt_mask == 0)
    bg_pred_init = (pred_mask == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0
    
    intersect_mask_init = (pred_mask == 1) & (gt_mask > 0)
    union_mask_init = (pred_mask == 1) | (gt_mask > 0)
    
    intersect_counts_init = np.bincount(segments_slic[intersect_mask_init].ravel(), minlength=max_id)
    union_counts_init = np.bincount(segments_slic[union_mask_init].ravel(), minlength=max_id)
    
    initial_sp_iou_data = []
    for sp_id in superpixels_to_measure:
        u_val = union_counts_init[sp_id]
        i_val = intersect_counts_init[sp_id]
        iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
        initial_sp_iou_data.append({"sp_id": int(sp_id), "iou": round(iou_val, 2)})

    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): sp_data['rank'] = rank
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        level_0_entry = {"noise_level": 0.0, "image_path": "original", "superpixels": sorted_initial_sp_data, "instant_spillover_ratio": 0.0}
        level_data_list = [level_0_entry]
        
        sp_iou_history = {sp_id: [] for sp_id in superpixels_to_measure}
        sp_rank_history = {sp_id: [] for sp_id in superpixels_to_measure}
        
        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0
        current_pred_mask = pred_mask

        # --- OPTIMIZATION: Dynamically cap PRISM inference to SOTA's boundary ---
        max_level = sota_max_valid_levels.get(noise_type, 0.0)
        
        if max_level < args.noise_step:
            noise_levels_arr = [] # SOTA failed on first step, skip loop entirely
        else:
            noise_levels_arr = np.arange(args.noise_step, max_level + 1e-5, args.noise_step)

        for step_idx, noise_level in enumerate(noise_levels_arr):
            perturbed_image_np_scaled, _ = perturb_superpixels_with_level(orig_image_np_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_np_01, perturbed_image_np_scaled, data_range=1.0, channel_axis=-1)
            
            # --- SSIM BREAK LOGIC (Secondary protection) ---
            if ssim_score >= 0.70:
                perturbed_tensor = torch.from_numpy(perturbed_image_np_scaled).permute(2, 0, 1).float()
                normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                perturbed_tensor_normalized = normalize(perturbed_tensor).unsqueeze(0).to(device)

                with torch.no_grad():
                    # TransFuse specific forward pass
                    _, _, res = model(perturbed_tensor_normalized)
                    res = F.upsample(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    current_pred_mask = np.where(res >= 0.5, 1, 0)
            else:
                logger.info(f"[{analysis_type.upper()}] Noise: {noise_type}, Level: {round(noise_level, 2)}, SSIM: {round(ssim_score, 4)} - Below SSIM threshold. Breaking.")
                break

            bg_pred_pert = (current_pred_mask == 0)
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

            intersect_mask_pert = (pred_mask == 1) & (current_pred_mask == 1)
            union_mask_pert = (pred_mask == 1) | (current_pred_mask == 1)
            
            intersect_counts_pert = np.bincount(segments_slic[intersect_mask_pert].ravel(), minlength=max_id)
            union_counts_pert = np.bincount(segments_slic[union_mask_pert].ravel(), minlength=max_id)
            
            current_level_sp_data = []
            for sp_id in superpixels_to_measure:
                u_val = union_counts_pert[sp_id]
                i_val = intersect_counts_pert[sp_id]
                iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
                current_level_sp_data.append({"sp_id": int(sp_id), "iou": round(iou_val, 2)})

            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            
            for sp_id in superpixels_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
            
            level_data_list.append({
                "noise_level": round(noise_level, 2), 
                "instant_spillover_ratio": round(spillover_ratio_at_j, 4),
                "superpixels": sorted_superpixels
            })
        
        avg_spillover = cumulative_spillover_ratio / conditional_bleed_count if conditional_bleed_count > 0 else 0.0
        spillover_discount = math.exp(-beta_param * avg_spillover)

        reliability_results = []
        for sp_id in superpixels_to_measure:
            avg_iou = np.mean(sp_iou_history[sp_id]) if len(sp_iou_history[sp_id]) > 0 else 0.0
            initial_iou = initial_iou_map.get(sp_id, 0.0)
            
            rank_list = [r for r in sp_rank_history[sp_id] if r is not None]
            num_changes = sum(1 for i in range(1, len(rank_list)) if rank_list[i] != rank_list[i-1])
            sum_of_jumps = sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list)))
            rank_perplexity = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0
            rank_perplexity = rank_perplexity / len(superpixels_to_measure) if len(superpixels_to_measure) > 0 else 0.0

            confidence_weight = exp(-lambda_param * rank_perplexity)
            weighted_iou = avg_iou * confidence_weight
            base_reliability_score = (weighted_iou / 100.0) if initial_iou > 1e-9 else 0.0

            reliability_results.append({
                "sp_id": int(sp_id),
                "base_reliability_score": base_reliability_score,
                "weighted_iou": round(weighted_iou, 4),
                "confidence_weight": round(confidence_weight, 4),
                "avg_iou": round(avg_iou, 4),
                "initial_iou": round(initial_iou, 4),
                "rank_perplexity": round(rank_perplexity, 4)
            })

        analysis_results[noise_type] = {
            "spillover_metrics": { 
                "conditional_average_spillover": round(avg_spillover, 4),
                "bleed_count": conditional_bleed_count,
                "global_spillover_discount": round(spillover_discount, 4)
            },
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['base_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results


# --------------------------------------------------------------------------
# --- 3. Main Rank Entropy Ablation Pipeline (TransFuse)
# --------------------------------------------------------------------------
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading TransFuse model...")
    model_instance = TransFuse_S().to(device)
    model_instance.load_state_dict(torch.load(args.pth_path, map_location=device))
    model_instance.eval()

    logger.info("Loading ISIC dataset...")
    image_root = os.path.join(args.dataroot_val, 'images/')
    gt_root = os.path.join(args.dataroot_val, 'masks/')
    test_dataset = PngDataset(image_root, gt_root, args.testsize)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    noise_char_map = {noise: f'{noise}_strength' for noise in args.noise_types}

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Dataset to Qualify EXACTLY {args.num_images} Images for Ablation...")
    qualified_images = []

    cohort_file = args.image_file
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing. Run generate_isic_cohort.py first.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        # The script strips extensions, so we just match the base name
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])
    
    
    for i, (image, gt, name_list) in enumerate(tqdm(test_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break
        
        name = name_list[0]
        image_tensor = image.to(device)
        _, _, h, w = image_tensor.shape

        #for checking, we will use the base name without extension to match against the frozen cohort
        base_name = os.path.splitext(name)[0] if '.' in name else name
        
        # STRICT FILTER: Skip if not in our frozen list
        if base_name not in frozen_cohort:
            logger.info(f"Skipping {name}: Not in frozen cohort.")
            continue


        gt_mask_np = np.asarray(gt, np.float32).squeeze(0).squeeze(0)
        pil_gt = Image.fromarray(gt_mask_np.astype(np.uint8))
        gt_mask_np = np.array(pil_gt.resize((w, h), Image.NEAREST)).astype(np.float32)
        gt_mask_np /= (gt_mask_np.max() + 1e-8)

        with torch.no_grad():
            _, _, res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(res >= 0.5, 1, 0)
        
        full_image_roi = np.ones_like(gt_mask_np, dtype=bool)
        # if calculate_iou_score(pred_mask, gt_mask_np, full_image_roi) < 50: continue

        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)

        segments_slic = slic(orig_image_np_01, n_segments=args.superpixel_size, compactness=10, sigma=1)

        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask_np, ic_threshold=1.0, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_score(pred_mask, gt_mask_np, segments_slic == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'name': name, 'orig_image_np_01': orig_image_np_01, 'gt_mask_np': gt_mask_np, 
            'segments_slic': segments_slic, 'pred_mask': pred_mask, 
            'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) < args.num_images:
        logger.warning(f"Only found {len(qualified_images)} qualified images. Continuing with available cohort.")

    # =========================================================================
    # PHASE 2: SOTA (First) -> PRISM (Truncated) SWEEPS
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing SOTA (Upper Bound) & PRISM Noise Analysis ({args.noise_step} to {args.noise_max})...")
    
    all_images_analysis = {}
    
    for q_data in tqdm(qualified_images, desc="Processing Image Cohort"):
        name = q_data['name']
        orig_image_np_01 = q_data['orig_image_np_01']
        gt_mask_np = q_data['gt_mask_np']
        segments_slic = q_data['segments_slic']
        pred_mask = q_data['pred_mask']
        qualified_ic_sp, oc_sps_all = q_data['qualified_ic_sp'], q_data['oc_sps_all']

        final_analysis = {"image_info": {"name": name.rsplit('.', 1)[0]}}

        # --- 1. SOTA Pipeline (Run FIRST to Establish Truncation Boundary) ---
        full_roi = np.ones_like(gt_mask_np, dtype=bool)
        Af_c = calculate_iou_score(pred_mask, gt_mask_np, full_roi) / 100.0
        if Af_c == 0: Af_c = 1e-6 

        sota_results = {}
        sota_max_valid_levels = {}
        
        for noise_type in args.noise_types:
            sota_results[noise_type] = []
            max_level_for_this_noise = 0.0
            
            for noise_level in np.arange(args.noise_step, args.noise_max + 1e-5, args.noise_step):
                pert_img = perturb_global_image(orig_image_np_01, noise_type, noise_level)
                ssim_score = structural_similarity(orig_image_np_01, pert_img, data_range=1.0, channel_axis=-1)
                
                # --- SSIM BREAK LOGIC ---
                if ssim_score >= 0.70:
                    max_level_for_this_noise = noise_level # Record survival limit
                    pert_tensor = torch.from_numpy(pert_img).permute(2, 0, 1).float()
                    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    pert_tensor_normalized = normalize(pert_tensor).unsqueeze(0).to(device)

                    with torch.no_grad():
                        _, _, res = model_instance(pert_tensor_normalized)
                        res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        pert_mask = np.where(res >= 0.5, 1, 0)
                else:
                    logger.info(f"[SOTA] Image: {name}, Noise: {noise_type}, Level: {round(noise_level, 2)}, SSIM: {round(ssim_score, 4)} - Skipping Prediction (SSIM < 0.70)")
                    break
                
                Af_ps = calculate_iou_score(pert_mask, gt_mask_np, full_roi) / 100.0
                gamma_r = Af_ps / Af_c  
                
                sota_results[noise_type].append({'noise_level': round(noise_level, 2), 'gamma_r': gamma_r})
            
            sota_max_valid_levels[noise_type] = max_level_for_this_noise
                
        final_analysis['sota_analysis'] = sota_results

        # --- 2. PRISM Pipeline (Dynamically Capped via sota_max_valid_levels) ---
        res_ic = run_ranking_analysis('in_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, segments_slic, qualified_ic_sp, qualified_ic_sp, args.noise_types, noise_char_map, sota_max_valid_levels, '', [], save_visuals=False)
        final_analysis['in_context_analysis'] = res_ic
        
        num_oc_to_perturb = int(0.70 * len(oc_sps_all))
        out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
        res_oc = run_ranking_analysis('out_of_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, segments_slic, out_of_context_sp_to_perturb, qualified_ic_sp, args.noise_types, noise_char_map, sota_max_valid_levels, '', [], save_visuals=False)
        final_analysis['out_of_context_analysis'] = res_oc
        
        all_images_analysis[name] = final_analysis

    # =========================================================================
    # PHASE 3: ABLATION STUDY - RANK ENTROPY EVALUATION
    # =========================================================================
    logger.info("\n=== [PHASE 3] Rank Entropy Ablation: Strict Common Ground Evaluation ===")
    
    entropy_results = []
    img_names_ordered = [q['name'] for q in qualified_images]
    
    for noise_type in args.noise_types:
        min_valid_steps = 999 
        
        for img_name in img_names_ordered:
            data_img = all_images_analysis[img_name]
            sota_len = len(data_img['sota_analysis'][noise_type])
            ic_len = len(data_img['in_context_analysis'][noise_type]['level_data']) - 1
            oc_len = len(data_img['out_of_context_analysis'][noise_type]['level_data']) - 1
            min_valid_steps = min(min_valid_steps, sota_len, ic_len, oc_len)
            
        logger.info(f"[{noise_type.upper()}] Common Ground strictly established at {min_valid_steps} perturbation steps.")
        
        if min_valid_steps == 0:
            logger.warning(f"[{noise_type.upper()}] An image failed SSIM instantly. Cannot compute entropy for this noise.")
            continue
            
        sota_rank_matrix = {img: [] for img in img_names_ordered}
        prism_rank_matrix = {img: [] for img in img_names_ordered}
        
        for step_idx in range(min_valid_steps):
            sota_level_scores = []
            prism_level_scores = []
            
            for img_name in img_names_ordered:
                img_data = all_images_analysis[img_name]
                
                sota_gamma = img_data['sota_analysis'][noise_type][step_idx]['gamma_r']
                sota_level_scores.append(sota_gamma)
                
                ic_lvl_data = img_data['in_context_analysis'][noise_type]['level_data'][step_idx + 1]
                oc_lvl_data = img_data['out_of_context_analysis'][noise_type]['level_data'][step_idx + 1]
                
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
            # Dynamically adapt bin size to number of images
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
            logger.warning(f"Skipping statistical proof for {noise_type.upper()}: No valid common ground data found.")
            continue

        df_sub = df_entropy[df_entropy['Noise Type'] == noise_type].copy()
        
        sota_entropies = df_sub['SOTA Entropy'].values
        prism_entropies = df_sub['PRISM Entropy'].values
        
        # 1. Normality Test (Shapiro-Wilk)
        differences = sota_entropies - prism_entropies
        if np.allclose(sota_entropies, prism_entropies) or len(differences) < 3:
            shapiro_stat, shapiro_p = 1.0, 1.0
            is_normal = True
        else:
            shapiro_stat, shapiro_p = stats.shapiro(differences)
            is_normal = shapiro_p > 0.05
            
        # 2. Parametric: Paired T-Test
        if np.allclose(sota_entropies, prism_entropies):
            t_stat, t_p_val = 0.0, 1.0 
        else:
            t_stat, t_p_val = stats.ttest_rel(sota_entropies, prism_entropies, alternative='greater')
            
        # 3. Non-Parametric: Wilcoxon Signed-Rank Test
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
            
        logger.info(f"    -> Selected Test: {test_used} (based on Shapiro-Wilk result)")
        
        if final_p_val < 0.05:
            logger.info(f"    -> Conclusion: Null hypothesis rejected ({test_used}). PRISM is significantly more stable.")
            sig_marker = "***" if final_p_val < 0.001 else "**" if final_p_val < 0.01 else "*"
        else:
            logger.info(f"    -> Conclusion: Failed to reject null hypothesis at alpha=0.05 ({test_used}).")
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
        
    logger.info(f"\n--- Ablation Finished. Visuals and JSON Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransFuse Model Reliability Inference + Rank Ablation")
    parser.add_argument('--pth_path', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/TransFuse-19_best.pth')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    
    parser.add_argument('--output_dir', type=str, default='./sota_prism_ablation_results')
    parser.add_argument('--num_images', type=int, default=50, help="Strictly process exactly 20 qualified images for ablation study.")
    parser.add_argument('--noise_step', type=float, default=0.01, help="Step size for noise loop (0.1 for Ablation).")
    parser.add_argument('--noise_max', type=float, default=3.0, help="Maximum noise level for perturbation sweeps.")
    
    parser.add_argument('--superpixel_size', type=int, default=500)
    # parser.add_argument('--noise_types', nargs='+', default=['gaussian_blur'])
    parser.add_argument('--noise_types', nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    
    parser.add_argument('--image_file', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/img_sel.txt', help="Text file containing the frozen cohort of image names (without extensions).")

    
    args = parser.parse_args()
    main(args)
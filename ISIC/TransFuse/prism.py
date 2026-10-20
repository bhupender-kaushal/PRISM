import os, sys
import json
import argparse
import logging
import random
import time
import itertools
import math
import gc
from math import exp, ceil

import numpy as np
import pandas as pd
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
from skimage.color import gray2rgb, rgb2gray
import torch.utils.data as data

from lib.TransFuse import TransFuse_S

random_seed = 42
random.seed(random_seed)
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

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

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    if superpixels_segments.shape != gt_mask.shape:
        from PIL import Image
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

# --- RGB CHANNEL-WISE FIX: Noise Application ---
# Noise strength per unit noise level, tuned for ISIC. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'blur_sigma': 50.0,            # sigma = 50 * level
    'rician_sigma': 0.1,           # std = 0.1 * level
}

def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """
    Applies strict channel-wise perturbations over RGB channels for the selected superpixels.
    """
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0

    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask):
        return perturbed_image, noise_char_value

    H, W, C = perturbed_image.shape

    if noise_type == 'gaussian':
        noise_char_value = NOISE_PARAMS['gaussian_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] += noise[region_mask]
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, NOISE_PARAMS['salt_pepper_density'] * noise_level)
        rows, cols = np.where(region_mask)
        num_pixels = len(rows)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            for c in range(C):
                indices = np.random.choice(num_pixels, num_noise, replace=False)
                salt_pepper_vals = np.random.choice([0.0, 1.0], num_noise)
                perturbed_image[rows[indices], cols[indices], c] = salt_pepper_vals

    elif noise_type == 'poisson':
        target_std = NOISE_PARAMS['gaussian_sigma'] * noise_level          # Poisson std on mid-grey matches the Gaussian std
        scale_factor = 0.5 / max(target_std, 1e-6) ** 2
        noise_char_value = target_std
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = np.clip(channel_data[region_mask], 0.0, 1.0)
            noisy_pixels = np.random.poisson(region_pixels * scale_factor) / scale_factor
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'speckle':
        noise_char_value = NOISE_PARAMS['speckle_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] *= (1.0 + noise[region_mask])
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = NOISE_PARAMS['blur_sigma'] * noise_level
        for c in range(C):
            blurred_channel = gaussian_filter(original_image_scaled[:, :, c], sigma=noise_char_value)
            perturbed_image[region_mask, c] = blurred_channel[region_mask]
        
    elif noise_type == 'rician':
        noise_char_value = NOISE_PARAMS['rician_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = channel_data[region_mask]
            n1 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            n2 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            noisy_pixels = np.sqrt(np.square(region_pixels + n1) + np.square(n2))
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

def calculate_eigencam(model, target_layer_name, image_tensor, image_size):
    feature_maps = []
    logger = logging.getLogger('base')

    try:
        target_layer = dict(model.named_modules())[target_layer_name]
    except KeyError:
        return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output):
        feature_maps.append(output)

    hook = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(image_tensor)

    hook.remove() 

    if not feature_maps:
        return np.zeros(image_size, dtype=np.float32)

    features = feature_maps[0].squeeze(0)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    
    pca = PCA(n_components=1).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    
    saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
    return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()

def visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    import matplotlib.pyplot as plt
    if pred_mask.shape != gt_mask.shape:
        pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        pred_mask_resized = np.array(pred_mask_img.resize(gt_mask.shape[::-1], Image.NEAREST))
    else:
        pred_mask_resized = pred_mask

    display_image_rgb = gray2rgb(orig_image_01) if orig_image_01.ndim == 2 else orig_image_01
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))
    
    axes[0].imshow(display_image_rgb); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_mask_resized, cmap='gray'); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
    axes[3].imshow(mark_boundaries(display_image_rgb, all_sp_map)); axes[3].set_title(f"All Superpixels"); axes[3].axis('off')
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map)); axes[4].set_title("In-Context Superpixels"); axes[4].axis('off')
    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map)); axes[5].set_title("Out-of-Context Superpixels"); axes[5].axis('off')
    axes[6].imshow(display_image_rgb); im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1); axes[6].set_title("EigenCAM"); axes[6].axis('off')

    plt.colorbar(im, ax=axes[6])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# --------------------------------------------------------------------------
# --- 3. Core Step-Wise Robustness Analysis (Upgraded with Global & Micro IoU)
# --------------------------------------------------------------------------
def run_ranking_analysis(
    analysis_type, args, device, model,
    orig_image_np_01, pred_mask, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, current_image_output_dir,
    initial_probs, ic_sps_all, oc_sps_all,
    save_visuals=False, alpha=1.0, beta=1.0
):
    logger = logging.getLogger('base')
    analysis_results = {
        "rank_distribution": [],
        "level_data_per_noise": {}
    }
    
    max_id = int(segments_slic.max()) + 1 

    # --- Precalculate Superpixel Areas ---
    sp_areas = {int(sp_id): int(np.sum(segments_slic == sp_id)) for sp_id in superpixels_to_measure}

    # --- BG-IoU SPILLOVER BASELINE CALCULATION ---
    bg_gt = (gt_mask == 0)
    bg_pred_init = (pred_mask == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0
    
    # --- Global IoU Baseline ---
    base_intersect = np.sum((pred_mask == 1) & (gt_mask > 0))
    base_union = np.sum((pred_mask == 1) | (gt_mask > 0))
    base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0

    # --- Micro IoU Baseline Arrays ---
    intersect_mask_0 = (pred_mask == 1) & (gt_mask > 0)
    union_mask_0 = (pred_mask == 1) | (gt_mask > 0)
    int_counts_0 = np.bincount(segments_slic[intersect_mask_0].ravel(), minlength=max_id)
    uni_counts_0 = np.bincount(segments_slic[union_mask_0].ravel(), minlength=max_id)

    # --- Step 0 Baseline Probability Ranking ---
    initial_sp_data = []
    for sp_id in superpixels_to_measure:
        mask = (segments_slic == sp_id)
        mean_p = float(np.mean(initial_probs[mask]))
        
        u_val = uni_counts_0[sp_id]
        i_val = int_counts_0[sp_id]
        sp_iou = (i_val / u_val * 100.0) if u_val > 0 else 100.0
        
        # Domain tagging for Objective 2 (Violins & Rank Densities)
        if sp_id in ic_sps_all: gt_domain = "IC"
        elif sp_id in oc_sps_all: gt_domain = "OC"
        else: gt_domain = "Mixed"

        initial_sp_data.append({
            "sp_id": int(sp_id), 
            "mean_prob": mean_p,
            "iou": round(float(sp_iou), 2),
            "area": sp_areas[sp_id],
            "gt_domain": gt_domain
        })
    
    # Sort firmly by Probability (Replaces IoU sorting)
    sorted_initial_sp_data = sorted(initial_sp_data, key=lambda x: (x['mean_prob'], x['area']), reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): 
        sp_data['rank'] = rank
        
    initial_ranks_map = {item['sp_id']: item['rank'] for item in sorted_initial_sp_data}
    analysis_results["rank_distribution"] = sorted_initial_sp_data

    N_measured = len(superpixels_to_measure)
    max_possible_jumps = math.floor((N_measured**2) / 2.0) if N_measured > 1 else 1.0

    # Base utility relies on probability ranking baseline
    base_raw_mean_prob = np.mean([sp['mean_prob'] for sp in sorted_initial_sp_data]) if sorted_initial_sp_data else 0.0
    base_utility_e0 = base_global_iou          # utility is IoU-based; probability is only used to rank superpixels

    for noise_type in noise_types:
        if save_visuals:
            noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
            os.makedirs(noise_specific_dir, exist_ok=True)

        # Step 0 Baseline Entry
        level_0_entry = {
            "step_idx": 0,
            "noise_level": 0.0,
            "ssim": 1.0,
            "global_iou": round(float(base_global_iou), 2),
            "mean_raw_prob": round(float(base_raw_mean_prob * 100.0), 2),
            "rank_perplexity": 0.0,
            "instant_spillover_ratio": 0.0,
            "discounted_utility": round(float(base_utility_e0), 2),
            "superpixels": sorted_initial_sp_data
        }
        level_data_list = [level_0_entry]
        
        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0
        peak_rp = 0.0

        noise_levels_arr = np.round(np.arange(1, int(round(args.max_level / args.noise_step)) + 1) * args.noise_step, 4)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False, position=1)

        for step_idx, noise_level in enumerate(pbar):
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})

            perturbed_image_np_scaled, _ = perturb_superpixels_with_level(orig_image_np_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_np_01, perturbed_image_np_scaled, data_range=1.0, channel_axis=-1)

            perturbed_tensor = torch.from_numpy(perturbed_image_np_scaled).permute(2, 0, 1).float()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            perturbed_tensor_normalized = normalize(perturbed_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, res = model(perturbed_tensor_normalized)
                res = F.upsample(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                perturbed_probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
                perturbed_pred_mask = np.where(perturbed_probs >= 0.5, 1, 0)

            # --- IMAGE SAVING AT 0.5 INTERVALS ---
            if save_visuals and int(round(noise_level, 2) * 100) % 50 == 0:
                Image.fromarray((perturbed_image_np_scaled * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                Image.fromarray((perturbed_pred_mask * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                
                img_rgb = gray2rgb(perturbed_image_np_scaled) if perturbed_image_np_scaled.ndim == 2 else perturbed_image_np_scaled
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask, color=(1, 0, 0), mode='thick')
                Image.fromarray((overlay_image * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            # --- Global IoU at Step k ---
            intersect_k = np.sum((perturbed_pred_mask == 1) & (gt_mask > 0))
            union_k = np.sum((perturbed_pred_mask == 1) | (gt_mask > 0))
            global_iou_k = (intersect_k / union_k * 100.0) if union_k > 0 else 100.0

            # --- Instant Background Spillover Ratio ---
            bg_pred_pert = (perturbed_pred_mask == 0)
            intersection_bg_pert = np.sum(bg_gt & bg_pred_pert)
            union_bg_pert = np.sum(bg_gt | bg_pred_pert)
            iou_bg_pert = (intersection_bg_pert / union_bg_pert * 100.0) if union_bg_pert > 0 else 100.0
            
            delta_iou_bg = iou_bg_base - iou_bg_pert
            if delta_iou_bg > 0:
                spillover_ratio_k = delta_iou_bg / 100.0
                cumulative_spillover_ratio += spillover_ratio_k
                conditional_bleed_count += 1
            else:
                spillover_ratio_k = 0.0

            # --- Micro IoU Vectors at Step k ---
            intersect_mask_k = (perturbed_pred_mask == 1) & (gt_mask > 0)
            union_mask_k = (perturbed_pred_mask == 1) | (gt_mask > 0)
            int_counts_k = np.bincount(segments_slic[intersect_mask_k].ravel(), minlength=max_id)
            uni_counts_k = np.bincount(segments_slic[union_mask_k].ravel(), minlength=max_id)

            # --- Dynamic Step-Wise Probability & IoU Extraction ---
            current_level_sp_data = []
            for sp_id in superpixels_to_measure:
                mask = (segments_slic == sp_id)
                mean_p_k = float(np.mean(perturbed_probs[mask]))
                
                u_val = uni_counts_k[sp_id]
                i_val = int_counts_k[sp_id]
                sp_iou = (i_val / u_val * 100.0) if u_val > 0 else 100.0

                current_level_sp_data.append({
                    "sp_id": int(sp_id), 
                    "mean_prob": mean_p_k,
                    "iou": round(float(sp_iou), 2),
                    "area": sp_areas[sp_id]
                })

            # SORT DYNAMICALLY BY PROBABILITY TO MEASURE RANK JUMPS
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: (x['mean_prob'], x['area']), reverse=True)          
            for rank, sp_data in enumerate(sorted_superpixels, 1): 
                sp_data['rank'] = rank
            
            current_ranks_map = {data['sp_id']: data['rank'] for data in sorted_superpixels}

            # --- Normalized Rank Perplexity at Step k ---
            total_rank_jumps = sum(abs(current_ranks_map[sp_id] - initial_ranks_map[sp_id]) for sp_id in superpixels_to_measure)
            norm_rp_k = total_rank_jumps / max_possible_jumps if max_possible_jumps > 0 else 0.0
            peak_rp = max(peak_rp, norm_rp_k)

            # --- Step-Wise Discounted Utility (E_k) ---
            raw_prob_mean_k = np.mean([sp['mean_prob'] for sp in sorted_superpixels]) if sorted_superpixels else 0.0
            e_k = global_iou_k * (1.0 - alpha * min(1.0, norm_rp_k)) * max(0.0, 1.0 - beta * spillover_ratio_k)

            level_data_list.append({
                "step_idx": step_idx + 1,
                "noise_level": round(float(noise_level), 3),
                "ssim": round(float(ssim_score), 4),
                "global_iou": round(float(global_iou_k), 2),
                "mean_raw_prob": round(float(raw_prob_mean_k * 100.0), 2),
                "rank_perplexity": round(float(norm_rp_k), 4),
                "instant_spillover_ratio": round(float(spillover_ratio_k), 4),
                "discounted_utility": round(float(e_k), 2),
                "superpixels": sorted_superpixels
            })
            
        # Aggressive memory cleanup to prevent VRAM fragmentation sluggishness
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_spillover = (cumulative_spillover_ratio / conditional_bleed_count) if conditional_bleed_count > 0 else 0.0
        terminal_ssim = level_data_list[-1]['ssim'] if level_data_list else 1.0
        mean_discounted_utility = np.mean([entry['discounted_utility'] for entry in level_data_list])

        analysis_results["level_data_per_noise"][noise_type] = {
            "base_utility_e0": round(float(base_utility_e0), 2),
            "mean_step_discounted_utility": round(float(mean_discounted_utility), 2),
            "terminal_ssim": round(float(terminal_ssim), 4),
            "max_rank_perplexity": round(float(peak_rp), 4),
            "spillover_metrics": { 
                "conditional_average_spillover": round(float(avg_spillover), 4),
                "bleed_count": int(conditional_bleed_count)
            },
            "level_data": level_data_list
        }
    return analysis_results
# --------------------------------------------------------------------------
# --- 4. Main Two-Phase Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM IC/OC Continuous Noise Engine - TransFuse")
    parser.add_argument('--pth_path', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/TransFuse/TransFuse-19_best.pth')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/cluster/home/bhupenk/bhupender/datasets/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--target_layer_name', type=str, default='up_c')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/results/transfuse/prism_out_icoc')
    parser.add_argument('--txtfile', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/img_sel.txt', help="Path to the selected images text file containing qualified image names.")
    parser.add_argument('--num_images', type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument('--num_dataset_runs', type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument('--noise_step', type=float, default=0.1, help="Step size for noise loop.")
    parser.add_argument('--max_level', type=float, default=5.0, help="Highest noise level of the grid (inclusive).")
    parser.add_argument('--superpixel_size', type=int, default=150)
    parser.add_argument('--noise_types', nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--perturb_ratio', type=float, default=1.0)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading TransFuse model...")
    model_instance = TransFuse_S().to(device)
    model_instance.load_state_dict(torch.load(args.pth_path, map_location=device))
    model_instance.eval()

    logger.info("Loading ISIC dataset...")
    image_root = os.path.join(args.dataroot_val, 'images/')
    gt_root = os.path.join(args.dataroot_val, 'masks/')
    test_dataset = PngDataset(image_root, gt_root, args.testsize)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    noise_char_map = {noise: f'{noise}_strength' for noise in args.noise_types}

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Strict Anatomical Qualifier Scan...")
    qualified_images = []
    
    if not os.path.exists(args.txtfile):
        logger.error(f"FATAL: Frozen cohort file '{args.txtfile}' not found.")
        sys.exit()
    
    with open(args.txtfile, 'r') as f:
        frozen_image_names = set([line.strip() for line in f.readlines()])
    
    for i, (image, gt, name_list) in enumerate(tqdm(test_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break
        
        name = name_list[0]
        
        if name not in frozen_image_names:
            continue
            
        image_tensor = image.to(device)
        _, _, h, w = image_tensor.shape

        gt_mask_np = np.asarray(gt, np.float32).squeeze(0).squeeze(0)
        pil_gt = Image.fromarray(gt_mask_np.astype(np.uint8))
        gt_mask_np = np.array(pil_gt.resize((w, h), Image.NEAREST)).astype(np.float32)
        
        gt_mask_np /= (gt_mask_np.max() + 1e-8)

        # TransFuse specific fix: unpack three returns
        with torch.no_grad():
            _, _, res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            initial_probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(initial_probs >= 0.5, 1, 0)
        
        full_image_roi = np.ones_like(gt_mask_np, dtype=bool)

        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)

        segments_slic = slic(orig_image_np_01, n_segments=args.superpixel_size, compactness=10, sigma=1)
        
        # Absolute Anchoring
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask_np, ic_threshold=1.0, oc_threshold=0.0)
        
        if not ic_sps_all: continue

        qualified_images.append({
            'name': name, 'image_tensor': image_tensor, 'orig_image_np_01': orig_image_np_01, 
            'gt_mask_np': gt_mask_np, 'segments_slic': segments_slic, 'pred_mask': pred_mask, 
            'initial_probs': initial_probs, 'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(gt_mask_np, pred_mask)
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: MULTI-RUN ROBUSTNESS INFERENCE (IC / OC)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Iterations...")

    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0, leave=True):
            name = q_data['name']
            orig_image_np_01 = q_data['orig_image_np_01']
            image_tensor = q_data['image_tensor']
            gt_mask_np = q_data['gt_mask_np']
            segments_slic = q_data['segments_slic']
            pred_mask = q_data['pred_mask']
            initial_probs = q_data['initial_probs']
            ic_sps_all = q_data['ic_sps_all']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, name.rsplit('.', 1)[0])
            os.makedirs(current_image_output_dir, exist_ok=True)
            final_analysis = {"image_info": {"name": name.rsplit('.', 1)[0], "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)}}

            # 1. In-Context Analysis
            res_ic = run_ranking_analysis(
                'in_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, 
                segments_slic, ic_sps_all, ic_sps_all, args.noise_types, noise_char_map, 
                current_image_output_dir, initial_probs, ic_sps_all, oc_sps_all, save_visuals=(run_idx == 0)
            )
            final_analysis['in_context_analysis'] = res_ic
            
            # 2. Out-of-Context Analysis
            num_oc_to_perturb = int(0.50 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments_slic)
                for sp_id in ic_sps_all: in_context_map[segments_slic == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments_slic)
                for sp_id in oc_sps_all: out_of_context_map[segments_slic == sp_id] = sp_id
                
                eigencam_heatmap = calculate_eigencam(model_instance, args.target_layer_name, image_tensor, (orig_image_np_01.shape[0], orig_image_np_01.shape[1]))
                
                gt_mask_for_sp = (np.array(Image.fromarray((gt_mask_np * 255).astype(np.uint8)).resize(segments_slic.shape[::-1], Image.NEAREST)) > 127).astype(np.uint8) if gt_mask_np.shape != segments_slic.shape else gt_mask_np.astype(np.uint8)
                
                visualize_segmentation_comparison(orig_image_np_01, gt_mask_for_sp, pred_mask, segments_slic, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, name.rsplit('.', 1)[0] + "_overview.png"))

            res_oc = run_ranking_analysis(
                'out_of_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, 
                segments_slic, out_of_context_sp_to_perturb, ic_sps_all, args.noise_types, noise_char_map, 
                current_image_output_dir, initial_probs, ic_sps_all, oc_sps_all, save_visuals=(run_idx == 0)
            )
            final_analysis['out_of_context_analysis'] = res_oc
            
            with open(os.path.join(current_image_output_dir, f"{name.rsplit('.', 1)[0]}_analysis.json"), 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
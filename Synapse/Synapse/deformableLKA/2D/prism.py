import os
import argparse
import json
import logging
import random
import math
import gc
from abc import ABC, abstractmethod
from math import exp
import sys

import numpy as np
import pandas as pd
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

import torch
import torch.nn.functional as F
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

# --- GRAYSCALE SAFE FIX: Noise Application ---
def perturb_image_slice(image_01, segments, sp_to_perturb, noise_type, noise_level):
    """
    Applies mathematically strict 2D perturbations (Grayscale safe). 
    No channel loops are used.
    """
    perturbed_image = np.copy(image_01)
    
    region_mask = np.isin(segments, sp_to_perturb)
    if not np.any(region_mask):
        return perturbed_image
        
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] += noise[region_mask]
        
    elif noise_type == 'salt_pepper' or noise_type == 'salt_pepper':
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
        noisy_region = np.random.poisson(np.clip(region, 0, 1) * 100 * sigma) / (100 * sigma)
        perturbed_image[region_mask] = noisy_region
        
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1.0 + noise[region_mask])
        
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=10 * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
        
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.05 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.05 * noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

def predict_slice_maxvit(model, image_slice_01, patch_size, target_class_index, device='cuda'):
    """MaxViT-specific inference routing extracting multiclass target probabilities."""
    h, w = image_slice_01.shape
    slice_rescaled = zoom(image_slice_01, (patch_size / h, patch_size / w), order=3) if (h != patch_size or w != patch_size) else image_slice_01
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = x_transforms(slice_rescaled).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs_tensor = torch.softmax(outputs, dim=1)
        
        out_mask = torch.argmax(probs_tensor, dim=1).squeeze(0).cpu().numpy()
        target_probs = probs_tensor[0, target_class_index, :, :].cpu().numpy()
        
    if h != patch_size or w != patch_size:
        out_mask = zoom(out_mask, (h / patch_size, w / patch_size), order=0)
        target_probs = zoom(target_probs, (h / patch_size, w / patch_size), order=1)
        
    pred_mask_bool = (out_mask == target_class_index)
    return pred_mask_bool, target_probs, input_tensor

def get_target_layer(model, layer_name):
    try:
        return model.get_submodule(layer_name)
    except AttributeError:
        for name, module in model.named_modules():
            if layer_name in name:
                return module
        raise AttributeError(f"Could not find target layer '{layer_name}'.")

def calculate_eigencam(model, input_tensor, target_layer_names, output_size):
    if not isinstance(target_layer_names, list):
        target_layer_names = [target_layer_names]
    all_saliency_maps = []
    for layer_name in target_layer_names:
        feature_maps = []
        try:
            target_layer = get_target_layer(model, layer_name)
        except AttributeError as e:
            continue
        def hook_fn(module, input, output):
            if output.dim() == 3 and output.shape[0] == input_tensor.shape[0]: 
                B, N, C = output.shape
                H = W = int(np.sqrt(N))
                feature_maps.append(output.permute(0, 2, 1).view(B, C, H, W))
            else: 
                feature_maps.append(output)
        hook = target_layer.register_forward_hook(hook_fn)
        _ = model(input_tensor)
        hook.remove()
        if not feature_maps: continue
        features = feature_maps[0].squeeze(0).to(torch.float32)
        features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().detach().numpy()
        pca = PCA(n_components=1).fit(features_flat)
        saliency_map = (features.cpu().detach().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
        saliency_map = np.maximum(saliency_map, 0)
        saliency_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float()
        resized_saliency = F.interpolate(saliency_tensor, size=output_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        all_saliency_maps.append(resized_saliency)
    if not all_saliency_maps: return np.zeros(output_size, dtype=np.float32)
    avg_saliency_map = np.mean(all_saliency_maps, axis=0)
    s_min, s_max = avg_saliency_map.min(), avg_saliency_map.max()
    return (avg_saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(avg_saliency_map)

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb, cmap='gray'); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
    axes[3].imshow(mark_boundaries(display_image_rgb, all_sp_map)); axes[3].set_title("All Superpixels"); axes[3].axis('off')
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map)); axes[4].set_title("In-Context SPs"); axes[4].axis('off')
    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map)); axes[5].set_title("Out-of-Context SPs"); axes[5].axis('off')
    im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1.0); axes[6].set_title("EigenCAM"); axes[6].axis('off')
    plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# --------------------------------------------------------------------------
# --- 3. Core Step-Wise Robustness Analysis (Upgraded for Mean Probability)
# --------------------------------------------------------------------------
def run_ranking_analysis(
    analysis_type, model, device, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, patch_size, target_class_index, current_image_output_dir, 
    initial_probs, ic_sps_all, oc_sps_all,
    noise_step=0.05, save_visuals=False, alpha=1.0, beta=1.0
):
    logger = logging.getLogger('base')
    analysis_results = {
        "rank_distribution": [],
        "level_data_per_noise": {}
    }
    
    # Safely handle 1D numpy arrays and lists
    if len(superpixels_to_measure) == 0: 
        return {}
        
    max_id = int(np.max(segments)) + 1
    
    # Ensure IDs are extracted as standard integers to prevent numpy truth-value clashes
    valid_sp_to_measure = [int(sp) for sp in superpixels_to_measure if int(sp) < max_id]
    sp_areas = {sp_id: int(np.sum(segments == sp_id)) for sp_id in valid_sp_to_measure}

    # --- BG-IoU SPILLOVER BASELINE CALCULATION ---
    bg_gt = (target_class_gt == 0)
    bg_pred_init = (initial_pred_mask_bool == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0

    # --- Global IoU Baseline ---
    base_intersect = np.sum((initial_pred_mask_bool == 1) & (target_class_gt == 1))
    base_union = np.sum((initial_pred_mask_bool == 1) | (target_class_gt == 1))
    base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0

    # --- Micro IoU Baseline Arrays ---
    intersect_mask_0 = (initial_pred_mask_bool == 1) & (target_class_gt == 1)
    union_mask_0 = (initial_pred_mask_bool == 1) | (target_class_gt == 1)
    int_counts_0 = np.bincount(segments[intersect_mask_0].ravel(), minlength=max_id)
    uni_counts_0 = np.bincount(segments[union_mask_0].ravel(), minlength=max_id)

    # --- Step 0 Baseline Probability Ranking ---
    initial_sp_data = []
    for sp_id in valid_sp_to_measure:
        mask = (segments == sp_id)
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
    
    # Sort firmly by Probability
    sorted_initial_sp_data = sorted(initial_sp_data, key=lambda x: (x['mean_prob'], x['area']), reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): 
        sp_data['rank'] = rank
        
    initial_ranks_map = {item['sp_id']: item['rank'] for item in sorted_initial_sp_data}
    analysis_results["rank_distribution"] = sorted_initial_sp_data

    N_measured = len(valid_sp_to_measure)
    max_possible_jumps = math.floor((N_measured**2) / 2.0) if N_measured > 1 else 1.0

    # Base utility relies on probability ranking baseline
    base_raw_mean_prob = np.mean([sp['mean_prob'] for sp in sorted_initial_sp_data]) if sorted_initial_sp_data else 0.0
    base_utility_e0 = base_raw_mean_prob * 100.0

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

        noise_levels_arr = np.arange(noise_step, 3.1, noise_step)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False, position=1)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            
            if ssim_score < 0.70:
                logger.info(f"\n[SSIM TRIGGER] SSIM ({ssim_score:.4f}) dropped below 0.70 at {noise_type.upper()} level {noise_level:.2f}. Halting.")
                break

            perturbed_pred_mask_bool, perturbed_probs, _ = predict_slice_maxvit(model, perturbed_img_gray, patch_size, target_class_index, device)

            if save_visuals and int(round(noise_level, 2) * 100) % 50 == 0:
                Image.fromarray((perturbed_img_gray * 255).astype(np.uint8)).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                mask_to_save = (perturbed_pred_mask_bool * 255).astype(np.uint8)
                Image.fromarray(mask_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                img_rgb = gray2rgb(perturbed_img_gray)
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask_bool, color=(1, 0, 0), mode='thick')
                overlay_to_save = (overlay_image * 255).astype(np.uint8)
                Image.fromarray(overlay_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            # --- Global IoU at Step k ---
            intersect_k = np.sum((perturbed_pred_mask_bool == 1) & (target_class_gt == 1))
            union_k = np.sum((perturbed_pred_mask_bool == 1) | (target_class_gt == 1))
            global_iou_k = (intersect_k / union_k * 100.0) if union_k > 0 else 100.0

            # --- BG-IoU SPILLOVER INSTANT TRACKING ---
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

            # --- Micro IoU Vectors at Step k ---
            intersect_mask_k = (perturbed_pred_mask_bool == 1) & (target_class_gt == 1)
            union_mask_k = (perturbed_pred_mask_bool == 1) | (target_class_gt == 1)
            int_counts_k = np.bincount(segments[intersect_mask_k].ravel(), minlength=max_id)
            uni_counts_k = np.bincount(segments[union_mask_k].ravel(), minlength=max_id)

            # --- Dynamic Step-Wise Probability & IoU Extraction ---
            current_level_sp_data = []
            for sp_id in valid_sp_to_measure:
                mask = (segments == sp_id)
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
            total_rank_jumps = sum(abs(current_ranks_map[sp_id] - initial_ranks_map[sp_id]) for sp_id in valid_sp_to_measure)
            norm_rp_k = total_rank_jumps / max_possible_jumps if max_possible_jumps > 0 else 0.0
            peak_rp = max(peak_rp, norm_rp_k)

            # --- Step-Wise Discounted Utility (E_k) ---
            raw_prob_mean_k = np.mean([sp['mean_prob'] for sp in sorted_superpixels]) if sorted_superpixels else 0.0
            e_k = raw_prob_mean_k * (1.0 - alpha * min(1.0, norm_rp_k)) * max(0.0, 1.0 - beta * spillover_ratio_at_j) * 100.0

            level_data_list.append({
                "step_idx": step_idx + 1,
                "noise_level": round(float(noise_level), 3),
                "ssim": round(float(ssim_score), 4),
                "global_iou": round(float(global_iou_k), 2),
                "mean_raw_prob": round(float(raw_prob_mean_k * 100.0), 2),
                "rank_perplexity": round(float(norm_rp_k), 4),
                "instant_spillover_ratio": round(float(spillover_ratio_at_j), 4),
                "discounted_utility": round(float(e_k), 2),
                "superpixels": sorted_superpixels
            })
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        avg_spillover = cumulative_spillover_ratio / conditional_bleed_count if conditional_bleed_count > 0 else 0.0
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
# 4. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()]
    )
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
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    for sample_info in tqdm(candidate_samples, desc="Qualifying Slices"):
        if len(qualified_images) >= args.num_images: break

        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        volume_data = db_test[vol_idx]
        image_slice, label_slice = volume_data['image'][slice_idx, :, :], volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"
        
        if case_name not in frozen_cohort:
            continue
        
        image_slice_01 = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        
        # Initial forward pass extracting specific probabilities
        initial_pred_mask_bool, initial_probs, input_tensor = predict_slice_maxvit(net, image_slice_01, args.img_size, args.target_class_index, device)
        target_class_gt = (label_slice == args.target_class_index).astype(np.uint8)
        
        segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        # Absolute Anchoring (No prediction threshold filters used)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=1.0, oc_threshold=0.0)
        
        if not ic_sps_all: continue

        qualified_images.append({
            'case_name': case_name, 'image_slice_01': image_slice_01, 'input_tensor': input_tensor,
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'initial_probs': initial_probs,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool, target_class_gt)
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: MULTI-RUN ROBUSTNESS INFERENCE (WITH GLOBAL SOTA)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Iterations...")

    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0, leave=True):
            case_name = q_data['case_name']
            image_slice_01 = q_data['image_slice_01']
            input_tensor = q_data['input_tensor']
            target_class_gt = q_data['target_class_gt']
            segments = q_data['segments']
            initial_pred_mask_bool = q_data['initial_pred_mask_bool']
            initial_probs = q_data['initial_probs']
            ic_sps_all = q_data['ic_sps_all']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, case_name)
            os.makedirs(current_image_output_dir, exist_ok=True)
            
            final_analysis = {
                "image_info": {
                    "name": case_name, 
                    "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)
                }
            }

# Extract ALL unique superpixels for the pure SOTA baseline
            all_sps = np.unique(segments)
            all_sps = all_sps[all_sps != 0]

            # 1. Global SOTA Analysis
            res_global = run_ranking_analysis(
                analysis_type='global_sota', 
                model=net, 
                device=device, 
                orig_image_01=image_slice_01, 
                segments=segments, 
                target_class_gt=target_class_gt, 
                superpixels_to_perturb=all_sps, 
                superpixels_to_measure=all_sps, 
                initial_pred_mask_bool=initial_pred_mask_bool,
                noise_types=args.noise_types, 
                patch_size=args.img_size, 
                target_class_index=args.target_class_index, 
                current_image_output_dir=current_image_output_dir, 
                initial_probs=initial_probs, 
                ic_sps_all=ic_sps_all, 
                oc_sps_all=oc_sps_all, 
                noise_step=args.noise_step, 
                save_visuals=(run_idx == 0)
            )
            final_analysis['global_sota_analysis'] = res_global

            # 2. In-Context Analysis
            res_ic = run_ranking_analysis(
                analysis_type='in_context', 
                model=net, 
                device=device, 
                orig_image_01=image_slice_01, 
                segments=segments, 
                target_class_gt=target_class_gt, 
                superpixels_to_perturb=ic_sps_all, 
                superpixels_to_measure=ic_sps_all, 
                initial_pred_mask_bool=initial_pred_mask_bool,
                noise_types=args.noise_types, 
                patch_size=args.img_size, 
                target_class_index=args.target_class_index, 
                current_image_output_dir=current_image_output_dir, 
                initial_probs=initial_probs, 
                ic_sps_all=ic_sps_all, 
                oc_sps_all=oc_sps_all, 
                noise_step=args.noise_step, 
                save_visuals=(run_idx == 0)
            )
            final_analysis['in_context_analysis'] = res_ic

            # 3. Out-of-Context Analysis
            num_oc_to_perturb = int(0.70 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments)
                for sp_id in ic_sps_all: in_context_map[segments == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments)
                for sp_id in oc_sps_all: out_of_context_map[segments == sp_id] = sp_id
                eigencam_heatmap = calculate_eigencam(net, input_tensor, args.target_layer_names, (args.img_size, args.img_size))
                visualize_segmentation_comparison(image_slice_01, target_class_gt, initial_pred_mask_bool, segments, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, "overview.png"))

            res_oc = run_ranking_analysis(
                analysis_type='out_of_context', 
                model=net, 
                device=device, 
                orig_image_01=image_slice_01, 
                segments=segments, 
                target_class_gt=target_class_gt, 
                superpixels_to_perturb=out_of_context_sp_to_perturb, 
                superpixels_to_measure=ic_sps_all, 
                initial_pred_mask_bool=initial_pred_mask_bool,
                noise_types=args.noise_types, 
                patch_size=args.img_size, 
                target_class_index=args.target_class_index, 
                current_image_output_dir=current_image_output_dir, 
                initial_probs=initial_probs, 
                ic_sps_all=ic_sps_all, 
                oc_sps_all=oc_sps_all, 
                noise_step=args.noise_step, 
                save_visuals=(run_idx == 0)
            )
            final_analysis['out_of_context_analysis'] = res_oc

            with open(os.path.join(current_image_output_dir, f"{case_name}_analysis.json"), 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxViT Model Reliability Inference + Rank Ablation")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--model_path", type=str, default="./MaxViT_deform_LKA.pth")
    parser.add_argument("--output_dir", type=str, default="./prism_out_original")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int,default=8)
    parser.add_argument("--img_cohort", type=str, default="/bhupender/2025/Mouse_other/Synapse/img_sel.txt", help="pre compiled image list.")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_dataset_runs", type=int, default=10, help="Number of dataset iterations for variance calculation.")
    parser.add_argument("--noise_step", type=float, default=0.05, help="Step size for noise loop.")
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--target_layer_names', nargs='+', default=['last_layer'])

    args = parser.parse_args()
    main(args)
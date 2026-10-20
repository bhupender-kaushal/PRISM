import os
import time
import argparse
import logging
import random
import json
import math
import gc
from math import exp
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb, rgb2gray
from skimage.metrics import structural_similarity
from skimage import color
from skimage.filters import sobel
from skimage.util import regular_grid

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

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
# Noise strength per unit noise level, tuned for XCAD. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'blur_sigma': 10.0,            # sigma = 10 * level
    'rician_sigma': 0.1,           # std = 0.1 * level
}

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
        noise = np.random.normal(0, NOISE_PARAMS['gaussian_sigma'] * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] += noise[region_mask]
        
    elif noise_type == 's&p' or noise_type == 'salt_pepper':
        noise_char_value = min(0.5, NOISE_PARAMS['salt_pepper_density'] * noise_level)
        num_pixels = np.sum(region_mask)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            rows, cols = np.where(region_mask)
            indices = np.random.choice(len(rows), num_noise, replace=False)
            perturbed_image[rows[indices], cols[indices]] = np.random.choice([0.0, 1.0], num_noise)
            
    elif noise_type == 'poisson':
        target_std = NOISE_PARAMS['gaussian_sigma'] * noise_level          # Poisson std on mid-grey matches the Gaussian std
        photons = 0.5 / max(target_std, 1e-6) ** 2
        region = perturbed_image[region_mask]
        noisy_region = np.random.poisson(np.clip(region, 0, 1) * photons) / photons
        perturbed_image[region_mask] = noisy_region
        
    elif noise_type == 'speckle':
        noise = np.random.normal(0, NOISE_PARAMS['speckle_sigma'] * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1.0 + noise[region_mask])
        
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=NOISE_PARAMS['blur_sigma'] * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
        
    elif noise_type == 'rician':
        n1 = np.random.normal(0, NOISE_PARAMS['rician_sigma'] * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, NOISE_PARAMS['rician_sigma'] * noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

def predict_slice_diffusion(model_diffusion, img_01, val_data_template, device):
    """Inference router mapping [0, 1] inputs to [-1, 1] tensors and extracting continuous probabilities."""
    if img_01.ndim == 2:
        img_tensor = torch.from_numpy(img_01).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        img_tensor = torch.from_numpy(img_01).permute(2,0,1).unsqueeze(0).float().to(device)
        
    img_m11 = (img_tensor * 2.0) - 1.0
    
    val_data = val_data_template.copy()
    val_data['A'] = img_m11
    
    model_diffusion.feed_data(val_data)
    with torch.no_grad():
        model_diffusion.test()
        
    predseg_raw = model_diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
    
    # Map [-1, 1] output safely to [0, 1] probabilities for SOTA ranking
    probs = np.clip((predseg_raw + 1.0) / 2.0, 0.0, 1.0)
    pred_mask_bool = (probs > 0.5)
    
    return pred_mask_bool, probs

def calculate_eigencam(diffusion, target_layer_name, val_data, image_size):
    feature_maps = []
    model_instance = diffusion.netG.segment_fn
    logger = logging.getLogger('base')

    try:
        target_layer = dict(model_instance.named_modules())[target_layer_name]
    except KeyError:
        logger.error(f"Target layer '{target_layer_name}' not found. Using blank heatmap.")
        return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)

    diffusion.feed_data(val_data)
    with torch.no_grad():
        diffusion.test() 

    hook.remove()

    if not feature_maps: return np.zeros(image_size, dtype=np.float32)

    features = feature_maps[0].squeeze(0)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    
    n_samples, n_features = features_flat.shape
    n_components = min(1, n_samples, n_features)
    if n_components == 0: return np.zeros(image_size, dtype=np.float32)

    pca = PCA(n_components=n_components).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    
    saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
    return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()

def visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    display_image_rgb = gray2rgb(orig_image_01) if orig_image_01.ndim == 2 else orig_image_01
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb); axes[0].set_title("Original Image"); axes[0].axis('off')
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
# --- 2. Core Step-Wise Robustness Analysis (Upgraded for Mean Probability)
# --------------------------------------------------------------------------
def run_ranking_analysis(
    analysis_type, model_diffusion, device, val_data, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, current_image_output_dir, initial_probs, ic_sps_all, oc_sps_all,
    noise_step=0.05, max_level=5.0, save_visuals=False, alpha=1.0, beta=1.0
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

        noise_levels_arr = np.round(np.arange(1, int(round(max_level / noise_step)) + 1) * noise_step, 4)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False, position=1)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            

            perturbed_pred_mask_bool, perturbed_probs = predict_slice_diffusion(model_diffusion, perturbed_img_gray, val_data, device)

            # --- IMAGE SAVING AT 0.5 INTERVALS ---
            if save_visuals and int(round(noise_level, 2) * 100) % 50 == 0:
                Image.fromarray((perturbed_img_gray * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                mask_to_save = (perturbed_pred_mask_bool * 255).astype(np.uint8)
                Image.fromarray(mask_to_save).resize((256, 256), Image.NEAREST).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                
                img_rgb = gray2rgb(perturbed_img_gray) if perturbed_img_gray.ndim == 2 else perturbed_img_gray
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask_bool, color=(1, 0, 0), mode='thick')
                overlay_to_save = (overlay_image * 255).astype(np.uint8)
                Image.fromarray(overlay_to_save).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            # --- Global IoU at Step k ---
            intersect_k = np.sum((perturbed_pred_mask_bool == 1) & (target_class_gt == 1))
            union_k = np.sum((perturbed_pred_mask_bool == 1) | (target_class_gt == 1))
            global_iou_k = (intersect_k / union_k * 100.0) if union_k > 0 else 100.0

            # --- Instant Background Spillover Ratio ---
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

# =================================================================================
# 3. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="PRISM Original Continuous Noise Engine (Upgraded Probability Rank) - Diffusion")
    
    # Diffusion Specific Logger / Config Args
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)

    # Core Args
    parser.add_argument("--output_dir", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/XCAD/results/cdarl/prism_out_global")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_dataset_runs", type=int, default=1, help="Dataset iterations.")
    parser.add_argument("--noise_step", type=float, default=0.1, help="Step size for noise loop.")
    parser.add_argument("--max_level", type=float, default=5.0, help="Highest noise level of the grid (inclusive).")
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--target_layer_names', nargs='+', default=['Conv'])
    parser.add_argument('--img_path', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/XCAD/img_sel.txt', help="Path to frozen images file.")

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
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    cohort_file = args.img_path
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])
    
    for idata, val_data in enumerate(tqdm(val_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break

        case_name = os.path.basename(val_data['P'][0]).replace('.png', '')
        if case_name not in frozen_cohort: continue
        
        orig_image_np_raw = val_data['A'][0].cpu().numpy()
        if orig_image_np_raw.ndim == 3:
            orig_image_np = rgb2gray(orig_image_np_raw.transpose(1, 2, 0)) if orig_image_np_raw.shape[0] == 3 else orig_image_np_raw.squeeze()
        else:
            orig_image_np = orig_image_np_raw
            
        orig_image_01 = np.clip((orig_image_np + 1.0) / 2.0, 0.0, 1.0)

        # Initial forward pass extracting specific continuous probabilities
        initial_pred_mask_bool, initial_probs = predict_slice_diffusion(diffusion, orig_image_01, val_data, device)

        gt_mask_raw = val_data['F'][0].squeeze().cpu().numpy()
        target_class_gt = ((gt_mask_raw + 1.0) / 2.0 > 0.5)
        
        segments = slic(orig_image_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        # Absolute Anchoring (No prediction threshold filters used; 0.85 tolerance for thin vessels)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=0.85, oc_threshold=0.0)
        
        if not ic_sps_all: continue

        qualified_images.append({
            'case_name': case_name, 'val_data': val_data, 'orig_image_01': orig_image_01, 
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'initial_probs': initial_probs,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool.astype(np.uint8), target_class_gt.astype(np.uint8))
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: MULTI-RUN ROBUSTNESS INFERENCE (GLOBAL)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Executing Dataset Extractions...")

    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0, leave=True):
            case_name = q_data['case_name']
            val_data = q_data['val_data']
            orig_image_01 = q_data['orig_image_01']
            target_class_gt = q_data['target_class_gt']
            segments = q_data['segments']
            initial_pred_mask_bool = q_data['initial_pred_mask_bool']
            initial_probs = q_data['initial_probs']
            ic_sps_all = q_data['ic_sps_all']
            oc_sps_all = q_data['oc_sps_all']

            current_dir = os.path.join(run_output_dir, case_name)
            os.makedirs(current_dir, exist_ok=True)
            
            final_analysis = {
                "image_info": {
                    "name": case_name, 
                    "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)
                }
            }


            # Extract ALL unique superpixels for the pure SOTA baseline
            all_sps = np.unique(segments)
            all_sps = all_sps[all_sps != 0]

            # --- Global Analysis: perturb and measure all superpixels ---
            res_global = run_ranking_analysis(
                analysis_type='global_sota', 
                model_diffusion=diffusion, 
                device=device, 
                val_data=val_data,
                orig_image_01=orig_image_01, 
                segments=segments, 
                target_class_gt=target_class_gt, 
                superpixels_to_perturb=all_sps, 
                superpixels_to_measure=all_sps, 
                initial_pred_mask_bool=initial_pred_mask_bool,
                noise_types=args.noise_types, 
                current_image_output_dir=current_dir, 
                initial_probs=initial_probs, 
                ic_sps_all=ic_sps_all, 
                oc_sps_all=oc_sps_all, 
                noise_step=args.noise_step, max_level=args.max_level, 
                save_visuals=(run_idx == 0)
            )
            final_analysis['global_sota_analysis'] = res_global

            with open(os.path.join(current_dir, f"{case_name}_analysis.json"), 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
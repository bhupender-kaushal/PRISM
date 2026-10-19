import os
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

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

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

def calculate_iou_in_roi(pred_mask, gt_mask, roi_mask):
    roi_mask = roi_mask.astype(bool)
    pred_roi = pred_mask[roi_mask]
    gt_roi = gt_mask[roi_mask]
    intersection = np.logical_and(pred_roi, gt_roi).sum()
    union = np.logical_or(pred_roi, gt_roi).sum()
    return (intersection / union)*100 if union > 0 else 0.0

def calculate_rbo(list1, list2, p=0.9):
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    if not list1 or not list2: return 0.0
    sl, ll = sorted([(len(list1), list1), (len(list2), list2)])
    s_len, s_list = sl; l_len, l_list = ll; agreement = 0.0
    for d in range(1, s_len + 1):
        set1, set2 = set(s_list[:d]), set(l_list[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
    for d in range(s_len + 1, l_len + 1):
        set1, set2 = set(s_list), set(l_list[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
    return (1 - p) * agreement

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
        noisy_region = np.random.poisson(region * 25 * sigma) / (25 * sigma)
        perturbed_image[region_mask] = noisy_region
    elif noise_type == 'speckle':
        noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1 + noise[region_mask])
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=0.5 * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
    elif noise_type == 'rician':
        n1 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

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

def run_ranking_analysis(
    analysis_type, model_diffusion, device, batch_data, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, current_image_output_dir, noise_step=0.1, save_visuals=False
):
    logger = logging.getLogger('base')
    analysis_results = {}
    
    lambda_param = 1.0 
    beta_param = 1.0
    max_id = segments.max() + 1
    
    # --- BG-IoU SPILLOVER BASELINE CALCULATION ---
    bg_gt = (target_class_gt == 0)
    bg_pred_init = (initial_pred_mask_bool == 0)
    
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0
    
    # --- VECTORIZED FIX: Initial IoU Calculation ---
    intersect_mask_init = (initial_pred_mask_bool == 1) & (target_class_gt == 1)
    union_mask_init = (initial_pred_mask_bool == 1) | (target_class_gt == 1)
    
    intersect_counts_init = np.bincount(segments[intersect_mask_init].ravel(), minlength=max_id)
    union_counts_init = np.bincount(segments[union_mask_init].ravel(), minlength=max_id)
    
    initial_sp_iou_data = []
    for sp_id in superpixels_to_measure:
        u_val = union_counts_init[sp_id]
        i_val = intersect_counts_init[sp_id]
        iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
        initial_sp_iou_data.append({"sp_id": int(sp_id), "iou": round(iou_val, 2)})

    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): sp_data['rank'] = rank
    initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        if save_visuals:
            noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
            os.makedirs(noise_specific_dir, exist_ok=True)
        
        level_data_list = [{"noise_level": 0.0, "image_path": "original", "superpixels": sorted_initial_sp_data}]
        rbo_scores_list, sp_iou_history, sp_rank_history = [], {sp_id: [] for sp_id in superpixels_to_measure}, {sp_id: [] for sp_id in superpixels_to_measure}

        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0

        noise_levels_arr = np.arange(noise_step, 3.1, noise_step)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            
            is_final_step = False
            if ssim_score < 0.70: 
                logger.info(f"\n[SSIM TRIGGER] SSIM ({ssim_score:.4f}) dropped below 0.70 at {noise_type.upper()} level {noise_level:.2f}. Halting.")
                is_final_step = True
            
            # Reconstruct [-1, 1] tensor for Diffusion model
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
            perturbed_pred_mask_bool = ((perturbed_predseg_raw + 1) / 2.0 > 0.5)

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

            if save_visuals and (step_idx % 5 == 0 or is_final_step):
                Image.fromarray((perturbed_img_gray * 255).astype(np.uint8)).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                mask_to_save = (perturbed_pred_mask_bool * 255).astype(np.uint8)
                Image.fromarray(mask_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                
                img_rgb = gray2rgb(perturbed_img_gray) if perturbed_img_gray.ndim == 2 else perturbed_img_gray
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask_bool, color=(1, 0, 0), mode='thick')
                overlay_to_save = (overlay_image * 255).astype(np.uint8)
                Image.fromarray(overlay_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            if is_final_step:
                break 

            # --- VECTORIZED FIX: Perturbed IoU Calculation ---
            intersect_mask_pert = (initial_pred_mask_bool == 1) & (perturbed_pred_mask_bool == 1)
            union_mask_pert = (initial_pred_mask_bool == 1) | (perturbed_pred_mask_bool == 1)
            
            intersect_counts_pert = np.bincount(segments[intersect_mask_pert].ravel(), minlength=max_id)
            union_counts_pert = np.bincount(segments[union_mask_pert].ravel(), minlength=max_id)
            
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
            
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            rbo_scores_list.append(round(calculate_rbo(initial_rank_list, current_rank_list, p=0.9), 4))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
            
            level_data_list.append({
                "noise_level": round(noise_level, 2), 
                "instant_spillover_ratio": round(spillover_ratio_at_j, 4),
                "superpixels": sorted_superpixels
            })
        
        # --- SPILLOVER MATH: Conditional Penalty Logic ---
        if conditional_bleed_count > 0:
            avg_spillover = cumulative_spillover_ratio / conditional_bleed_count
        else:
            avg_spillover = 0.0
            
        spillover_discount = math.exp(-beta_param * avg_spillover)
        
        reliability_results = []
        for sp_id in superpixels_to_measure:
            avg_iou = np.mean(sp_iou_history[sp_id]) if sp_iou_history[sp_id] else 0.0
            initial_iou = initial_iou_map.get(sp_id, 0.0)
            
            rank_list = [r for r in sp_rank_history[sp_id] if r is not None]
            num_changes = sum(1 for i in range(1, len(rank_list)) if rank_list[i] != rank_list[i-1])
            sum_of_jumps = sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list)))
            
            rank_perplexity = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0
            total_measured = len(superpixels_to_measure)
            rank_perplexity = rank_perplexity / total_measured if total_measured > 0 else 0.0

            confidence_weight = exp(-lambda_param * rank_perplexity)
            weighted_iou = avg_iou * confidence_weight
            
            # --- DECOUPLED FIX: Base reliability independent of spillover penalty ---
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
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "rbo_at_each_level": rbo_scores_list,
            "spillover_metrics": { 
                "conditional_average_spillover": round(avg_spillover, 4),
                "bleed_count": conditional_bleed_count,
                "global_spillover_discount": round(spillover_discount, 4)
            },
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['base_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results

# =================================================================================
# 3. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="High-Speed Two-Phase Model Reliability Inference (Diffusion)")
    
    # Diffusion Specific Logger / Config Args
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)

    # Two-Phase Core Args
    parser.add_argument("--output_dir", type=str, default="./new_expresults_ws")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_dataset_runs", type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument("--noise_step", type=float, default=0.05, help="Step size for noise loop.")
    
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--target_layer_names', nargs='+', default=['Conv'])

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
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    for idata, val_data in enumerate(tqdm(val_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break

        case_name = os.path.basename(val_data['P'][0]).replace('.png', '')
        
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
        
        if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, full_image_roi) < 55: continue

        # segments = slic(orig_image_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        # for quickshift
        # segments = quickshift(gray2rgb(orig_image_01), kernel_size=5, max_dist=10, ratio=0.5)

        # for Watershed
        gradient = sobel(orig_image_01)
        grid = regular_grid(orig_image_01.shape[:2], n_points=500)
        markers = np.zeros(orig_image_01.shape[:2], dtype=int)
        
        # Safely reshape the arange array to match the N-dimensional grid space
        target_shape = markers[grid].shape
        markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
        
        segments = watershed(gradient, markers, compactness=0.001)

        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=0.85, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'case_name': case_name, 'val_data': val_data, 'orig_image_01': orig_image_01, 
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool.astype(np.uint8), target_class_gt.astype(np.uint8))
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: N DATASET ITERATIONS (PERTURBATION & MEASUREMENT)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Iterations...")
    dataset_run_harmonic_scores = []
    
    all_runs_summary = {
        'reliability': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}}, 
        'rbo': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}}
    }

    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        run_ic_scores, run_oc_scores = [], []

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0):
            case_name = q_data['case_name']
            val_data = q_data['val_data']
            orig_image_01 = q_data['orig_image_01']
            target_class_gt = q_data['target_class_gt']
            segments = q_data['segments']
            initial_pred_mask_bool = q_data['initial_pred_mask_bool']
            qualified_ic_sp = q_data['qualified_ic_sp']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, case_name)
            os.makedirs(current_image_output_dir, exist_ok=True)
            final_analysis = {"image_info": {"name": case_name, "initial_dice_vs_gt": round(q_data['initial_dice_vs_gt'], 2)}}

            res_ic = run_ranking_analysis(
                'in_context', diffusion, device, val_data, orig_image_01, segments, target_class_gt, 
                qualified_ic_sp, qualified_ic_sp, initial_pred_mask_bool,
                args.noise_types, current_image_output_dir, noise_step=args.noise_step, save_visuals=(run_idx == 0)
            )
            final_analysis['in_context_analysis'] = res_ic
            
            # --- DECOUPLED FIX: Image Level In-Context Aggregation ---
            img_ic_scores = []
            for nt in args.noise_types: 
                all_runs_summary['rbo']['in_context'][nt].append(res_ic[nt]['average_rank_biased_overlap'])
                nt_rels = [r['base_reliability_score'] for r in res_ic[nt]['reliability_analysis']]
                
                if nt_rels: 
                    base_img_score = np.mean(nt_rels)
                    discount = res_ic[nt]['spillover_metrics']['global_spillover_discount']
                    discounted_img_score = base_img_score * discount
                    
                    all_runs_summary['reliability']['in_context'][nt].append(discounted_img_score)
                    img_ic_scores.append(discounted_img_score)
            
            if img_ic_scores: run_ic_scores.append(np.mean(img_ic_scores))

            # --- DYNAMIC 35% OC SELECTION FIX ---
            num_oc_to_perturb = int(0.35 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments)
                for sp_id in qualified_ic_sp: in_context_map[segments == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments)
                for sp_id in oc_sps_all: out_of_context_map[segments == sp_id] = sp_id
                
                try:
                    eigencam_heatmap = calculate_eigencam(diffusion, 'conv_img', val_data, orig_image_01.shape)
                except Exception as e:
                    eigencam_heatmap = np.zeros_like(orig_image_01, dtype=np.float32)

                visualize_segmentation_comparison(orig_image_01, target_class_gt, initial_pred_mask_bool, segments, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, "overview.png"))

            res_oc = run_ranking_analysis(
                'out_of_context', diffusion, device, val_data, orig_image_01, segments, target_class_gt, 
                out_of_context_sp_to_perturb, qualified_ic_sp, initial_pred_mask_bool,
                args.noise_types, current_image_output_dir, noise_step=args.noise_step, save_visuals=(run_idx == 0)
            )
            final_analysis['out_of_context_analysis'] = res_oc
            
            # --- DECOUPLED FIX: Image Level Out-Of-Context Aggregation ---
            img_oc_scores = []
            for nt in args.noise_types: 
                all_runs_summary['rbo']['out_of_context'][nt].append(res_oc[nt]['average_rank_biased_overlap'])
                nt_rels = [r['base_reliability_score'] for r in res_oc[nt]['reliability_analysis']]
                
                if nt_rels: 
                    base_img_score = np.mean(nt_rels)
                    discount = res_oc[nt]['spillover_metrics']['global_spillover_discount']
                    discounted_img_score = base_img_score * discount
                    
                    all_runs_summary['reliability']['out_of_context'][nt].append(discounted_img_score)
                    img_oc_scores.append(discounted_img_score)

            if img_oc_scores: run_oc_scores.append(np.mean(img_oc_scores))

            with open(os.path.join(current_image_output_dir, f"{case_name}_analysis.json"), 'w') as f: json.dump(final_analysis, f, indent=4, cls=NpEncoder)

        E_R_IC = np.mean(run_ic_scores) if run_ic_scores else 0.0
        E_R_OC = np.mean(run_oc_scores) if run_oc_scores else 0.0
        R_k = 2 * (E_R_IC * E_R_OC) / (E_R_IC + E_R_OC + 1e-6)
        dataset_run_harmonic_scores.append(R_k)
        logger.info(f"Run {run_idx + 1} Aggregate | E[R_IC]: {E_R_IC:.4f} | E[R_OC]: {E_R_OC:.4f} | R_harmonic: {R_k:.4f}")

    # =========================================================================
    # PHASE 3: FINAL DATASET RELIABILITY MATH (\Theta) & PLOTTING
    # =========================================================================
    logger.info("\n=== [PHASE 3] Final Intra-Dataset Reliability Calculation & Plotting ===")
    
    R_bar = np.mean(dataset_run_harmonic_scores)
    sigma_R = np.std(dataset_run_harmonic_scores)
    stability_coefficient = 1.0 - sigma_R
    Theta_D = R_bar * stability_coefficient

    logger.info(f"Array of N Harmonic Scores: {[round(r,4) for r in dataset_run_harmonic_scores]}")
    logger.info(f"Expected Robustness (R_bar): {R_bar:.4f}")
    logger.info(f"Intra-dataset Std Dev (sigma_R): {sigma_R:.4f}")
    logger.info(f"Stability Coefficient (1 - sigma_R): {stability_coefficient:.4f}")
    logger.info(f"Dataset-Specific Reliability (Theta_D): {Theta_D:.4f}")
    
    final_metrics = {
        "Dataset_Runs_N": args.num_dataset_runs,
        "Harmonic_Scores_Per_Run": dataset_run_harmonic_scores,
        "Expected_Robustness_R_bar": R_bar,
        "Intra_Dataset_Std_Dev_Sigma": sigma_R,
        "Stability_Coefficient": stability_coefficient,
        "Dataset_Reliability_Theta": Theta_D
    }
    with open(os.path.join(args.output_dir, "FINAL_DATASET_RELIABILITY.json"), 'w') as f: json.dump(final_metrics, f, indent=4)

    logger.info("--- Generating Final Summary Plots ---")
    rbo_plot_data = [{'Analysis Type': at.replace('_', ' ').title(), 'Noise Type': nt, 'RBO Score': np.clip(s, 0,1)} for at, nd in all_runs_summary['rbo'].items() for nt, sc in nd.items() for s in sc]

    if rbo_plot_data:
        df_rbo = pd.DataFrame(rbo_plot_data)
        min_rbo, max_rbo = df_rbo['RBO Score'].min(), df_rbo['RBO Score'].max()
        if max_rbo > min_rbo: df_rbo['RBO Score'] = (df_rbo['RBO Score'] - min_rbo) / (max_rbo - min_rbo)
        else: df_rbo['RBO Score'] = 0.0

        g = sns.catplot(data=df_rbo, x='Noise Type', y='RBO Score', col='Analysis Type', kind='violin', palette='viridis', sharey=True, height=6, aspect=1.2)
        g.fig.suptitle(f'RBO Distribution Across {args.num_dataset_runs} Runs', y=1.03, fontsize=16); g.set_axis_labels("Noise Type", "RBO Score"); g.set_xticklabels(rotation=45, ha='right'); g.set_titles("{col_name} Analysis")
        for ax in g.axes.flat: ax.set_ylim(0, 1)
        plt.savefig(os.path.join(args.output_dir, "RBO_Distributions_ViolinPlot.png"), bbox_inches='tight', dpi=300); plt.close(g.fig)

    for analysis_type, noise_data in all_runs_summary['reliability'].items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7)); bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='mako')
            plt.title(f"Expected Reliability ({analysis_type.replace('_', ' ').title()}) over {args.num_dataset_runs} Runs", fontsize=16)
            for bar in bars.patches: bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, f"{analysis_type}_final_reliability_summary.png"), dpi=300); plt.close()

    plt.figure(figsize=(10, 6))
    runs_x = np.arange(1, args.num_dataset_runs + 1)
    plt.plot(runs_x, dataset_run_harmonic_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Run Robustness ($R_k$)')
    plt.axhline(R_bar, color='r', linestyle='--', linewidth=2, label=f'Expected Mean ($\overline{{R}}$) = {R_bar:.4f}')
    plt.fill_between(runs_x, R_bar - sigma_R, R_bar + sigma_R, color='r', alpha=0.2, label=f'$\pm 1\sigma_R$ ({sigma_R:.4f})')
    plt.title(f'Intra-Dataset Robustness Variance over {args.num_dataset_runs} Runs', fontsize=16, pad=15)
    plt.xlabel('Dataset Iteration (N)', fontsize=14); plt.ylabel('Dataset Robustness Score (Harmonic)', fontsize=14)
    plt.xticks(runs_x)
    safe_min = max(0, min(dataset_run_harmonic_scores) - (2 * sigma_R) - 0.05)
    safe_max = min(1, max(dataset_run_harmonic_scores) + (2 * sigma_R) + 0.05)
    plt.ylim(safe_min, safe_max)
    plt.legend(loc='best', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, "Run_Variance_Plot.png"), dpi=300); plt.close()
    
    logger.info(f"--- Pipeline Finished. Results and Plots are in {args.output_dir} ---")

if __name__ == "__main__":
    main()
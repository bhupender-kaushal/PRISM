# --- Imports at the top of the file ---
from email.policy import default
import torch
import numpy as np
import os
import time
import argparse
import logging
from math import ceil, exp
from PIL import Image
import random
import json
import itertools

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

# --- START: Helper functions for superpixel processing and visualization ---
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import slic, mark_boundaries, quickshift
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr
import pandas as pd
import seaborn as sns


# Imports for REAL EigenCAM
from sklearn.decomposition import PCA
import torch.nn.functional as F



# --- MODIFIED FUNCTION ---
def calculate_eigencam(diffusion, target_layer_name, val_data, image_size):
    """
    Calculates the EigenCAM heatmap using the main model's inference pipeline for consistency.
    """
    feature_maps = []
    # Get the model instance where the target layer resides
    model_instance = diffusion.netG.segment_fn
    logger = logging.getLogger('base')

    try:
        # Locate the target layer within the model instance
        target_layer = dict(model_instance.named_modules())[target_layer_name]
    except KeyError:
        logger.error(f"Target layer '{target_layer_name}' not found in model. EigenCAM calculation failed.")
        available_layers = list(dict(model_instance.named_modules()).keys())
        logger.debug(f"Available layers are: {available_layers}")
        return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Register the forward hook to capture the feature maps from the target layer
    hook = target_layer.register_forward_hook(hook_fn)

    # Use the SAME data feeding and testing mechanism as the main analysis loop
    # This ensures methodological consistency.
    diffusion.feed_data(val_data)
    with torch.no_grad():
        diffusion.test()  # This executes the forward pass and triggers the hook

    hook.remove() # Always remove hooks after use

    if not feature_maps:
        logger.error("EigenCAM hook did not capture any feature maps. The forward pass might have failed.")
        return np.zeros(image_size, dtype=np.float32)

    # --- The rest of the EigenCAM logic remains the same ---
    features = feature_maps[0].squeeze(0)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    
    # Perform PCA to find the principal components of the feature maps
    pca = PCA(n_components=1).fit(features_flat)
    
    # Project the feature maps onto the first principal component
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    
    # Normalize the saliency map to be in the [0, 1] range
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    
    # Resize the heatmap to match the original image size
    saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
    return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()


# --- Scoring and Selection Functions (remain the same) ---
def calculate_dice_score(mask1, mask2, label=1):
    """Calculates the Dice Similarity Coefficient (DSC) between two binary masks."""
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
    """Calculates the Intersection over Union (IoU) within a given Region of Interest (ROI)."""
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

def calculate_rbo(list1, list2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two ranked lists.
    p: Top-weightedness parameter (0 < p < 1).
    """
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    
    if not list1 or not list2: return 0.0
        
    sl, ll = sorted([(len(list1), list1), (len(list2), list2)])
    s, S = sl
    l, L = ll
    
    agreement = 0.0
    for d in range(1, s + 1):
        set1 = set(S[:d])
        set2 = set(L[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
        
    for d in range(s + 1, l + 1):
        set1 = set(S)
        set2 = set(L[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
        
    return (1 - p) * agreement

def get_refined_superpixel_map(gt_mask, superpixels_segments, foreground_class_id=1):
    """Creates an in-context superpixel map by clipping superpixels to the GT foreground."""
    gt_foreground = (gt_mask == foreground_class_id).astype(bool)
    in_context_segments = np.copy(superpixels_segments)
    in_context_segments[~gt_foreground] = 0
    return in_context_segments

def select_superpixels_by_pred_gt_iou(superpixels_segments, pred_mask, gt_mask, iou_threshold=0.5):
    """Selects superpixels where the IoU between prediction and GT meets a threshold."""
    selected_sp_ids = []
    unique_labels = np.unique(superpixels_segments)
    unique_labels = unique_labels[unique_labels != 0]
    for sp_id in unique_labels:
        sp_mask = (superpixels_segments == sp_id)
        intersection = np.sum((pred_mask == 1) & (gt_mask == 1) & sp_mask)
        union = np.sum(((pred_mask == 1) | (gt_mask == 1)) & sp_mask)
        if union > 0:
            if (intersection / union) >= iou_threshold:
                selected_sp_ids.append(sp_id)
    return selected_sp_ids

def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """Perturbs image regions and returns the image and the noise characteristic value used."""
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0
    for sp_id in selected_superpixel_ids:
        superpixel_mask = (superpixels_segments == sp_id)
        if noise_type == 'gaussian':
            noise_char_value = 0.1 * noise_level
            noise = np.random.normal(0, noise_char_value, perturbed_image[superpixel_mask].shape)
            perturbed_image[superpixel_mask] += noise
        elif noise_type == 'salt_pepper':
            noise_char_value = min(0.5, 0.05 * noise_level)
            num_pixels = np.sum(superpixel_mask)
            num_noise = int(noise_char_value * num_pixels)
            rows, cols = np.where(superpixel_mask)
            if num_noise > 0:
                indices = random.sample(range(len(rows)), num_noise)
                for i in indices: perturbed_image[rows[i], cols[i]] = random.choice([0.0, 1.0])
        elif noise_type == 'poisson':
            sigma = max(0.05, 3.2 - noise_level)
            region = perturbed_image[superpixel_mask]
            noisy_region = np.random.poisson(region * 25 * sigma) / (25 * sigma)
            perturbed_image[superpixel_mask] = noisy_region
            noise_char_value = sigma
        elif noise_type == 'speckle':
            noise_char_value = 0.1 * noise_level
            noise = np.random.normal(0, noise_char_value, perturbed_image[superpixel_mask].shape)
            perturbed_image[superpixel_mask] *= (1 + noise)
        elif noise_type == 'gaussian_blur':
            noise_char_value = 0.5 * noise_level
            blurred_image = gaussian_filter(original_image_scaled, sigma=noise_char_value)
            perturbed_image[superpixel_mask] = blurred_image[superpixel_mask]
        elif noise_type == 'rician':
            noise_char_value = 0.1 * noise_level
            region = perturbed_image[superpixel_mask]
            n1 = np.random.normal(0, noise_char_value, region.shape)
            n2 = np.random.normal(0, noise_char_value, region.shape)
            noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
            perturbed_image[superpixel_mask] = noisy_region
    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value


def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    """Visualizes the original image, GT, prediction, and various superpixel maps."""
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))
    axes[0].imshow(display_image_rgb); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
    axes[3].imshow(mark_boundaries(display_image_rgb, all_sp_map)); axes[3].set_title(f"All Superpixels"); axes[3].axis('off')
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map)); axes[4].set_title("In-Context Superpixels"); axes[4].axis('off')
    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map)); axes[5].set_title("Out-of-Context Superpixels"); axes[5].axis('off')
    axes[6].imshow(display_image_rgb); im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1); axes[6].set_title("EigenCAM"); axes[6].axis('off')
    plt.colorbar(im, ax=axes[6])
    plt.tight_layout()

    # # Save in-context and out-of-context superpixel maps
    # # in_context_save_path = os.path.join(os.path.dirname(save_path), "in_context_superpixels.png")
    # out_of_context_save_path = os.path.join(os.path.dirname(save_path), "out_of_context_superpixels.png")
    
    # plt.figure(figsize=(8, 8))
    # plt.imshow(mark_boundaries(display_image_rgb, out_of_context_map, color=(0, 1, 0)))  # Red color for boundaries
    # plt.title("Out-of-Context Superpixels")
    # plt.axis('off')
    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.close()

    # ##unitl here

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# --- REFACTORED: Main Analysis Runner ---
def run_ranking_analysis(
    analysis_type, args, device, diffusion, val_data,
    orig_image_scaled, predseg, gt_mask_np, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, current_image_output_dir, dataInfo,
    ssim_data_collector
):
    """
    Runs the full ranking analysis for a given set of superpixels and returns the results.
    """
    logger = logging.getLogger('base')
    logger.info(f"--- Starting Ranking Analysis: {analysis_type.upper()} ---")
    logger.info(f"Perturbing {len(superpixels_to_perturb)} SPs, measuring effect on {len(superpixels_to_measure)} SPs.")

    analysis_results = {}

    initial_sp_iou_data = [
        {"sp_id": int(sp_id), "iou": round(calculate_iou_score(predseg, gt_mask_np, segments_slic == sp_id), 2)}
        for sp_id in superpixels_to_measure
    ]
    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1):
        sp_data['rank'] = rank
        
    # Create a lookup map for initial IoU for efficient access later
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        logger.info(f"--- Analyzing Noise Type: {noise_type.upper()} for {analysis_type} ---")
        noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
        os.makedirs(noise_specific_dir, exist_ok=True)

        level_0_entry = {noise_char_map.get(noise_type, 'noise_value'): 0.0, "ssim_score": 1.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
        level_data_list = [level_0_entry]
        
        rbo_scores_list = []
        sp_iou_history = {sp_id: [] for sp_id in superpixels_to_measure}
        sp_rank_history = {sp_id: [] for sp_id in superpixels_to_measure}
        
        for noise_level in np.arange(0.05, 3, 0.05):
            perturbed_image_np_scaled, noise_char_value = perturb_superpixels_with_level(orig_image_scaled, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_scaled, perturbed_image_np_scaled, data_range=1.0)

            if ssim_score < 0.85:
                logger.warning(f"SSIM dropped below 0.85 ({ssim_score:.3f}) at noise level {noise_level:.2f}. Stopping further perturbations for {noise_type}.")
                break
            
            if args.plot_ssim:
                ssim_data_collector.append({
                    "Analysis Type": analysis_type,
                    "Noise Type": noise_type,
                    "Noise Level": noise_level,
                    "SSIM": ssim_score
                })
            
            pil_img = Image.fromarray((perturbed_image_np_scaled * 255).astype(np.uint8))
            img_save_path = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.3f}.png")
            pil_img.save(img_save_path)
            
            perturbed_image_tensor = torch.from_numpy(perturbed_image_np_scaled[None, None, ...]).float()
            perturbed_input_for_netG = ((perturbed_image_tensor * 2) - 1).to(device)
            val_data_perturbed = val_data.copy(); val_data_perturbed['A'] = perturbed_input_for_netG
            diffusion.feed_data(val_data_perturbed); diffusion.test()
            perturbed_predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
            perturbed_predseg = ((perturbed_predseg_raw + 1) / 2.0 > 0.5).astype(np.uint8)


            # Save the perturbed segmentation map
            segmentation_save_path = os.path.join(noise_specific_dir, f"segmentation_level_{noise_level:.2f}_ssim_{ssim_score:.3f}.png")
            segmentation_image = Image.fromarray((perturbed_predseg * 255).astype(np.uint8))
            segmentation_image.save(segmentation_save_path)


            current_level_sp_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(predseg, perturbed_predseg, segments_slic == sp_id), 2)} for sp_id in superpixels_to_measure]
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            for sp_id in superpixels_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))

            initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            current_rbo = calculate_rbo(initial_rank_list, current_rank_list, p=0.9)
            rbo_scores_list.append(round(current_rbo, 4))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
            level_data_list.append({noise_char_map.get(noise_type, 'noise_value'): round(noise_char_value, 4), "ssim_score": round(ssim_score, 4), "image_path": os.path.relpath(img_save_path, args.output_dir), "superpixels": sorted_superpixels})
        
        # <<< --- START OF MODIFIED CALCULATION BLOCK --- >>>
        reliability_results = []
        for sp_id in superpixels_to_measure:
            # Note: sp_iou_history does NOT include the initial_iou, so this is the average of perturbed IoUs.
            avg_iou = np.mean(sp_iou_history[sp_id]) if sp_iou_history[sp_id] else 0.0
            initial_iou = initial_iou_map.get(sp_id, 0.0)
            
            # Calculate rank perplexity (formerly avg_rank_jump)
            rank_list = sp_rank_history[sp_id]
            num_changes = sum(1 for i in range(1, len(rank_list)) if rank_list[i] is not None and rank_list[i-1] is not None and rank_list[i] != rank_list[i-1])
            sum_of_jumps = sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list)) if rank_list[i] is not None and rank_list[i-1] is not None and rank_list[i] != rank_list[i-1])
            rank_perplexity = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0

            # Step 1: Calculate the penalty score
            delta_iou = avg_iou - initial_iou
            penalty_score = 0.0
            if abs(delta_iou) > 1e-9:
                exponent = rank_perplexity / abs(delta_iou)
                factor = 1.0 - (1.0 / exp(exponent))
                penalty_score = 2 * delta_iou * factor
            
            # Step 2: Calculate pIoU
            pIoU = avg_iou + penalty_score
            
            # Step 3: Calculate the final reliability score
            final_reliability_score = 0.0
            if initial_iou > 1e-9:
                numerator = abs(initial_iou - pIoU)
                final_reliability_score = 1.0 - (numerator / initial_iou)

            reliability_results.append({
                "sp_id": int(sp_id),
                "final_reliability_score": round(final_reliability_score, 4),
                "pIoU": round(pIoU, 4),
                "penalty_score": round(penalty_score, 4),
                "avg_iou": round(avg_iou, 4),
                "initial_iou": round(initial_iou, 4), # Also adding initial_iou for clarity
                "rank_perplexity": round(rank_perplexity, 4)
            })
        
        avg_final_reliability = np.mean([res['final_reliability_score'] for res in reliability_results])
        logger.info(f"[{analysis_type.upper()}] FINAL Average Reliability Score for {noise_type}: {avg_final_reliability:.4f}")
        # <<< --- END OF MODIFIED CALCULATION BLOCK --- >>>

        analysis_results[noise_type] = {
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "rbo_at_each_level": rbo_scores_list,
            # Sort by the new final score in descending order (higher is better)
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['final_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Segmentation Prediction and Superpixel Ranking Pipeline")
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    # parser.add_argument('--output_dir', type=str, default='./expresults/mouse_ranked_quickshift')
    parser.add_argument('--output_dir', type=str, default='./expresults/mouse_ranked')
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--perturb_ratio', type=float, default=1.0)
    parser.add_argument('--num_out_of_context_perturb', type=int, default=40)
    parser.add_argument('--plot_ssim', action='store_true', help='If set, enables SSIM data saving and plotting.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode for more verbose logging.')


    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir: opt['datasets']['test']['dataroot'] = args.data_dir
    opt['phase'] = args.phase
    opt['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else []
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    Logger.setup_logger(None, log_dir, opt['phase'], level=log_level, screen=True)
    logger = logging.getLogger('base')

    dataset_opt = opt['datasets']['test']
    val_set = Data.create_dataset_xcad(dataset_opt, 'test')
    val_loader = Data.create_dataloader(val_set, dataset_opt, 'test')
    
    diffusion = Model.create_model(opt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    diffusion.netG.to(device)
    
    noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
    noise_char_map = {
        'salt_pepper': 'noise_density', 'gaussian': 'noise_std_dev', 'speckle': 'noise_std_dev',
        'poisson': 'noise_intensity_scale', 'gaussian_blur': 'blur_sigma', 'rician': 'noise_sigma'
    }

    all_images_summary_metrics = {
        'reliability': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}},
        'rbo': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}}
    }
    ssim_vs_noise_data = []

    for idata, val_data in enumerate(val_loader):
        if idata >= args.num_iterations: break
        dataInfo = os.path.basename(val_data['P'][0]).replace('.png', '')
        current_image_output_dir = os.path.join(args.output_dir, dataInfo)
        os.makedirs(current_image_output_dir, exist_ok=True)
        logger.info(f"Processing image: {dataInfo} ({idata + 1}/{args.num_iterations})")

        diffusion.feed_data(val_data); diffusion.test()
        visuals = diffusion.get_current_segment()
        predseg = ((visuals['test_V'].squeeze().cpu().numpy() + 1) / 2.0 > 0.5).astype(np.uint8)
        
        # The original input image tensor is typically in range [-1, 1]
        # We scale it to [0, 1] for skimage functions
        orig_image_scaled = (val_data['A'][0].squeeze().cpu().numpy() + 1) / 2.0
        gt_mask_np = ((val_data['F'][0].squeeze().cpu().numpy() + 1) / 2.0 > 0.5).astype(np.uint8)
        
        segments_slic = slic(orig_image_scaled, n_segments=300, compactness=0.1, sigma=1.0, channel_axis=None)
        
        # if orig_image_scaled.ndim == 2:  # Single-channel image
        #     img_quick = np.stack([orig_image_scaled] * 3, axis=-1)  # Create a 3-channel version for Quickshift
        # else:
        #     img_quick = orig_image_scaled
        # segments_slic = quickshift(img_quick, kernel_size=5, max_dist=10, ratio=0.7)

        iou_qualified_sp_ids = select_superpixels_by_pred_gt_iou(segments_slic, predseg, gt_mask_np, iou_threshold=0.5)
        
        if not iou_qualified_sp_ids:
            logger.warning("No superpixels met the 50% Pred/GT IoU threshold. Skipping.")
            continue
        
        in_context_sp_to_perturb = random.sample(iou_qualified_sp_ids, int(len(iou_qualified_sp_ids) * args.perturb_ratio))
        all_sp_ids = np.unique(segments_slic)
        out_of_context_ids = [sp_id for sp_id in all_sp_ids if sp_id not in iou_qualified_sp_ids and sp_id != 0]
        out_of_context_sp_to_perturb = random.sample(out_of_context_ids, min(args.num_out_of_context_perturb, len(out_of_context_ids)))
        
        in_context_map = get_refined_superpixel_map(gt_mask_np, segments_slic)
        out_of_context_map = np.zeros_like(segments_slic)
        for sp_id in out_of_context_ids: out_of_context_map[segments_slic == sp_id] = sp_id
        
        # --- MODIFIED CALL ---
        # Pass the whole diffusion model and the original val_data dictionary
        eigencam_heatmap = calculate_eigencam(
            diffusion, 'conv_img', val_data, orig_image_scaled.shape
        )

        visualize_segmentation_comparison(
            orig_image_scaled, gt_mask_np, predseg, segments_slic, in_context_map, out_of_context_map, eigencam_heatmap,
            save_path=os.path.join(current_image_output_dir, f"{dataInfo}_overview.png")
        )

        final_analysis = {"image_info": {"name": dataInfo, "initial_dice_vs_gt": round(calculate_dice_score(gt_mask_np, predseg), 2)}}

        final_analysis['in_context_analysis'] = run_ranking_analysis(
            'in_context', args, device, diffusion, val_data,
            orig_image_scaled, predseg, gt_mask_np, segments_slic,
            in_context_sp_to_perturb, iou_qualified_sp_ids,
            noise_types, noise_char_map, current_image_output_dir, dataInfo,
            ssim_data_collector=ssim_vs_noise_data
        )

        if out_of_context_sp_to_perturb:
            final_analysis['out_of_context_analysis'] = run_ranking_analysis(
                'out_of_context', args, device, diffusion, val_data,
                orig_image_scaled, predseg, gt_mask_np, segments_slic,
                out_of_context_sp_to_perturb, iou_qualified_sp_ids,
                noise_types, noise_char_map, current_image_output_dir, dataInfo,
                ssim_data_collector=ssim_vs_noise_data
            )

        json_save_path = os.path.join(current_image_output_dir, f"{dataInfo}_analysis.json")
        with open(json_save_path, 'w') as f: json.dump(final_analysis, f, indent=4)
        logger.info(f"Full analysis saved to {json_save_path}")

        for analysis_type in ['in_context', 'out_of_context']:
            if f'{analysis_type}_analysis' in final_analysis:
                for noise_type, results in final_analysis[f'{analysis_type}_analysis'].items():
                    all_images_summary_metrics['rbo'][analysis_type][noise_type].append(results['average_rank_biased_overlap'])
                    # --- CORRECTED LINE ---
                    # We now access 'final_reliability_score' which is the new output key
                    scores = [res['final_reliability_score'] for res in results.get('reliability_analysis', [])]
                    if scores:
                        all_images_summary_metrics['reliability'][analysis_type][noise_type].append(np.mean(scores))
    # --- RBO PLOTTING AND JSON SUMMARY ---
    logger.info("--- Preparing RBO data for plotting ---")
    rbo_scores_map = all_images_summary_metrics['rbo']
    rbo_dist_results = {}
    for at, nd in rbo_scores_map.items():
        rbo_dist_results[at] = {}
        for nt, sc in nd.items():
            if len(sc) > 1 and len(np.unique(sc)) > 1:
                try: rbo_dist_results[at][nt] = {'density_y': gaussian_kde(sc)(np.linspace(0,1,200)).tolist()}
                except np.linalg.LinAlgError: logger.warning(f"Could not calculate RBO KDE for {at} - {nt}")
    with open(os.path.join(args.output_dir, "rbo_distribution_summary.json"), 'w') as f: json.dump(rbo_dist_results, f)

    logger.info("--- Generating RBO Distribution Violin Plot ---")
    rbo_plot_data = []
    for analysis_type, noise_data in rbo_scores_map.items():
        for noise_type, scores in noise_data.items():
            for score in scores:
                rbo_plot_data.append({
                    'Analysis Type': analysis_type.replace('_', ' ').title(),
                    'Noise Type': noise_type,
                    'RBO Score': np.clip(score, 0.0, 1.0)  # Ensure scores are within [0, 1]
                })

    if rbo_plot_data:
        df_rbo = pd.DataFrame(rbo_plot_data)
        g = sns.catplot(
            data=df_rbo, x='Noise Type', y='RBO Score', col='Analysis Type',
            kind='violin', palette='viridis', sharey=True, height=6, aspect=1.2
        )
        g.fig.suptitle('RBO Score Distribution by Noise Type', y=1.03, fontsize=16)
        g.set_axis_labels("Noise Type", "RBO Score")
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles("{col_name} Analysis")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(args.output_dir, "RBO_Distributions_ViolinPlot.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved RBO distribution violin plot to {save_path}")

    # --- SSIM PLOTTING AND JSON SUMMARY ---
    if args.plot_ssim and ssim_vs_noise_data:
        logger.info("--- Preparing SSIM data and plots ---")
        df_ssim = pd.DataFrame(ssim_vs_noise_data)
        ssim_kde_results = {}
        for level in sorted(df_ssim['Noise Level'].round(1).unique()):
            df_level = df_ssim[np.isclose(df_ssim['Noise Level'], level, atol=0.1)]
            if not df_level.empty:
                ssim_kde_results[f"{level:.1f}"] = {}
                for nt in df_level['Noise Type'].unique():
                    scores = df_level[df_level['Noise Type'] == nt]['SSIM'].values
                    if len(scores) > 1 and len(np.unique(scores)) > 1:
                        try: ssim_kde_results[f"{level:.1f}"][nt] = {'density_y': gaussian_kde(scores)(np.linspace(0,1,200)).tolist()}
                        except np.linalg.LinAlgError: logger.warning(f"Could not calculate SSIM KDE for {nt} at level {level}")
        with open(os.path.join(args.output_dir, "ssim_kde_by_level.json"), 'w') as f: json.dump(ssim_kde_results, f)

    # --- ADDED: Final Reliability Plotting ---
    logger.info("--- Generating Final Reliability Summary Plots ---")
    reliability_scores_map = all_images_summary_metrics['reliability']
    for analysis_type, noise_data in reliability_scores_map.items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7))
            bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='viridis')
            plt.title(f"Final Reliability Score ({analysis_type.replace('_', ' ').title()})", fontsize=16)
            plt.ylim(0, 1)  # Ensure y-axis max is 1
            for bar in bars.patches:
                bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            plt.tight_layout()
            save_path = os.path.join(args.output_dir, f"{analysis_type}_final_reliability_summary.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"Saved {analysis_type} final reliability plot to {save_path}")

    logger.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()

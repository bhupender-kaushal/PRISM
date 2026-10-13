import os
import json
import argparse
import logging
import random
import time
import itertools
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

from skimage.segmentation import slic, quickshift, mark_boundaries
from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

from models.MSDUNet import MSDUFormer_Xnet_3463_FUESMS as model
from utils.dataloader import test_dataset

# --- START: Helper functions for superpixel processing and visualization ---
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import slic, mark_boundaries
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr
import pandas as pd
import seaborn as sns


# Imports for REAL EigenCAM
from sklearn.decomposition import PCA
import torch.nn.functional as F

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
    Clips output to [0, 1].
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
        
    rbo = (1 - p) * agreement
    # Clip to [0, 1]
    rbo = max(0.0, min(1.0, rbo))
    return rbo

def get_refined_superpixel_map(gt_mask, superpixels_segments, foreground_class_id=1):
    """Creates an in-context superpixel map by clipping superpixels to the GT foreground."""
    # Ensure gt_mask has the same shape as superpixels_segments
    if gt_mask.shape != superpixels_segments.shape:
        pil_img = Image.fromarray(gt_mask.astype(np.uint8))
        resized_img = pil_img.resize(superpixels_segments.shape[::-1], Image.NEAREST)
        gt_mask = np.array(resized_img)

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
            noise_char_value = max(0.05, 3.2 - noise_level)
            region = perturbed_image[superpixel_mask]
            noisy_region = np.random.poisson(region * 25 * noise_char_value) / (25 * noise_char_value)
            perturbed_image[superpixel_mask] = noisy_region
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

# --- MODIFIED FUNCTION ---
def calculate_eigencam(model, target_layer_name, image_tensor, image_size):
    """
    Calculates the EigenCAM heatmap using the main model's inference pipeline for consistency.
    """
    feature_maps = []
    logger = logging.getLogger('base')

    try:
        target_layer = dict(model.named_modules())[target_layer_name]
    except KeyError:
        logger.error(f"Target layer '{target_layer_name}' not found in model. EigenCAM calculation failed.")
        available_layers = list(dict(model.named_modules()).keys())
        logger.debug(f"Available layers are: {available_layers}")
        return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output):
        feature_maps.append(output)

    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image_tensor)

    hook.remove() 

    if not feature_maps:
        logger.error("EigenCAM hook did not capture any feature maps. The forward pass might have failed.")
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


def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    """Visualizes the original image, GT, prediction, and various superpixel maps."""
    if pred_mask.shape != gt_mask.shape:
        pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        pred_mask_resized = np.array(pred_mask_img.resize(gt_mask.shape[::-1], Image.NEAREST))
    else:
        pred_mask_resized = pred_mask

    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
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

def run_ranking_analysis(
    analysis_type, args, device, model,
    orig_image_np_01, pred_mask, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, current_image_output_dir,
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
        {"sp_id": int(sp_id), "iou": round(calculate_iou_score(pred_mask, gt_mask, segments_slic == sp_id), 2)}
        for sp_id in superpixels_to_measure
    ]
    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1):
        sp_data['rank'] = rank
    
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
        
        for noise_level in np.arange(0.05, 2.5, 0.05):
            perturbed_image_np_scaled, noise_char_value = perturb_superpixels_with_level(orig_image_np_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_np_01, perturbed_image_np_scaled, data_range=1.0, channel_axis=-1)

            if ssim_score < 0.8 and noise_type!='poisson':
                logger.warning(f"SSIM score {ssim_score:.4f} dropped below 0.8 for '{noise_type}'. Stopping noise application.")
                break
            
            if args.plot_ssim:
                ssim_data_collector.append({
                    "Analysis Type": analysis_type, "Noise Type": noise_type,
                    "Noise Level": noise_level, "SSIM": ssim_score
                })
            
            pil_img = Image.fromarray((perturbed_image_np_scaled * 255).astype(np.uint8))
            img_save_path = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.3f}.png")
            pil_img.save(img_save_path)
            
            perturbed_tensor = torch.from_numpy(perturbed_image_np_scaled).permute(2, 0, 1).float()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            perturbed_tensor_normalized = normalize(perturbed_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                res = model(perturbed_tensor_normalized)
                res = F.upsample(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                perturbed_pred_mask = np.where(res >= 0.5, 1, 0)

            current_level_sp_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(pred_mask, perturbed_pred_mask, segments_slic == sp_id), 2)} for sp_id in superpixels_to_measure]
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            for sp_id in superpixels_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))

            initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            current_rbo = calculate_rbo(initial_rank_list, current_rank_list, p=0.9)
            rbo_scores_list.append(round(current_rbo, 2))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
            level_data_list.append({noise_char_map.get(noise_type, 'noise_value'): round(noise_char_value, 2), "ssim_score": round(ssim_score, 2), "image_path": os.path.relpath(img_save_path, args.output_dir), "superpixels": sorted_superpixels})
        
        reliability_results = []
        for sp_id in superpixels_to_measure:
            avg_iou = np.mean(sp_iou_history[sp_id]) if sp_iou_history[sp_id] else 0.0
            initial_iou = initial_iou_map.get(sp_id, 0.0)
            
            rank_list = [r for r in sp_rank_history[sp_id] if r is not None]
            num_changes = sum(1 for i in range(1, len(rank_list)) if rank_list[i] != rank_list[i-1])
            sum_of_jumps = sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list)))
            rank_perplexity = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0

            delta_iou = avg_iou - initial_iou
            penalty_score = 0.0
            if abs(delta_iou) > 1e-9:
                exponent = rank_perplexity / abs(delta_iou)
                try:
                    factor = 1.0 - (1.0 / exp(exponent))
                except OverflowError:
                    factor = 1.0
                penalty_score = 2 * delta_iou * factor
            
            pIoU = avg_iou + penalty_score
            
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
                "initial_iou": round(initial_iou, 4),
                "rank_perplexity": round(rank_perplexity, 4)
            })
            
        avg_reliability_score = np.mean([res['final_reliability_score'] for res in reliability_results]) if reliability_results else 0
        logger.info(f"[{analysis_type.upper()}] FINAL Average Reliability Score for {noise_type}: {avg_reliability_score:.4f}")

        analysis_results[noise_type] = {
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "rbo_at_each_level": rbo_scores_list,
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['final_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results

def main():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--pth_path', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/ISIC2016_MSDUNet.pth', help='Path to the model weights.')
    parser.add_argument('--testsize', type=int, default=352, help='Testing size')
    parser.add_argument('--num_classes', type=int, default=1, help='Output channel of network')
    parser.add_argument('--dataroot_val', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test', help='Path to the validation dataset.')

    # Analysis options
    parser.add_argument('--target_layer_name', type=str, default='FU_E.conv1', help='Target layer for EigenCAM.')
    parser.add_argument('--noise_types', nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'], help='Types of noise to apply.')
    parser.add_argument('--noise_levels', type=int, default=10, help='Number of noise levels to test.')
    parser.add_argument('--superpixel_size', type=int, default=20, help='Superpixel size for SLIC.')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for selecting superpixels.')
    parser.add_argument('--plot_ssim', action='store_true', help='Enable plotting SSIM vs. Noise Level.')
    ### MODIFIED: Added perturb_ratio argument ###
    parser.add_argument('--perturb_ratio', type=float, default=.5, help='Fraction of qualified in-context superpixels to randomly select for perturbation and assessment (e.g., 0.8 for 80%%).')
    
    # Logging and output
    parser.add_argument('--log_dir', type=str, default='./expresults/logs', help='Directory for logs.')
    # parser.add_argument('--output_dir', type=str, default='./expresults_quickshift', help='Directory for output images and data.')
    parser.add_argument('--output_dir', type=str, default='./expresults', help='Directory for output images and data.')
    
    # iterations to run
    parser.add_argument('--num_iterations', type=int, default=20, help='Number of iterations to run the analysis.')
    args = parser.parse_args()
    
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('base')

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading model...")
    model_instance = model(num_classes=args.num_classes).to(device)
    try:
        model_instance.load_state_dict(torch.load(args.pth_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return
    model_instance.eval()
    logger.info("Model loaded successfully.")

    logger.info("Loading dataset...")
    image_root = os.path.join(args.dataroot_val, 'images/')
    gt_root = os.path.join(args.dataroot_val, 'masks/')
    test_loader = test_dataset(image_root, gt_root, args.testsize)
    num_images = len(os.listdir(gt_root))
    logger.info(f"Found {num_images} images in the dataset.")

    final_analysis_data = {}
    noise_char_map = {noise: f'{noise}_strength' for noise in args.noise_types}
    ssim_data_collector = []
    all_images_summary_metrics = {
        'rbo': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}},
        'reliability': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}}
    }
    logger.info(f"Starting analysis iterations...for {args.num_iterations} images")

    for i in range(args.num_iterations):
        image, gt, name = test_loader.load_data()
        gt_mask_np = np.asarray(gt, np.float32)
        gt_mask_np /= (gt_mask_np.max() + 1e-8)
        
        image_tensor = image.to(device)

        with torch.no_grad():
            res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(res >= 0.5, 1, 0)

        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)
        
        segments_slic = slic(orig_image_np_01, n_segments=200, compactness=10, sigma=1)
        # if orig_image_np_01.ndim == 2:  # Single-channel image
        #     img_quick = np.stack([orig_image_np_01] * 3, axis=-1)  # Create a 3-channel version for Quickshift
        # else:
        #     img_quick = orig_image_np_01
        # segments_slic = quickshift(img_quick, kernel_size=5, max_dist=10, ratio=0.7)


        eigencam_heatmap = calculate_eigencam(model_instance, args.target_layer_name, image_tensor, (orig_image_np_01.shape[0], orig_image_np_01.shape[1]))
        
        current_image_output_dir = os.path.join(args.output_dir, os.path.splitext(name)[0])
        os.makedirs(current_image_output_dir, exist_ok=True)
        
        if gt_mask_np.shape != segments_slic.shape:
            gt_pil = Image.fromarray((gt_mask_np * 255).astype(np.uint8))
            gt_mask_resized_pil = gt_pil.resize(segments_slic.shape[::-1], Image.NEAREST)
            gt_mask_for_sp = (np.array(gt_mask_resized_pil) > 127).astype(np.uint8)
        else:
            gt_mask_for_sp = gt_mask_np.astype(np.uint8)

        in_context_superpixels_map = get_refined_superpixel_map(gt_mask_for_sp, segments_slic)
        out_of_context_superpixels_map = segments_slic.copy()
        out_of_context_superpixels_map[gt_mask_for_sp == 1] = 0

        visualize_segmentation_comparison(orig_image_np_01, gt_mask_for_sp, pred_mask, segments_slic, in_context_superpixels_map, out_of_context_superpixels_map, eigencam_heatmap, os.path.join(current_image_output_dir, name.rsplit('.', 1)[0] + "_overview.png"))
        
        ### MODIFIED: Added full filtering and sampling logic ###
        all_in_context_sp_ids = np.unique(in_context_superpixels_map)
        all_in_context_sp_ids = all_in_context_sp_ids[all_in_context_sp_ids != 0]

        qualified_in_context_sp = []
        if all_in_context_sp_ids.size > 0:
            logger.info(f"Found {len(all_in_context_sp_ids)} superpixels overlapping with GT. Filtering for IoU > 50...")
            for sp_id in all_in_context_sp_ids:
                iou = calculate_iou_score(pred_mask, gt_mask_np, segments_slic == sp_id)
                if iou > 50:
                    qualified_in_context_sp.append(sp_id)
            logger.info(f"{len(qualified_in_context_sp)} superpixels qualified with IoU > 50.")
        
        if not qualified_in_context_sp:
            logger.warning(f"No superpixels for {name} met the >50 IoU threshold. Skipping analysis for this image.")
            continue

        # Randomly sample a percentage of the qualified superpixels for the in-context analysis
        num_to_select = int(len(qualified_in_context_sp) * args.perturb_ratio)
        in_context_sp_for_analysis = random.sample(qualified_in_context_sp, num_to_select)
        logger.info(f"Randomly selected {len(in_context_sp_for_analysis)} ({args.perturb_ratio*100}%) qualified superpixels for in-context analysis.")

        out_of_context_sp_ids = np.unique(out_of_context_superpixels_map)
        out_of_context_sp_ids = out_of_context_sp_ids[out_of_context_sp_ids != 0]

        final_analysis = {"image_info": {"name": name.rsplit('.', 1)[0], "initial_dice_vs_gt": calculate_dice_score(gt_mask_np, pred_mask)}}

        if in_context_sp_for_analysis:
            final_analysis['in_context_analysis'] = run_ranking_analysis(
                analysis_type='in_context', args=args, device=device, model=model_instance,
                orig_image_np_01=orig_image_np_01, pred_mask=pred_mask, gt_mask=gt_mask_np, 
                segments_slic=segments_slic, superpixels_to_perturb=in_context_sp_for_analysis, 
                superpixels_to_measure=in_context_sp_for_analysis, noise_types=args.noise_types, 
                noise_char_map=noise_char_map, current_image_output_dir=current_image_output_dir,
                ssim_data_collector=ssim_data_collector
            )

        # For out-of-context, we still measure the effect on ALL qualified superpixels
        if len(out_of_context_sp_ids) > 0 and qualified_in_context_sp:
            final_analysis['out_of_context_analysis'] = run_ranking_analysis(
                analysis_type='out_of_context', args=args, device=device, model=model_instance,
                orig_image_np_01=orig_image_np_01, pred_mask=pred_mask, gt_mask=gt_mask_np,
                segments_slic=segments_slic, superpixels_to_perturb=out_of_context_sp_ids,
                superpixels_to_measure=qualified_in_context_sp, noise_types=args.noise_types,
                noise_char_map=noise_char_map, current_image_output_dir=current_image_output_dir,
                ssim_data_collector=ssim_data_collector
            )

        final_analysis = json.loads(json.dumps(final_analysis, default=lambda o: float(o) if isinstance(o, np.floating) else o))

        json_save_path = os.path.join(current_image_output_dir, f"{name.rsplit('.', 1)[0]}_analysis.json")
        with open(json_save_path, 'w') as f:
            json.dump(final_analysis, f, indent=4)
        logger.info(f"Full analysis saved to {json_save_path}")
        
        for analysis_type in ['in_context', 'out_of_context']:
            if f'{analysis_type}_analysis' in final_analysis:
                for noise_type, results in final_analysis[f'{analysis_type}_analysis'].items():
                    all_images_summary_metrics['rbo'][analysis_type][noise_type].append(results['average_rank_biased_overlap'])
                    # ### MODIFIED: Changed reliability_score to final_reliability_score to prevent KeyError ###
                    scores = [res['final_reliability_score'] for res in results.get('reliability_analysis', [])]
                    if scores:
                        all_images_summary_metrics['reliability'][analysis_type][noise_type].append(np.mean(scores))

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
                rbo_plot_data.append({'Analysis Type': analysis_type.replace('_', ' ').title(), 'Noise Type': noise_type, 'RBO Score': np.clip(score, 0, 1)})

    if rbo_plot_data:
        df_rbo = pd.DataFrame(rbo_plot_data)
        g = sns.catplot(
            data=df_rbo,
            x='Noise Type',
            y='RBO Score',
            col='Analysis Type',
            kind='violin',
            palette='viridis',
            sharey=True,
            height=6,
            aspect=1.2
        )
        g.fig.suptitle('RBO Score Distribution by Noise Type', y=1.03, fontsize=16)
        g.set_axis_labels("Noise Type", "RBO Score")
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles("{col_name} Analysis")

        # Limit x-axis to [0, 1]
        for ax in g.axes.flat:
            ax.set_ylim(0, 1)

        plt.savefig(os.path.join(args.output_dir, "RBO_Distributions_ViolinPlot.png"), bbox_inches='tight', dpi=300)
        plt.close(g.fig)

    logger.info("--- Generating Reliability Score Summary Plot ---")
    reliability_plot_data = []
    for analysis_type, noise_data in all_images_summary_metrics['reliability'].items():
        for noise_type, scores in noise_data.items():
            if scores:
                reliability_plot_data.append({'Analysis Type': analysis_type.replace('_', ' ').title(), 'Noise Type': noise_type, 'Average Reliability Score': np.mean(scores)})
    
    if reliability_plot_data:
        df_reliability = pd.DataFrame(reliability_plot_data)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_reliability, x='Noise Type', y='Average Reliability Score', hue='Analysis Type', palette='mako')
        plt.title('Average Model Reliability by Noise Type', fontsize=16)
        plt.xlabel('Noise Type', fontsize=12); plt.ylabel('Average Reliability Score', fontsize=12); plt.xticks(rotation=45, ha='right'); plt.legend(title='Analysis Type'); plt.tight_layout()
        save_path = os.path.join(args.output_dir, "final_reliability_summary.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved reliability summary plot to {save_path}")

    if args.plot_ssim and ssim_data_collector:
        logger.info("--- Preparing SSIM data and plots ---")
        df_ssim = pd.DataFrame(ssim_data_collector)
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

    logger.info("--- Generating Final Reliability Summary Plots ---")
    reliability_scores_map = all_images_summary_metrics['reliability']
    for analysis_type, noise_data in reliability_scores_map.items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7))
            bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='viridis')
            plt.title(f"Final Reliability Score ({analysis_type.replace('_', ' ').title()})", fontsize=16)
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
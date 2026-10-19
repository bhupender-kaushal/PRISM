import torch
import numpy as np
import os
import time
import argparse
import logging
from math import ceil
from PIL import Image
import random
import json

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

# --- START: Helper functions for superpixel processing and visualization ---
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import slic, mark_boundaries
from skimage.metrics import structural_similarity

# Imports for REAL EigenCAM
from sklearn.decomposition import PCA
import torch.nn.functional as F

# --- Scoring and Selection Functions ---
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
            noise_char_value = noise_level
            region = perturbed_image[superpixel_mask]
            noisy_region = np.random.poisson(region * 25 * noise_char_value) / (25 * noise_char_value)
            perturbed_image[superpixel_mask] = noisy_region
        elif noise_type == 'speckle':
            noise_char_value = 0.1 * noise_level
            noise = np.random.normal(0, noise_char_value, perturbed_image[superpixel_mask].shape)
            perturbed_image[superpixel_mask] *= (1 + noise)
    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

def calculate_eigencam(diffusion_netG, target_layer_name, single_channel_input_tensor, image_size):
    """Calculates the EigenCAM heatmap."""
    feature_maps = []
    model_instance = diffusion_netG.segment_fn
    def hook_fn(module, input, output): feature_maps.append(output)
    target_layer = dict(model_instance.named_modules())[target_layer_name]
    hook = target_layer.register_forward_hook(hook_fn)
    input_for_fn = torch.cat([single_channel_input_tensor, single_channel_input_tensor], dim=1)
    model_instance.to(input_for_fn.device).eval()
    with torch.no_grad(): _ = model_instance(input_for_fn)
    hook.remove()
    features = feature_maps[0].squeeze(0)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    pca = PCA(n_components=1).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
    return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, original_superpixels_segments, in_context_superpixel_map, eigencam_heatmap, save_path=None):
    """Visualizes the original image, GT, prediction, superpixels, and EigenCAM."""
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    axes[0].imshow(display_image_rgb); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
    axes[3].imshow(mark_boundaries(display_image_rgb, original_superpixels_segments)); axes[3].set_title(f"All Superpixels"); axes[3].axis('off')
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_superpixel_map)); axes[4].set_title("In-Context Superpixels (GT)"); axes[4].axis('off')
    axes[5].imshow(display_image_rgb); im = axes[5].imshow(eigencam_heatmap, cmap='jet', alpha=0.5); axes[5].set_title("EigenCAM"); axes[5].axis('off')
    plt.colorbar(im, ax=axes[5])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Segmentation Prediction and Superpixel Ranking Pipeline")
    parser.add_argument('-c', '--config', type=str, default='config/test.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode with reduced dataset size and frequency')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./expresults/mouse_ranked')
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--perturb_ratio', type=float, default=1.0)
    parser.add_argument('--rank_track_ratio', type=float, default=100.0)
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir: opt['datasets']['test']['dataroot'] = args.data_dir
    opt['phase'] = args.phase
    opt['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else []
    
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    Logger.setup_logger(None, log_dir, opt['phase'], level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    dataset_opt = opt['datasets']['test']
    val_set = Data.create_dataset_xcad(dataset_opt, 'test')
    val_loader = Data.create_dataloader(val_set, dataset_opt, 'test')
    logger.info(f'Test dataset loaded with {len(val_set)} samples.')

    diffusion = Model.create_model(opt)
    logger.info('Model Finished Initialization.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    diffusion.netG.to(device)

    logger.info(f"--- Starting Superpixel Ranking Pipeline for {args.num_iterations} images ---")
    
    noise_char_map = {'salt_pepper': 'noise_density', 'gaussian': 'noise_std_dev', 'speckle': 'noise_std_dev', 'poisson': 'noise_intensity_scale'}

    for idata, val_data in enumerate(val_loader):
        if idata >= args.num_iterations: break
        dataInfo = os.path.basename(val_data['P'][0]).replace('.png', '')
        current_image_output_dir = os.path.join(args.output_dir, dataInfo)
        os.makedirs(current_image_output_dir, exist_ok=True)
        logger.info(f"Processing image: {dataInfo} ({idata + 1}/{args.num_iterations})")

        # --- 1. Initial Prediction and Data Prep ---
        diffusion.feed_data(val_data); diffusion.test()
        visuals = diffusion.get_current_segment()
        predseg_raw = visuals['test_V'].squeeze().cpu().numpy()
        predseg = ((predseg_raw + 1) / 2.0 > 0.5).astype(np.uint8)

        orig_image_tensor_raw = val_data['A'][0]
        orig_image_np = orig_image_tensor_raw.squeeze().cpu().numpy()
        orig_image_scaled_for_display = (orig_image_np + 1) / 2.0

        gt_mask_raw = val_data['F'][0].squeeze().cpu().numpy()
        gt_mask_np = ((gt_mask_raw + 1) / 2.0 > 0.5).astype(np.uint8)
        dice_pred_vs_gt = calculate_dice_score(gt_mask_np, predseg)
        logger.info(f"Initial Dice Score (Pred vs. GT): {dice_pred_vs_gt:.2f}%")
        
        # --- 2. Superpixel Generation and Selection ---
        segments_slic = slic(orig_image_scaled_for_display, n_segments=300, compactness=0.1, sigma=1.0, channel_axis=None)
        iou_qualified_sp_ids = select_superpixels_by_pred_gt_iou(segments_slic, predseg, gt_mask_np, iou_threshold=0.5)
        
        if not iou_qualified_sp_ids:
            logger.warning("No superpixels met the 50% Pred/GT IoU threshold. Skipping ranking for this image.")
            continue
        logger.info(f"Found {len(iou_qualified_sp_ids)} superpixels with >= 50% Pred/GT IoU.")

        num_to_select_for_ranking = max(1, int(len(iou_qualified_sp_ids) * args.perturb_ratio))
        selected_sp_ids_for_ranking = random.sample(iou_qualified_sp_ids, num_to_select_for_ranking)
        logger.info(f"Randomly selected {len(selected_sp_ids_for_ranking)} of them for ranking analysis.")
        
        # --- 3. EigenCAM and Initial Visualization ---
        in_context_superpixel_map = get_refined_superpixel_map(gt_mask_np, segments_slic)
        try:
            input_tensor_for_model = orig_image_tensor_raw.unsqueeze(0).float().to(device)
            input_tensor_for_eigencam = F.interpolate(input_tensor_for_model, size=(256, 256), mode='bilinear', align_corners=False)
            original_eigencam_heatmap = calculate_eigencam(diffusion.netG, 'Conv', input_tensor_for_eigencam, orig_image_np.shape)
        except Exception as e:
            logger.error(f"Error calculating EigenCAM: {e}. Using blank heatmap.")
            original_eigencam_heatmap = np.zeros_like(orig_image_np, dtype=np.float32)
        visualize_segmentation_comparison(orig_image_scaled_for_display, gt_mask_np, predseg, segments_slic, in_context_superpixel_map, original_eigencam_heatmap,
                                          save_path=os.path.join(current_image_output_dir, f"{dataInfo}_overview.png"))

        # --- 4. Superpixel Ranking Workflow ---
        analysis_results = {"image_info": {"name": dataInfo, "initial_dice_vs_gt": round(dice_pred_vs_gt, 2), "pertubed_superpixel_ids": [int(x) for x in selected_sp_ids_for_ranking]}}
        noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper']

        initial_sp_iou_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(predseg, gt_mask_np, segments_slic == sp_id), 2)} for sp_id in selected_sp_ids_for_ranking]
        sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
        for rank, sp_data in enumerate(sorted_initial_sp_data, 1): sp_data['rank'] = rank

        for noise_type in noise_types:
            logger.info(f"--- Analyzing Noise Type: {noise_type.upper()} ---")
            noise_specific_dir = os.path.join(current_image_output_dir, noise_type); os.makedirs(noise_specific_dir, exist_ok=True)
            
            level_0_entry = {noise_char_map[noise_type]: 0.0, "ssim_score": 1.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
            level_data_list = [level_0_entry]
            
            num_to_track = ceil(len(sorted_initial_sp_data) * (args.rank_track_ratio / 100.0))
            num_to_track = max(1, min(num_to_track, len(sorted_initial_sp_data)))
            sp_ids_to_track = {sp['sp_id'] for sp in sorted_initial_sp_data[:num_to_track]}
            logger.info(f"Tracking top {len(sp_ids_to_track)} superpixels ({args.rank_track_ratio}%) for rank stability.")

            rank_change_counts = {sp_id: 0 for sp_id in sp_ids_to_track}
            previous_ranks = {sp['sp_id']: sp['rank'] for sp in sorted_initial_sp_data if sp['sp_id'] in sp_ids_to_track}
            
            # --- MODIFIED: List to store RBO scores at each level ---
            rbo_scores_list = []

            for noise_level in np.arange(0.1, 3.0, 0.1):
                perturbed_image_np_scaled, noise_char_value = perturb_superpixels_with_level(orig_image_scaled_for_display, segments_slic, selected_sp_ids_for_ranking, noise_type, noise_level)
                ssim_score = structural_similarity(orig_image_scaled_for_display, perturbed_image_np_scaled, data_range=1.0)
                if ssim_score < 0.90:
                    logger.info(f"Stopping {noise_type} at level {noise_level:.2f}, SSIM dropped to {ssim_score:.3f}")
                    break
                
                pil_img = Image.fromarray((perturbed_image_np_scaled * 255).astype(np.uint8))
                img_save_path = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.3f}.png")
                pil_img.save(img_save_path)
                
                perturbed_image_tensor = torch.from_numpy(perturbed_image_np_scaled[None, None, ...]).float()
                perturbed_input_for_netG = ((perturbed_image_tensor * 2) - 1).to(device)
                val_data_perturbed = val_data.copy(); val_data_perturbed['A'] = perturbed_input_for_netG
                diffusion.feed_data(val_data_perturbed); diffusion.test()
                perturbed_predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
                perturbed_predseg = ((perturbed_predseg_raw + 1) / 2.0 > 0.5).astype(np.uint8)

                current_level_sp_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(predseg, perturbed_predseg, segments_slic == sp_id), 2)} for sp_id in selected_sp_ids_for_ranking]
                sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)

                # --- MODIFIED: Calculate RBO for the current level ---
                initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
                current_rank_list = [d['sp_id'] for d in sorted_superpixels]
                current_rbo = calculate_rbo(initial_rank_list, current_rank_list, p=0.9)
                rbo_scores_list.append(round(current_rbo, 4))
                
                for current_rank, sp_data in enumerate(sorted_superpixels, 1):
                    sp_id = sp_data['sp_id']
                    if sp_id in sp_ids_to_track:
                        if current_rank != previous_ranks.get(sp_id): rank_change_counts[sp_id] += 1
                        previous_ranks[sp_id] = current_rank
                
                for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
                level_data_list.append({noise_char_map[noise_type]: round(noise_char_value, 4), "ssim_score": round(ssim_score, 4), "image_path": os.path.relpath(img_save_path, args.output_dir), "superpixels": sorted_superpixels})
            
            # --- MODIFIED: Calculate average RBO and prepare results ---
            average_rbo = np.mean(rbo_scores_list) if rbo_scores_list else 0.0
            logger.info(f"Average Rank-Biased Overlap (RBO) for {noise_type}: {average_rbo:.4f}")

            stability_results = [{"sp_id": sp_id, "rank_changes": count} for sp_id, count in rank_change_counts.items()]
            analysis_results[noise_type] = {
                "average_rank_biased_overlap": round(average_rbo, 4),
                "rbo_at_each_level": rbo_scores_list,
                "rank_stability_analysis": sorted(stability_results, key=lambda x: x['rank_changes']),
                "level_data": level_data_list
            }

        # --- 5. Save Results ---
        json_save_path = os.path.join(current_image_output_dir, f"{dataInfo}_analysis.json")
        with open(json_save_path, 'w') as f: json.dump(analysis_results, f, indent=4)
        logger.info(f"Full analysis saved to {json_save_path}")

    logger.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
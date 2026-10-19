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
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from scipy.ndimage import gaussian_filter

# Imports for REAL EigenCAM
from sklearn.decomposition import PCA
import torch.nn.functional as F

# --- Standard Dice Score function for global comparison ---
# --- MODIFIED: Standard Dice Score function using user's preferred logic ---

def calculate_dice_score(mask1, mask2, label=1):
    """
    Calculates the Dice Similarity Coefficient (DSC) between two binary masks.

    Args:
        mask1 (np.array): The first binary mask.
        mask2 (np.array): The second binary mask.
        label (int): The foreground label value to compare.

    Returns:
        float: The Dice score as a percentage.
    """
    if mask1.shape != mask2.shape:
        # Resize mask1 to match mask2's shape for comparison
        pil_img = Image.fromarray(mask1.astype(np.uint8))
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img)


    # Convert masks to 0/1 integer arrays for calculation
    m1 = (mask1 == label).astype(int)
    m2 = (mask2 == label).astype(int)

    intersect = float(np.sum(m1 * m2))
    sum_of_masks = float(np.sum(m1) + np.sum(m2))

    if sum_of_masks == 0:
        # If both masks are empty, they are identical.
        return 100.0

    dice_score = ((2. * intersect) / sum_of_masks) * 100
    return dice_score

# --- MODIFIED: Masked Dice Score function using user's preferred logic ---

def calculate_masked_dice_score(mask1, mask2, roi_mask, label=1):
    """
    Calculates the Dice Similarity Coefficient (DSC) between two binary masks,
    considering only the pixels within a given Region of Interest (ROI).

    Args:
        mask1 (np.array): The first binary mask (e.g., original prediction).
        mask2 (np.array): The second binary mask (e.g., perturbed prediction).
        roi_mask (np.array): A boolean mask defining the region of interest.
        label (int): The foreground label value to compare (e.g., 1).

    Returns:
        float: The Dice score within the ROI as a percentage.
    """
    # Ensure the ROI mask is boolean
    roi_mask = roi_mask.astype(bool)
    if mask1.shape != mask2.shape:
        pil_img = Image.fromarray(mask1.astype(np.uint8))
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img)

    if mask1.shape != roi_mask.shape:
        pil_img_roi = Image.fromarray(roi_mask.astype(np.uint8))
        resized_img_roi = pil_img_roi.resize(mask1.shape[::-1], Image.NEAREST)
        roi_mask = np.array(resized_img_roi).astype(bool)


    # Apply the ROI mask to both segmentation masks
    mask1_roi = mask1[roi_mask]
    mask2_roi = mask2[roi_mask]

    # Convert ROI masks to 0/1 integer arrays
    m1_roi = (mask1_roi == label).astype(int)
    m2_roi = (mask2_roi == label).astype(int)

    intersect = float(np.sum(m1_roi * m2_roi))
    sum_of_masks = float(np.sum(m1_roi) + np.sum(m2_roi))

    if sum_of_masks == 0:
        # If both masks are empty within the ROI, they are identical.
        return 100.0

    dice_score = ((2. * intersect) / sum_of_masks) * 100
    return dice_score

def get_refined_superpixel_map(gt_mask, superpixels_segments, foreground_class_id=1):
    """Creates an in-context superpixel map by clipping superpixels to the GT foreground."""
    gt_foreground = (gt_mask == foreground_class_id).astype(bool)
    in_context_segments = np.copy(superpixels_segments)
    in_context_segments[~gt_foreground] = 0
    return in_context_segments

def select_random_superpixels(superpixel_map, num_to_select):
    """
    Selects a specified number of unique superpixel IDs randomly from a superpixel map,
    excluding the background label (0).
    """
    unique_labels = np.unique(superpixel_map)
    # Exclude background label if 0 is used for background
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) == 0:
        return []  # No superpixels to select from

    if num_to_select >= len(unique_labels):
        return list(unique_labels)  # Select all if requested number is too high
    else:
        return random.sample(list(unique_labels), num_to_select)

def perturb_superpixels(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type):
    """
    Perturbs selected superpixel regions in an image with different noise types.
    """
    perturbed_image = np.copy(original_image_scaled)
    is_grayscale = original_image_scaled.ndim == 2

    for sp_id in selected_superpixel_ids:
        superpixel_mask = (superpixels_segments == sp_id)
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.05, perturbed_image[superpixel_mask].shape).astype(np.float32)
            perturbed_region = perturbed_image[superpixel_mask] + noise
            perturbed_image[superpixel_mask] = np.clip(perturbed_region, 0.0, 1.0)
        elif noise_type == 'salt_pepper':
            s_p_ratio = 0.05
            num_pixels_in_sp = np.sum(superpixel_mask)
            num_s_p_pixels = int(num_pixels_in_sp * s_p_ratio)
            sp_indices_rows, sp_indices_cols = np.where(superpixel_mask)
            if num_pixels_in_sp > 0:
                s_p_indices = random.sample(range(num_pixels_in_sp), num_s_p_pixels)
                for idx in s_p_indices:
                    r, c = sp_indices_rows[idx], sp_indices_cols[idx]
                    if is_grayscale:
                        perturbed_image[r, c] = 0.0 if random.random() < 0.5 else 1.0
                    else:
                        perturbed_image[r, c, :] = 0.0 if random.random() < 0.5 else 1.0
        elif noise_type == 'poisson':
            temp_region = (perturbed_image[superpixel_mask] * 255).astype(np.float32)
            noise = np.random.poisson(lam=5, size=temp_region.shape).astype(np.float32) - 5
            perturbed_region = (temp_region + noise) / 255.0
            perturbed_image[superpixel_mask] = np.clip(perturbed_region, 0.0, 1.0)
        elif noise_type == 'speckle':
            noise = np.random.normal(0, 0.05, perturbed_image[superpixel_mask].shape).astype(np.float32)
            perturbed_region = perturbed_image[superpixel_mask] * (1 + noise)
            perturbed_image[superpixel_mask] = np.clip(perturbed_region, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    return perturbed_image

def calculate_eigencam(diffusion_netG, target_layer_name, single_channel_input_tensor, image_size):
    """
    Calculates the EigenCAM heatmap for a given PyTorch model.
    The input tensor is processed, and the resulting heatmap is upsampled to the target image_size.
    """
    feature_maps = []
    model_instance_for_eigencam = diffusion_netG.segment_fn

    def hook_fn(module, input, output):
        feature_maps.append(output)

    target_layer = None
    for name, module in model_instance_for_eigencam.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    if target_layer is None:
        raise ValueError(f"Target layer '{target_layer_name}' not found in the model '{model_instance_for_eigencam.__class__.__name__}'.")

    hook = target_layer.register_forward_hook(hook_fn)
    # The input tensor might be of a different size (e.g., 256x256)
    input_for_segment_fn = torch.cat([single_channel_input_tensor, single_channel_input_tensor], dim=1)
    model_instance_for_eigencam = model_instance_for_eigencam.to(input_for_segment_fn.device)

    model_instance_for_eigencam.eval()
    with torch.no_grad():
        _ = model_instance_for_eigencam(input_for_segment_fn)

    hook.remove()

    if not feature_maps:
        raise RuntimeError(f"No feature maps captured from layer '{target_layer_name}'. Check model and layer name.")

    features = feature_maps[0].squeeze(0)
    if features.ndim != 3:
        raise RuntimeError(f"Expected 3D feature map (C, H, W) after squeeze(0), but got {features.ndim} dimensions. Shape: {features.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0 or features.shape[2] == 0:
        raise RuntimeError(f"Feature map has zero dimensions after squeeze(0). Shape: {features.shape}")

    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()

    if np.isnan(features_flat).any() or np.isinf(features_flat).any():
        raise ValueError("Features for PCA contain NaN or Inf values. Check model output or input scaling.")

    n_samples, n_features = features_flat.shape
    n_components = min(1, n_samples, n_features)
    if n_components == 0:
        logging.getLogger('base').warning("PCA n_components is 0 due to insufficient data. Returning zero heatmap.")
        return np.zeros(image_size, dtype=np.float32)

    pca = PCA(n_components=n_components)
    pca.fit(features_flat)
    principal_component = pca.components_[0]

    saliency_map = (features.cpu().numpy() * principal_component[:, None, None]).sum(axis=0)
    saliency_map = np.maximum(saliency_map, 0)

    saliency_map_min = saliency_map.min()
    saliency_map_max = saliency_map.max()
    if saliency_map_max > saliency_map_min:
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
    else:
        saliency_map = np.zeros_like(saliency_map)
        
    saliency_map_tensor = torch.from_numpy(saliency_map.astype(np.float32))[np.newaxis, np.newaxis, :, :].float()
    
    # Upsample the heatmap to the original image size for correct visualization
    saliency_map_upsampled = F.interpolate(saliency_map_tensor,
                                         size=image_size,
                                         mode='bilinear',
                                         align_corners=False).squeeze().numpy()
    return saliency_map_upsampled

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, original_superpixels_segments, in_context_superpixel_map, out_of_context_superpixel_map, eigencam_heatmap, foreground_class_id=1, save_path=None):
    """
    Visualizes the original image, ground truth, predicted mask, original superpixels,
    in-context (GT-clipped) superpixels, out-of-context superpixels, and EigenCAM
    in a single comprehensive plot.
    """
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(display_image_rgb)
    axes[1].imshow(gt_mask.squeeze(), cmap='gray', alpha=1.0)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(display_image_rgb)
    axes[2].imshow(pred_mask, cmap='gray', alpha=1.0)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    original_superpixel_image = mark_boundaries(display_image_rgb, original_superpixels_segments, color=(1, 1, 0), outline_color=(0, 0, 0))
    axes[3].imshow(original_superpixel_image)
    axes[3].set_title(f"Original Superpixels (N={len(np.unique(original_superpixels_segments))})")
    axes[3].axis('off')

    gt_foreground = (gt_mask == foreground_class_id).astype(bool)
    temp_image_with_boundaries_in_context = mark_boundaries(display_image_rgb, in_context_superpixel_map,
                                                           color=(1, 1, 0), outline_color=(0, 0, 0))
    faded_image_outside_gt = np.copy(temp_image_with_boundaries_in_context)
    faded_image_outside_gt[~gt_foreground] = faded_image_outside_gt[~gt_foreground] * 0.2
    gt_boundary_pixels = find_boundaries(gt_foreground, mode='inner').astype(bool)
    faded_image_outside_gt[gt_boundary_pixels] = [1, 1, 0]

    num_in_context_superpixels = len(np.unique(in_context_superpixel_map))
    if 0 in np.unique(in_context_superpixel_map):
        num_in_context_superpixels -= 1

    axes[4].imshow(faded_image_outside_gt)
    axes[4].set_title(f"In-Context Superpixels (N={num_in_context_superpixels})")
    axes[4].axis('off')

    out_of_context_display_image = np.copy(display_image_rgb)
    num_out_of_context_superpixels = len(np.unique(out_of_context_superpixel_map))
    if 0 in np.unique(out_of_context_superpixel_map):
        num_out_of_context_superpixels -= 1

    out_of_context_boundaries_image = mark_boundaries(out_of_context_display_image, out_of_context_superpixel_map,
                                                     color=(1, 1, 0), outline_color=(0, 0, 0))
    faded_out_of_context_image = np.copy(out_of_context_boundaries_image)
    faded_out_of_context_image[out_of_context_superpixel_map == 0] = faded_out_of_context_image[out_of_context_superpixel_map == 0] * 0.2

    axes[5].imshow(faded_out_of_context_image)
    axes[5].set_title(f"Out-of-Context Superpixels (N={num_out_of_context_superpixels})")
    axes[5].axis('off')

    axes[6].imshow(display_image_rgb)
    im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1.0)
    axes[6].set_title("EigenCAM")
    axes[6].axis('off')
    plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logging.getLogger('base').info(f"Figure saved to: {save_path}")
    plt.close(fig)
    return fig

def visualize_perturbation_analysis(orig_image, perturbed_image, perturbed_pred_mask, perturbed_eigencam, original_pred_mask, original_eigencam, noise_type, save_path=None):
    """
    Visualizes the original image, a perturbed image, its segmentation, and its EigenCAM,
    alongside the original prediction and EigenCAM for comparison.
    """
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    perturbed_image_rgb = gray2rgb(perturbed_image) if perturbed_image.ndim == 2 else perturbed_image

    fig, axes = plt.subplots(1, 6, figsize=(36, 6))

    axes[0].imshow(display_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(perturbed_image_rgb)
    axes[1].set_title(f"Perturbed ({noise_type})")
    axes[1].axis('off')

    axes[2].imshow(display_image_rgb)
    axes[2].imshow(original_pred_mask, cmap='gray', alpha=1.0)
    axes[2].set_title("Original Prediction")
    axes[2].axis('off')

    axes[3].imshow(perturbed_image_rgb)
    axes[3].imshow(perturbed_pred_mask, cmap='gray', alpha=1.0)
    axes[3].set_title(f"Segmentation ({noise_type})")
    axes[3].axis('off')

    axes[4].imshow(display_image_rgb)
    im_original_cam = axes[4].imshow(original_eigencam, cmap='jet', alpha=1.0)
    axes[4].set_title("Original EigenCAM")
    axes[4].axis('off')
    plt.colorbar(im_original_cam, ax=axes[4], fraction=0.046, pad=0.04)

    axes[5].imshow(perturbed_image_rgb)
    im_perturbed_cam = axes[5].imshow(perturbed_eigencam, cmap='jet', alpha=1.0)
    axes[5].set_title(f"EigenCAM ({noise_type})")
    axes[5].axis('off')
    plt.colorbar(im_perturbed_cam, ax=axes[5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logging.getLogger('base').info(f"Figure saved to: {save_path}")
    plt.close(fig)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Segmentation Prediction and Visualization Pipeline")
    parser.add_argument('-c', '--config', type=str, default='config/test.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                       help='Run either train(training) or test(inference)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing input images and ground truth masks (overrides config if specified).')
    parser.add_argument('--output_dir', type=str, default='./expresults/mouse',
                       help='Directory for output logs and visualization results.')
    parser.add_argument('--num_iterations', type=int, default=50,
                       help='Number of validation iterations (images) to process.')
    parser.add_argument('--perturb_ratio', type=float, default=0.2,
                       help='Ratio of in-context superpixels to perturb (e.g., 0.1 for 10%).')
    parser.add_argument('--num_out_of_context_perturb_sp', type=int, default=20,
                       help='Number of out-of-context superpixels to perturb. Set to >0 to enable this analysis.')

    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir:
        if 'datasets' in opt and 'val' in opt['datasets']:
            opt['datasets']['val']['dataroot'] = args.data_dir
        elif 'datasets' in opt and 'test' in opt['datasets']:
            opt['datasets']['test']['dataroot'] = args.data_dir
        else:
            opt['datasets'] = {'test': {'name': 'xcad_test', 'mode': args.phase, 'dataroot': args.data_dir}}
            logging.getLogger('base').warning("Dataset configuration not found in opt. Using command-line --data_dir for 'test' dataset.")

    opt['phase'] = args.phase

    if args.gpu_ids:
        opt['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
    elif opt['gpu_ids'] is None:
        opt['gpu_ids'] = []

    if 'path' not in opt:
        opt['path'] = {}
    if 'log' not in opt['path']:
        opt['path']['log'] = './logs'
    if 'results' not in opt['path']:
        opt['path']['results'] = './results'

    log_base_path_from_opt = opt['path']['log']
    if os.path.splitext(log_base_path_from_opt)[1]:
        final_log_dir = os.path.dirname(log_base_path_from_opt)
    else:
        final_log_dir = log_base_path_from_opt

    os.makedirs(final_log_dir, exist_ok=True)

    Logger.setup_logger(None, final_log_dir, opt['phase'], level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    dataset_opt = opt['datasets']['test']
    phase = 'test'
    val_set = Data.create_dataset_xcad(dataset_opt, phase)
    val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info(f'Test dataset loaded with {len(val_set)} samples.')

    diffusion = Model.create_model(opt)
    logger.info('Model Finished Initialization.')

    device = torch.device('cuda:0')
    logger.info(f"CUDA available, using default GPU:0.")
    diffusion.netG.to(device)

    total_inference_time = 0
    processed_samples = 0

    logger.info(f"Starting prediction, superpixel refinement, and perturbation analysis for up to {args.num_iterations} iterations...")

    for idata, val_data in enumerate(val_loader):
        if processed_samples >= args.num_iterations:
            break

        dataInfo = os.path.basename(val_data['P'][0]).replace('.png', '')

        current_image_output_dir = os.path.join(args.output_dir, dataInfo)
        os.makedirs(current_image_output_dir, exist_ok=True)
        dice_results = {
            "prediction_vs_ground_truth": None,
            "pred_vs_pred_on_in_context_superpixels": None,
            "in_context_perturbation": {},
            "out_of_context_perturbation": {}
        }

        torch.cuda.synchronize()
        time1 = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Processing image: {dataInfo}')

        orig_image_tensor_raw = val_data['A'][0]
        # The model will process the image at its original resolution
        input_tensor_for_model = orig_image_tensor_raw.unsqueeze(0).float().to(device)

        diffusion.feed_data(val_data)
        diffusion.test()

        torch.cuda.synchronize()
        time2 = time.time()

        inference_time = time2 - time1
        total_inference_time += inference_time
        processed_samples += 1

        visuals = diffusion.get_current_segment()
        predseg_raw = visuals['test_V'].squeeze().cpu().numpy()
        predseg_scaled = (predseg_raw + 1) / 2.0
        predseg = (predseg_scaled > 0.5).astype(np.uint8)

        orig_image_np_raw = input_tensor_for_model.squeeze(0).cpu().numpy()
        if orig_image_np_raw.ndim == 3:
            if orig_image_np_raw.shape[0] == 3:
                orig_image_np = rgb2gray(orig_image_np_raw.transpose(1, 2, 0))
            elif orig_image_np_raw.shape[0] == 1:
                orig_image_np = orig_image_np_raw.squeeze(0)
            else:
                logger.warning(f"Unexpected channels: {orig_image_np_raw.shape[0]}. Squeezing to 2D.")
                orig_image_np = orig_image_np_raw.squeeze()
        else:
            orig_image_np = orig_image_np_raw

        orig_image_scaled_for_display = (orig_image_np + 1) / 2.0
        orig_image_scaled_for_display = orig_image_scaled_for_display.astype(np.float32)

        gt_mask_tensor_raw = val_data['F'][0]
        gt_mask_tensor_for_processing = gt_mask_tensor_raw.unsqueeze(0).float().to(device)
        
        gt_mask_raw = gt_mask_tensor_for_processing.squeeze().cpu().numpy()
        gt_mask_scaled = (gt_mask_raw + 1) / 2.0
        gt_mask_np = (gt_mask_scaled > 0.5).astype(np.uint8)

        dice_pred_vs_gt = calculate_dice_score(gt_mask_np, predseg)
        dice_results["prediction_vs_ground_truth"] = dice_pred_vs_gt
        logger.info(f"DICE SCORE (Prediction vs. Ground Truth): {dice_pred_vs_gt:.2f}%")

        num_superpixels_for_slic = 300
        segments_for_viz = slic(orig_image_scaled_for_display, n_segments=num_superpixels_for_slic,
                               compactness=0.1, sigma=1.0, start_label=1,
                               enforce_connectivity=True, channel_axis=None, convert2lab=True)

        in_context_superpixel_map = get_refined_superpixel_map(gt_mask_np, segments_for_viz, foreground_class_id=1)
        out_of_context_superpixel_map = np.copy(segments_for_viz)
        in_context_labels = np.unique(in_context_superpixel_map)
        if 0 in in_context_labels:
            in_context_labels = in_context_labels[in_context_labels != 0]

        is_in_context_pixel = np.isin(segments_for_viz, in_context_labels)
        out_of_context_superpixel_map[is_in_context_pixel] = 0

        roi_mask_for_foreground = (in_context_superpixel_map > 0)
        dice_pred_vs_pred_in_context = calculate_masked_dice_score(predseg, predseg, roi_mask_for_foreground)
        dice_results["pred_vs_pred_on_in_context_superpixels"] = dice_pred_vs_pred_in_context
        logger.info(f"DICE SCORE (Pred vs. Pred in In-Context Area): {dice_pred_vs_pred_in_context:.2f}% (Sanity Check)")

        num_in_context_superpixels = len(np.unique(in_context_superpixel_map)) - (1 if 0 in np.unique(in_context_superpixel_map) else 0)
        num_out_of_context_superpixels = len(np.unique(out_of_context_superpixel_map)) - (1 if 0 in np.unique(out_of_context_superpixel_map) else 0)

        logger.info(f"Superpixel maps generated. In-Context: {num_in_context_superpixels}, Out-of-Context: {num_out_of_context_superpixels}")

        try:
            # MODIFIED: Downsample the input tensor to 256x256 specifically for EigenCAM to prevent OOM errors
            input_tensor_for_eigencam = F.interpolate(input_tensor_for_model, size=(256, 256), mode='bilinear', align_corners=False)
            
            # MODIFIED: Calculate EigenCAM using the downsampled tensor, but instruct it to upsample the final map back to the original image's shape for visualization.
            original_eigencam_heatmap = calculate_eigencam(diffusion.netG, 'Conv', input_tensor_for_eigencam, orig_image_np.shape)
        except Exception as e:
            logger.error(f"Error calculating original EigenCAM: {e}. Using blank heatmap.")
            original_eigencam_heatmap = np.zeros_like(orig_image_np, dtype=np.float32)

        visualize_segmentation_comparison(orig_image_scaled_for_display, gt_mask_np, predseg, segments_for_viz, in_context_superpixel_map, out_of_context_superpixel_map, original_eigencam_heatmap, save_path=os.path.join(current_image_output_dir, f"{dataInfo}_comprehensive.png"))

        noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper']

        # --- In-Context Perturbation Analysis ---
        logger.info("--- Starting In-Context Perturbation Analysis ---")
        num_sp_to_perturb_in_context = max(1, int(num_in_context_superpixels * args.perturb_ratio))
        selected_sp_ids_in_context = select_random_superpixels(in_context_superpixel_map, num_sp_to_perturb_in_context)
        if not selected_sp_ids_in_context:
            logger.warning("No in-context superpixels selected. Skipping in-context analysis.")
        else:
            roi_mask_in_context = np.isin(segments_for_viz, selected_sp_ids_in_context)
            logger.info(f"Selected {len(selected_sp_ids_in_context)} in-context superpixels for perturbation.")
            for noise_type in noise_types:
                dice_results["in_context_perturbation"][noise_type] = {}

                perturbed_image_np_scaled = perturb_superpixels(orig_image_scaled_for_display, segments_for_viz, selected_sp_ids_in_context, noise_type)

                perturbed_image_tensor_for_model = torch.from_numpy(perturbed_image_np_scaled[np.newaxis, np.newaxis, :, :]).float()
                perturbed_input_for_netG = (perturbed_image_tensor_for_model * 2) - 1
                perturbed_input_for_netG = perturbed_input_for_netG.to(device)
                val_data_perturbed = val_data.copy()
                val_data_perturbed['A'] = perturbed_input_for_netG
                diffusion.feed_data(val_data_perturbed)
                diffusion.test()
                perturbed_predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
                perturbed_predseg = (perturbed_predseg_raw > 0.5).astype(np.uint8)

                dice_global = calculate_dice_score(predseg, perturbed_predseg)
                logger.info(f"IN-CONTEXT DICE (Pred vs Perturbed Pred Global): {dice_global:.2f}%")
                dice_results["in_context_perturbation"][noise_type]['pred_vs_perturbed_pred_global'] = dice_global

                dice_local = calculate_masked_dice_score(predseg, perturbed_predseg, roi_mask_in_context)
                logger.info(f"IN-CONTEXT DICE (Pred vs Perturbed Pred on Perturbed SPs): {dice_local:.2f}%")
                dice_results["in_context_perturbation"][noise_type]['pred_vs_perturbed_pred_on_perturbed_in_context_superpixels'] = dice_local

                try:
                    # MODIFIED: Downsample perturbed tensor for EigenCAM
                    perturbed_input_for_eigencam = F.interpolate(perturbed_input_for_netG, size=(256, 256), mode='bilinear', align_corners=False)
                    perturbed_eigencam_heatmap = calculate_eigencam(diffusion.netG, 'Conv', perturbed_input_for_eigencam, orig_image_np.shape)
                except Exception as e:
                    logger.error(f"Error on in-context EigenCAM ({noise_type}): {e}")
                    perturbed_eigencam_heatmap = np.zeros_like(orig_image_np, dtype=np.float32)

                visualize_perturbation_analysis(orig_image_scaled_for_display, perturbed_image_np_scaled, perturbed_predseg, perturbed_eigencam_heatmap, predseg, original_eigencam_heatmap, noise_type, save_path=os.path.join(current_image_output_dir, f"{dataInfo}_in_context_perturb_{noise_type}.png"))

        # --- Out-of-Context Perturbation Analysis ---
        if args.num_out_of_context_perturb_sp > 0:
            logger.info("--- Starting Out-of-Context Perturbation Analysis ---")
            selected_sp_ids_out_of_context = select_random_superpixels(out_of_context_superpixel_map, args.num_out_of_context_perturb_sp)

            if not selected_sp_ids_out_of_context:
                logger.warning("No out-of-context superpixels selected. Skipping out-of-context analysis.")
            else:
                roi_mask_for_foreground = (in_context_superpixel_map > 0)
                logger.info(f"Selected {len(selected_sp_ids_out_of_context)} out-of-context superpixels for perturbation.")
                for noise_type in noise_types:
                    dice_results["out_of_context_perturbation"][noise_type] = {}

                    perturbed_image_np_scaled = perturb_superpixels(orig_image_scaled_for_display, segments_for_viz, selected_sp_ids_out_of_context, noise_type)

                    perturbed_image_tensor_for_model = torch.from_numpy(perturbed_image_np_scaled[np.newaxis, np.newaxis, :, :]).float()
                    perturbed_input_for_netG = (perturbed_image_tensor_for_model * 2) - 1
                    perturbed_input_for_netG = perturbed_input_for_netG.to(device)

                    val_data_perturbed = val_data.copy()
                    val_data_perturbed['A'] = perturbed_input_for_netG
                    diffusion.feed_data(val_data_perturbed)
                    diffusion.test()
                    perturbed_predseg_raw = diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
                    perturbed_predseg = (perturbed_predseg_raw > 0.5).astype(np.uint8)

                    dice_global = calculate_dice_score(predseg, perturbed_predseg)
                    logger.info(f"OUT-OF-CONTEXT DICE (Pred vs Perturbed Pred Global): {dice_global:.2f}%")
                    dice_results["out_of_context_perturbation"][noise_type]['pred_vs_perturbed_pred_global'] = dice_global

                    dice_foreground = calculate_masked_dice_score(predseg, perturbed_predseg, roi_mask_for_foreground)
                    logger.info(f"OUT-OF-CONTEXT DICE (Pred vs Perturbed Pred on In-Context SPs): {dice_foreground:.2f}%")
                    dice_results["out_of_context_perturbation"][noise_type]['pred_vs_perturbed_pred_on_in_context_superpixels'] = dice_foreground

                    try:
                        # MODIFIED: Downsample perturbed tensor for EigenCAM
                        perturbed_input_for_eigencam = F.interpolate(perturbed_input_for_netG, size=(256, 256), mode='bilinear', align_corners=False)
                        perturbed_eigencam_heatmap = calculate_eigencam(diffusion.netG, 'Conv', perturbed_input_for_eigencam, orig_image_np.shape)
                    except Exception as e:
                        logger.error(f"Error on out-of-context EigenCAM ({noise_type}): {e}")
                        perturbed_eigencam_heatmap = np.zeros_like(orig_image_np, dtype=np.float32)

                    visualize_perturbation_analysis(orig_image_scaled_for_display, perturbed_image_np_scaled, perturbed_predseg, perturbed_eigencam_heatmap, predseg, original_eigencam_heatmap, noise_type, save_path=os.path.join(current_image_output_dir, f"{dataInfo}_out_of_context_perturb_{noise_type}.png"))

        json_save_path = os.path.join(current_image_output_dir, f"{dataInfo}_dice_scores.json")
        with open(json_save_path, 'w') as f:
            json.dump(dice_results, f, indent=4)
        logger.info(f"Dice scores saved to {json_save_path}")

        logger.info(f"Image {dataInfo}: Inference Time = {inference_time:.4f} seconds.")

    logger.info(f"\n--- Pipeline Summary ---")
    logger.info(f"Processed {processed_samples} samples.")
    logger.info(f"Total Inference Time: {total_inference_time:.4f} seconds.")
    if processed_samples > 0:
        logger.info(f"Average Inference Time per Sample: {total_inference_time / processed_samples:.4f} seconds.")
    logger.info(f"Visualizations saved to individual image folders inside: {args.output_dir}")

if __name__ == "__main__":
    main()

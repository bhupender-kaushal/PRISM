import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import argparse
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, quickshift
from skimage.color import gray2rgb
from scipy.ndimage import zoom, gaussian_filter
from torchvision import transforms
from math import exp
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd
import seaborn as sns
from skimage.metrics import structural_similarity


# Directly import the model from your project structure
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer

# =================================================================================
# 1. DATASET LOADER (from rel_run.py)
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
        data = h5py.File(filepath)
        image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label, 'case_name': vol_name}
        return sample

# =================================================================================
# 2. HELPER & ANALYSIS FUNCTIONS (Structure from mouse.py, Content from rel_run.py)
# =================================================================================

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

def get_refined_superpixel_map(gt_mask, superpixels_segments, foreground_class_id=1):
    """Creates an in-context superpixel map by clipping superpixels to the GT foreground."""
    gt_foreground = (gt_mask == foreground_class_id).astype(bool)
    in_context_segments = np.copy(superpixels_segments)
    in_context_segments[~gt_foreground] = 0
    return in_context_segments

def select_superpixels_by_gt_mask(superpixels_segments, gt_mask):
    """Selects all superpixels that have any overlap with the ground truth mask."""
    gt_mask_bool = gt_mask > 0
    overlapping_labels = np.unique(superpixels_segments[gt_mask_bool])
    return [label for label in overlapping_labels if label != 0]

def get_target_layer(model, layer_name):
    """Helper to find a specific layer by name in the model."""
    try:
        return model.get_submodule(layer_name)
    except AttributeError:
        for name, module in model.named_modules():
            if layer_name in name:
                return module
        raise AttributeError(f"Could not find target layer '{layer_name}'.")

def calculate_eigencam(model, input_tensor, target_layer_names, output_size):
    """Calculates EigenCAM, averaging over multiple target layers."""
    if not isinstance(target_layer_names, list):
        target_layer_names = [target_layer_names]
    all_saliency_maps = []
    for layer_name in target_layer_names:
        feature_maps = []
        try:
            target_layer = get_target_layer(model, layer_name)
        except AttributeError as e:
            logging.warning(f"{e}. Skipping this layer for EigenCAM.")
            continue
        def hook_fn(module, input, output):
            if output.dim() == 3 and output.shape[0] == input_tensor.shape[0]: # Shape (B, N, C)
                B, N, C = output.shape
                H = W = int(np.sqrt(N))
                feature_maps.append(output.permute(0, 2, 1).view(B, C, H, W))
            else: # Standard Conv output
                feature_maps.append(output)
        hook = target_layer.register_forward_hook(hook_fn)
        _ = model(input_tensor)
        hook.remove()
        if not feature_maps:
            logging.warning(f"EigenCAM hook did not capture feature maps for layer {layer_name}. Skipping.")
            continue
        features = feature_maps[0].squeeze(0).to(torch.float32)
        features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().detach().numpy()
        pca = PCA(n_components=1).fit(features_flat)
        saliency_map = (features.cpu().detach().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
        saliency_map = np.maximum(saliency_map, 0)
        saliency_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float()
        resized_saliency = F.interpolate(saliency_tensor, size=output_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        all_saliency_maps.append(resized_saliency)
    if not all_saliency_maps:
        logging.error("Could not generate any saliency maps. Returning a zero map.")
        return np.zeros(output_size, dtype=np.float32)
    avg_saliency_map = np.mean(all_saliency_maps, axis=0)
    s_min, s_max = avg_saliency_map.min(), avg_saliency_map.max()
    return (avg_saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(avg_saliency_map)

def run_inference_on_slice(model, image_slice, patch_size, device='cuda'):
    """Performs inference on a single grayscale slice."""
    h, w = image_slice.shape
    slice_rescaled = zoom(image_slice, (patch_size / h, patch_size / w), order=3) if (h != patch_size or w != patch_size) else image_slice
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = x_transforms(slice_rescaled).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    if h != patch_size or w != patch_size:
        prediction = zoom(prediction, (h / patch_size, w / patch_size), order=0)
    return prediction, input_tensor

def perturb_image_slice(image_01, segments, sp_to_perturb, noise_type, noise_level):
    """Applies noise directly to a single-channel grayscale image."""
    perturbed_image = image_01.copy()
    for sp_id in sp_to_perturb:
        sp_mask = (segments == sp_id)
        region = perturbed_image[sp_mask]
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.1 * noise_level, region.shape)
            perturbed_image[sp_mask] += noise
        elif noise_type == 's&p':
            num_pixels = region.size; num_noise = int(min(0.5, 0.05 * noise_level) * num_pixels)
            if num_noise > 0:
                indices = np.random.choice(num_pixels, num_noise, replace=False)
                region[indices] = np.random.choice([0.0, 1.0], num_noise)
                perturbed_image[sp_mask] = region
        elif noise_type == 'poisson':
            sigma = max(0.05, 3.2 - noise_level)
            noisy_region = np.random.poisson(region * 25 * sigma) / (25 * sigma)
            perturbed_image[sp_mask] = noisy_region
        elif noise_type == 'speckle':
            noise = np.random.normal(0, 0.1 * noise_level, region.shape)
            perturbed_image[sp_mask] *= (1 + noise)
        elif noise_type == 'gaussian_blur':
            blurred_image = gaussian_filter(perturbed_image, sigma=0.5 * noise_level)
            perturbed_image[sp_mask] = blurred_image[sp_mask]
        elif noise_type == 'rician':
            n1 = np.random.normal(0, 0.1 * noise_level, region.shape); n2 = np.random.normal(0, 0.1 * noise_level, region.shape)
            noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
            perturbed_image[sp_mask] = noisy_region
    return np.clip(perturbed_image, 0.0, 1.0)

def calculate_iou_in_roi(pred_mask, gt_mask, roi_mask):
    """
    Calculates IoU between prediction and ground truth, but only within a Region of Interest (ROI).
    Assumes pred_mask and gt_mask are boolean.
    """
    roi_mask = roi_mask.astype(bool)
    pred_roi = pred_mask[roi_mask]
    gt_roi = gt_mask[roi_mask]
    intersection = np.logical_and(pred_roi, gt_roi).sum()
    union = np.logical_or(pred_roi, gt_roi).sum()
    return (intersection / union)*100 if union > 0 else 0.0

def calculate_rbo(list1, list2, p=0.9):
    """Calculates Rank-Biased Overlap (RBO)."""
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

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None, prompt_points=None):
    """Visualizes the original image, GT, prediction, and various superpixel maps, with prompt points overlayed."""
    display_image_rgb = orig_image
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb, cmap='gray')
    if prompt_points is not None:
        for point in prompt_points:
            y, x = point
            axes[0].scatter(x, y, c='red', s=200, marker='*', edgecolors='yellow', linewidths=1)
        axes[0].legend(loc='upper right')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    axes[3].imshow(mark_boundaries(display_image_rgb, all_sp_map))
    axes[3].set_title("All Superpixels")
    axes[3].axis('off')

    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map))
    axes[4].set_title("In-Context SPs")
    axes[4].axis('off')

    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map))
    axes[5].set_title("Out-of-Context SPs")
    axes[5].axis('off')

    im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1.0)
    axes[6].set_title("EigenCAM")
    axes[6].axis('off')
    plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# =================================================================================
# 3. CORE ANALYSIS PIPELINE (Structure from mouse.py)
# =================================================================================
def run_ranking_analysis(model, device, orig_image_01, segments, target_class_gt, initial_pred_mask, sp_to_perturb, sp_to_measure, noise_types, patch_size, target_class_index, current_image_output_dir, analysis_type, args):
    """
    Runs the full ranking analysis, behaving identically to the mouse.py implementation.
    """
    logger = logging.getLogger('base')
    logger.info(f"--- Starting Ranking Analysis: {analysis_type.upper()} ---")
    logger.info(f"Perturbing {len(sp_to_perturb)} SPs, measuring stability effect on {len(sp_to_measure)} SPs.")

    analysis_results = {}
    initial_pred_mask_bool = (initial_pred_mask == target_class_index)

    # Initial ranking is based on IoU between the initial prediction and the Ground Truth.
    initial_sp_iou_data_vs_gt = [
        {"sp_id": int(sp_id), "iou": round(calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id), 2)}
        for sp_id in sp_to_measure
    ]
    sorted_initial_sp_data = sorted(initial_sp_iou_data_vs_gt, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1):
        sp_data['rank'] = rank
    initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
    
    # ### MODIFIED: Create a lookup map for the initial IoU vs. GT for efficient access later
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
        os.makedirs(noise_specific_dir, exist_ok=True)

        level_0_entry = {"noise_level": 0.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
        level_data_list = [level_0_entry]
        
        rbo_scores_list = []
        sp_iou_history = {sp_id: [] for sp_id in sp_to_measure}
        sp_rank_history = {sp_id: [] for sp_id in sp_to_measure}
        
        for noise_level in np.arange(0.1, 3, 0.05):
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, sp_to_perturb, noise_type, noise_level)

            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            if ssim_score < 0.85:
                logger.warning(f"SSIM dropped below 0.85 at noise level {noise_level:.2f} for {noise_type}. Stopping further perturbations.")
                break

            img_save_path = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.3f}.png")
            Image.fromarray((perturbed_img_gray * 255).astype(np.uint8)).save(img_save_path)
            
            perturbed_pred_mask_full, _ = run_inference_on_slice(model, perturbed_img_gray, patch_size, device)
            perturbed_pred_mask_bool = (perturbed_pred_mask_full == target_class_index)

            # Core Metric: IoU between INITIAL prediction and PERTURBED prediction (measures stability).
            current_level_sp_data = [
                {"sp_id": int(sp_id), "iou": round(calculate_iou_in_roi(initial_pred_mask_bool, perturbed_pred_mask_bool, segments == sp_id), 2)}
                for sp_id in sp_to_measure
            ]
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            for sp_id in sp_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))

            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            current_rbo = calculate_rbo(initial_rank_list, current_rank_list, p=0.9)
            rbo_scores_list.append(round(current_rbo, 4))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1):
                sp_data['rank'] = rank
            level_data_list.append({
                "noise_level": round(noise_level, 2),
                "image_path": os.path.relpath(img_save_path, args.output_dir),
                "superpixels": sorted_superpixels
            })
        
        # ### MODIFIED: Replaced original reliability calculation with new multi-step logic and OverflowError handling
        reliability_results = []
        for sp_id in sp_to_measure:
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
                    # If exponent is too large, exp(exponent) is infinity, so 1/exp(...) is 0.
                    # The factor becomes 1.0, preventing a crash.
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
            
        analysis_results[noise_type] = {
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "rbo_at_each_level": rbo_scores_list,
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['final_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results

# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================

def find_samples_with_class(dataset, target_class, area_threshold_percentage=0.9):
    """
    For each volume, finds the max area of the target class, then collects all slices
    from that volume with an area >= (max_area * threshold).
    Returns a list of all such slices from all volumes, sorted by area in descending order.
    """
    candidate_slices = []
    logging.info(f"Scanning each volume to find slices with area >= {area_threshold_percentage*100}% of max area for class {target_class}...")

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
                if s['area'] >= area_threshold:
                    candidate_slices.append(s)
            
    selected_slices = sorted(candidate_slices, key=lambda x: x['area'], reverse=True)
    return selected_slices

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "analysis.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading model...")
    net = MaxViT_deformableLKAFormer(num_classes=args.num_classes).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    logger.info("Loading dataset...")
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    candidate_samples = find_samples_with_class(db_test, args.target_class_index)
    
    if not candidate_samples:
        logger.error(f"No samples found with target class {args.target_class_index}. Exiting.")
        return
    
    random.shuffle(candidate_samples)
    
    logger.info(f"Found {len(candidate_samples)} high-quality candidate slices. Will now process until {args.num_samples} valid samples (IoU > 70%) are analyzed.")

    processed_count = 0
    all_images_summary = {
        'reliability': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}},
        'rbo': {'in_context': {nt: [] for nt in args.noise_types}, 'out_of_context': {nt: [] for nt in args.noise_types}}
    }

    for sample_info in tqdm(candidate_samples, desc="Finding and Processing Valid Images"):
        if processed_count >= args.num_samples:
            logger.info(f"Successfully processed {processed_count} samples. Target of {args.num_samples} reached.")
            break

        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        volume_data = db_test[vol_idx]
        image_slice, label_slice = volume_data['image'][slice_idx, :, :], volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"
        
        image_slice_01 = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        initial_pred_mask, input_tensor = run_inference_on_slice(net, image_slice_01, args.img_size, device)
        target_class_gt = (label_slice == args.target_class_index)
        initial_pred_mask_bool = (initial_pred_mask == args.target_class_index)
        full_image_roi = np.ones_like(target_class_gt, dtype=bool)
        
        if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, full_image_roi) < 70:
            logger.warning(f"Initial IoU for {case_name} is below 70%. Skipping and finding next valid slice.")
            continue

        processed_count += 1
        logger.info(f"Sample {case_name} is valid (IoU > 70%). Processing... ({processed_count}/{args.num_samples})")
        current_image_output_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(current_image_output_dir, exist_ok=True)

        segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)
        # if image_slice_01.ndim == 2:  # Single-channel image
        #     img_quick = np.stack([image_slice_01] * 3, axis=-1)  # Create a 3-channel version for Quickshift
        # else:
        #     img_quick = image_slice_01
        # segments = quickshift(img_quick, kernel_size=5, max_dist=10, ratio=0.7)
        
        ### MODIFIED: Added filtering logic for superpixels based on IoU > 50
        in_context_sp_all = select_superpixels_by_gt_mask(segments, target_class_gt)
        qualified_in_context_sp = []
        if in_context_sp_all:
            logger.info(f"Found {len(in_context_sp_all)} superpixels overlapping with GT. Filtering for IoU > 50...")
            for sp_id in in_context_sp_all:
                iou = calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id)
                if iou > 50:
                    qualified_in_context_sp.append(sp_id)
            logger.info(f"{len(qualified_in_context_sp)} superpixels qualified with IoU > 50.")
        
        if not qualified_in_context_sp:
            logger.warning(f"No superpixels for {case_name} met the >50 IoU threshold. Skipping analysis.")
            continue
        
        all_sp_ids = np.unique(segments)
        all_out_of_context_ids = [sp_id for sp_id in all_sp_ids if sp_id not in in_context_sp_all and sp_id != 0]
        out_of_context_sp_to_perturb = random.sample(all_out_of_context_ids, min(args.num_out_of_context_samples, len(all_out_of_context_ids)))
        
        eigencam_heatmap = calculate_eigencam(net, input_tensor, args.target_layer_names, (args.img_size, args.img_size))
        in_context_map = get_refined_superpixel_map(target_class_gt, segments)
        out_of_context_map = np.copy(segments)
        for sp_id in in_context_sp_all: out_of_context_map[out_of_context_map == sp_id] = 0
        
        visualize_segmentation_comparison(gray2rgb(image_slice_01), target_class_gt, initial_pred_mask == args.target_class_index, segments, in_context_map, out_of_context_map, eigencam_heatmap, os.path.join(current_image_output_dir, "overview.png"))

        final_analysis = {
            "image_info": {
                "name": case_name,
                "initial_dice_vs_gt": round(calculate_dice_score(initial_pred_mask_bool, target_class_gt), 2)
            }
        }
        
        ### MODIFIED: Updated run_ranking_analysis calls to use the qualified list for perturbation and measurement
        if qualified_in_context_sp:
            logger.info("Analyzing IN-CONTEXT perturbations...")
            final_analysis['in_context_analysis'] = run_ranking_analysis(net, device, image_slice_01, segments, target_class_gt, initial_pred_mask, qualified_in_context_sp, qualified_in_context_sp, args.noise_types, args.img_size, args.target_class_index, current_image_output_dir, 'in_context', args)
        if out_of_context_sp_to_perturb:
            logger.info("Analyzing OUT-OF-CONTEXT perturbations...")
            final_analysis['out_of_context_analysis'] = run_ranking_analysis(net, device, image_slice_01, segments, target_class_gt, initial_pred_mask, out_of_context_sp_to_perturb, qualified_in_context_sp, args.noise_types, args.img_size, args.target_class_index, current_image_output_dir, 'out_of_context', args)

        json_save_path = os.path.join(current_image_output_dir, f"{case_name}_analysis.json")
        with open(json_save_path, 'w') as f:
            json.dump(final_analysis, f, indent=4, cls=NpEncoder)
        logger.info(f"Full analysis saved to {json_save_path}")
        
        ### MODIFIED: Changed reliability_score to final_reliability_score to prevent KeyError
        for at in ['in_context', 'out_of_context']:
            if f'{at}_analysis' in final_analysis:
                for nt, res in final_analysis[f'{at}_analysis'].items():
                    all_images_summary['rbo'][at][nt].append(res['average_rank_biased_overlap'])
                    scores = [r['final_reliability_score'] for r in res.get('reliability_analysis', [])]
                    if scores: all_images_summary['reliability'][at][nt].append(np.mean(scores))

    if processed_count < args.num_samples:
        logger.warning(f"Could only find and process {processed_count} valid samples, which is less than the target of {args.num_samples}.")

    logger.info("--- Generating Final Summary Plots ---")
    rbo_plot_data = [{'Analysis Type': at.replace('_', ' ').title(), 'Noise Type': nt, 'RBO Score': np.clip(s,0,1)} for at, nd in all_images_summary['rbo'].items() for nt, sc in nd.items() for s in sc]
    if rbo_plot_data:
        df_rbo = pd.DataFrame(rbo_plot_data)
        g = sns.catplot(data=df_rbo, x='Noise Type', y='RBO Score', col='Analysis Type', kind='violin', palette='viridis', height=6, aspect=1.2)
        g.fig.suptitle('RBO Score Distribution by Noise Type', y=1.03, fontsize=16); g.set_axis_labels("Noise Type", "RBO Score"); g.set_xticklabels(rotation=45, ha='right'); g.set_titles("{col_name} Analysis")
        plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(args.output_dir, "RBO_Distributions_ViolinPlot.png"), dpi=300); plt.close()

    for analysis_type, noise_data in all_images_summary['reliability'].items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7)); bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='mako')
            plt.title(f"Final Reliability Score ({analysis_type.replace('_', ' ').title()})", fontsize=16)
            for bar in bars.patches: bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, f"{analysis_type}_final_reliability_summary.png"), dpi=300); plt.close()

    logger.info(f"--- Pipeline Finished. Results are in {args.output_dir} ---")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Reliability Analysis for Segmentation Models")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--model_path", type=str, default="./MaxViT_deform_LKA.pth")
    # parser.add_argument("--output_dir", type=str, default="./expresults_quickshift")
    parser.add_argument("--output_dir", type=str, default="./expresults")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int,default=8, required=False, help="The specific class index to analyze.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of random images to select and process.")
    parser.add_argument("--num_superpixels", type=int, default=200, help="Number of superpixels to generate.")
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 's&p', 'poisson', 'speckle', 'gaussian_blur', 'rician'], help="List of noise types to apply.")
    parser.add_argument('--target_layer_names', nargs='+', default=['last_layer'], help='List of model layers for EigenCAM.')
    parser.add_argument("--num_out_of_context_samples", type=int, default=40, help="Number of random out-of-context superpixels to perturb.")

    args = parser.parse_args()
    main(args)


    # {"1": "spleen", "2": "right kidney", "3": "left kidney", "4": "gallbladder", "5": "esophagus", "6": "liver", "7": "stomach", "8": "aorta"}' }
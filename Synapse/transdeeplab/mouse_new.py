import importlib
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import argparse
import json
import logging
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid
from skimage.color import gray2rgb
from scipy.ndimage import zoom, gaussian_filter
from torchvision import transforms
from math import exp
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd
import seaborn as sns
from skimage.metrics import structural_similarity

from model.swin_deeplab import SwinDeepLab

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
        data = h5py.File(filepath)
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
    elif noise_type == 's&p' or noise_type == 'salt_pepper':
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

def run_inference_on_slice(model, image_slice, patch_size, device='cuda'):
    h, w = image_slice.shape
    if h != patch_size or w != patch_size:
        slice_rescaled = zoom(image_slice, (patch_size / h, patch_size / w), order=3)
    else:
        slice_rescaled = image_slice
    
    input_tensor = torch.from_numpy(slice_rescaled).unsqueeze(0).unsqueeze(0).float().to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction_rescaled = out.cpu().detach().numpy()

    if h != patch_size or w != patch_size:
        prediction = zoom(prediction_rescaled, (h / patch_size, w / patch_size), order=0)
    else:
        prediction = prediction_rescaled
        
    return prediction, input_tensor

def get_target_layer(model, layer_name):
    try:
        return model.get_submodule(layer_name)
    except AttributeError:
        for name, module in model.named_modules():
            if layer_name in name: return module
        raise AttributeError(f"Could not find target layer '{layer_name}'.")

def calculate_eigencam(model, input_tensor, target_layer_names, output_size):
    if not isinstance(target_layer_names, list): target_layer_names = [target_layer_names]
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

def visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    display_image_rgb = gray2rgb(orig_image_01)
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
    analysis_type, model, device, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, superpixels_to_measure, initial_pred_mask_bool,
    noise_types, patch_size, target_class_index, current_image_output_dir, 
    noise_step=0.1, save_visuals=False
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
            
            perturbed_pred_mask_full, _ = run_inference_on_slice(model, perturbed_img_gray, patch_size, device)
            perturbed_pred_mask_bool = (perturbed_pred_mask_full == target_class_index)

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
                img_rgb = gray2rgb(perturbed_img_gray)
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
# 3. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Loading SwinDeepLab model...")
    model_config = importlib.import_module(f'model.configs.{args.config_file}')
    args.img_size = model_config.EncoderConfig.img_size 
    logger.info(f"Image size set to {args.img_size} from model config.")
    
    net = SwinDeepLab(
        model_config.EncoderConfig, 
        model_config.ASPPConfig, 
        model_config.DecoderConfig
    ).to(device)
    net.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    logger.info("Loading Synapse dataset...")
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    candidate_samples = find_samples_with_class(db_test, args.target_class_index)
    if not candidate_samples: sys.exit()
    random.shuffle(candidate_samples)

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    for sample_info in tqdm(candidate_samples, desc="Qualifying Slices"):
        if len(qualified_images) >= args.num_images: break

        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        volume_data = db_test[vol_idx]
        image_slice, label_slice = volume_data['image'][slice_idx, :, :], volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"
        
        image_slice_01 = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        initial_pred_mask, input_tensor = run_inference_on_slice(net, image_slice_01, args.img_size, device)
        
        target_class_gt = (label_slice == args.target_class_index)
        initial_pred_mask_bool = (initial_pred_mask == args.target_class_index)
        full_image_roi = np.ones_like(target_class_gt, dtype=bool)
        
        if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, full_image_roi) < 70: continue

        # segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        ## for quickshift.
        # segments = quickshift(color.gray2rgb(image_slice_01), kernel_size=5, max_dist=10, ratio=0.5)
        # for Watershed. 

        # gradient = sobel(image_slice_01)
        # grid = regular_grid(image_slice_01.shape[:2], n_points=500)
        # markers = np.zeros(image_slice_01.shape[:2], dtype=int)
        
        # # Safely reshape the arange array to match the N-dimensional grid space
        # target_shape = markers[grid].shape
        # markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
        
        # segments = watershed(gradient, markers, compactness=0.001)

        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=1.0, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_in_roi(initial_pred_mask_bool, target_class_gt, segments == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'case_name': case_name, 'image_slice_01': image_slice_01, 'input_tensor': input_tensor,
            'target_class_gt': target_class_gt, 'segments': segments, 'initial_pred_mask_bool': initial_pred_mask_bool,
            'initial_pred_mask': initial_pred_mask, 'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool, target_class_gt)
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
            image_slice_01 = q_data['image_slice_01']
            input_tensor = q_data['input_tensor']
            target_class_gt = q_data['target_class_gt']
            segments = q_data['segments']
            initial_pred_mask_bool = q_data['initial_pred_mask_bool']
            initial_pred_mask = q_data['initial_pred_mask']
            qualified_ic_sp = q_data['qualified_ic_sp']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, case_name)
            os.makedirs(current_image_output_dir, exist_ok=True)
            final_analysis = {"image_info": {"name": case_name, "initial_dice_vs_gt": round(q_data['initial_dice_vs_gt'], 2)}}

            res_ic = run_ranking_analysis(
                'in_context', net, device, image_slice_01, segments, target_class_gt, 
                qualified_ic_sp, qualified_ic_sp, initial_pred_mask_bool,
                args.noise_types, args.img_size, args.target_class_index, current_image_output_dir, 
                noise_step=args.noise_step, save_visuals=(run_idx == 0)
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

            # --- DYNAMIC 70% OC SELECTION FIX ---
            num_oc_to_perturb = int(0.70 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments)
                for sp_id in qualified_ic_sp: in_context_map[segments == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments)
                for sp_id in oc_sps_all: out_of_context_map[segments == sp_id] = sp_id
                eigencam_heatmap = calculate_eigencam(net, input_tensor, args.target_layer_names, (args.img_size, args.img_size))
                visualize_segmentation_comparison(image_slice_01, target_class_gt, initial_pred_mask == args.target_class_index, segments, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, "overview.png"))

            res_oc = run_ranking_analysis(
                'out_of_context', net, device, image_slice_01, segments, target_class_gt, 
                out_of_context_sp_to_perturb, qualified_ic_sp, initial_pred_mask_bool,
                args.noise_types, args.img_size, args.target_class_index, current_image_output_dir, 
                noise_step=args.noise_step, save_visuals=(run_idx == 0)
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
    parser = argparse.ArgumentParser(description="High-Speed Two-Phase Model Reliability Inference (SwinDeepLab)")
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument('--ckpt_path', type=str, default='/bhupender/2025/Mouse_other/Synapse/transdeeplab/swin_224_7_3level.pth')
    parser.add_argument('--config_file', type=str, default='swin_224_7_3level')
    parser.add_argument("--output_dir", type=str, default="./new_expresults_ws")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int, default=8)
    
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_dataset_runs", type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument("--noise_step", type=float, default=0.05, help="Step size for noise loop.")
    
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--target_layer_names', nargs='+', default=['final_up'])
    
    args = parser.parse_args()
    main(args)
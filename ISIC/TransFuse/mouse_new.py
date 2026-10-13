import os
import json
import argparse
import logging
import random
import time
import itertools
import math
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

from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid

from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr
from skimage.color import gray2rgb, rgb2gray
import torch.utils.data as data

from lib.TransFuse import TransFuse_S 

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

def calculate_iou_score(mask1, mask2, roi_mask, label=1):
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

def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0
    
    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    
    if not np.any(region_mask):
        return perturbed_image, noise_char_value
        
    region_mask_3c = np.stack([region_mask]*3, axis=-1)

    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] += noise[region_mask_3c]
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, 0.05 * noise_level)
        num_pixels_in_region = np.sum(region_mask)
        num_noise = int(noise_char_value * num_pixels_in_region)
        if num_noise > 0:
            for c in range(3):
                rows, cols = np.where(region_mask)
                indices = np.random.choice(len(rows), num_noise, replace=False)
                perturbed_image[rows[indices], cols[indices], c] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        noise_char_value = max(0.05, 3.2 - noise_level)
        region = perturbed_image[region_mask_3c]
        noisy_region = np.random.poisson(region * 25 * noise_char_value) / (25 * noise_char_value)
        perturbed_image[region_mask_3c] = noisy_region
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] *= (1 + noise[region_mask_3c])
    elif noise_type == 'gaussian_blur':
        noise_char_value = 0.5 * noise_level
        blurred_image = gaussian_filter(original_image_scaled, sigma=(noise_char_value, noise_char_value, 0))
        perturbed_image[region_mask_3c] = blurred_image[region_mask_3c]
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        region = perturbed_image[region_mask_3c]
        n1 = np.random.normal(0, noise_char_value, region.shape)
        n2 = np.random.normal(0, noise_char_value, region.shape)
        noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
        perturbed_image[region_mask_3c] = noisy_region

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

def get_target_layer(model, layer_name):
    try:
        return model.get_submodule(layer_name)
    except AttributeError:
        for name, module in model.named_modules():
            if layer_name in name: return module
        raise AttributeError(f"Could not find target layer '{layer_name}'.")

def calculate_eigencam(model, target_layer_name, image_tensor, image_size):
    feature_maps = []
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

def run_ranking_analysis(
    analysis_type, args, device, model,
    orig_image_np_01, pred_mask, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, current_image_output_dir,
    ssim_data_collector, save_visuals=False
):
    logger = logging.getLogger('base')
    analysis_results = {}
    
    lambda_param = 1.0  
    beta_param = 1.0
    max_id = segments_slic.max() + 1 

    # --- BG-IoU SPILLOVER BASELINE CALCULATION ---
    bg_gt = (gt_mask == 0)
    bg_pred_init = (pred_mask == 0)
    
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0
    
    # --- VECTORIZED FIX: Initial IoU Calculation ---
    intersect_mask_init = (pred_mask == 1) & (gt_mask > 0)
    union_mask_init = (pred_mask == 1) | (gt_mask > 0)
    
    intersect_counts_init = np.bincount(segments_slic[intersect_mask_init].ravel(), minlength=max_id)
    union_counts_init = np.bincount(segments_slic[union_mask_init].ravel(), minlength=max_id)
    
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

        level_0_entry = {"noise_level": 0.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
        level_data_list = [level_0_entry]
        
        rbo_scores_list, sp_iou_history, sp_rank_history = [], {sp_id: [] for sp_id in superpixels_to_measure}, {sp_id: [] for sp_id in superpixels_to_measure}
        
        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0

        noise_levels_arr = np.arange(args.noise_step, 3.1, args.noise_step)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False)

        for step_idx, noise_level in enumerate(pbar):
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})

            perturbed_image_np_scaled, noise_char_value = perturb_superpixels_with_level(orig_image_np_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_np_01, perturbed_image_np_scaled, data_range=1.0, channel_axis=-1)

            is_final_step = False
            if ssim_score < 0.70 :
                logger.info(f"\n[SSIM TRIGGER] SSIM ({ssim_score:.4f}) dropped below 0.70 at {noise_type.upper()} level {noise_level:.2f}. Halting.")
                is_final_step = True
            
            perturbed_tensor = torch.from_numpy(perturbed_image_np_scaled).permute(2, 0, 1).float()
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            perturbed_tensor_normalized = normalize(perturbed_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, res = model(perturbed_tensor_normalized)
                res = F.upsample(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                perturbed_pred_mask = np.where(res >= 0.5, 1, 0)

            # --- BG-IoU SPILLOVER INSTANT TRACKING ---
            bg_pred_pert = (perturbed_pred_mask == 0)
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
                Image.fromarray((perturbed_image_np_scaled * 255).astype(np.uint8)).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                Image.fromarray((perturbed_pred_mask * 255).astype(np.uint8)).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                
                img_rgb = gray2rgb(perturbed_image_np_scaled) if perturbed_image_np_scaled.ndim == 2 else perturbed_image_np_scaled
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask, color=(1, 0, 0), mode='thick')
                Image.fromarray((overlay_image * 255).astype(np.uint8)).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            if is_final_step:
                break

            # --- VECTORIZED FIX: Perturbed IoU Calculation ---
            intersect_mask_pert = (pred_mask == 1) & (perturbed_pred_mask == 1)
            union_mask_pert = (pred_mask == 1) | (perturbed_pred_mask == 1)
            
            intersect_counts_pert = np.bincount(segments_slic[intersect_mask_pert].ravel(), minlength=max_id)
            union_counts_pert = np.bincount(segments_slic[union_mask_pert].ravel(), minlength=max_id)
            
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

            initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            current_rbo = calculate_rbo(initial_rank_list, current_rank_list, p=0.9)
            rbo_scores_list.append(round(current_rbo, 2))
            
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

# --------------------------------------------------------------------------
# --- 3. Main Two-Phase Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="High-Speed Two-Phase Model Reliability Inference (TransFuse)")
    parser.add_argument('--pth_path', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/TransFuse-19_best.pth')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--target_layer_name', type=str, default='up_c')
    parser.add_argument('--output_dir', type=str, default='./new_expresults_ws')
    
    parser.add_argument('--num_images', type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument('--num_dataset_runs', type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument('--noise_step', type=float, default=0.05, help="Step size for noise loop.")
    
    parser.add_argument('--superpixel_size', type=int, default=500)
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
    logger.info(f"\n[PHASE 1] Scanning Dataset to Qualify {args.num_images} Images...")
    qualified_images = []
    
    for i, (image, gt, name_list) in enumerate(tqdm(test_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break
        
        name = name_list[0]
        image_tensor = image.to(device)
        _, _, h, w = image_tensor.shape

        gt_mask_np = np.asarray(gt, np.float32).squeeze(0).squeeze(0)
        pil_gt = Image.fromarray(gt_mask_np.astype(np.uint8))
        # Dynamically resize GT to match the loaded image dimensions
        gt_mask_np = np.array(pil_gt.resize((w, h), Image.NEAREST)).astype(np.float32)
        
        gt_mask_np /= (gt_mask_np.max() + 1e-8)

        with torch.no_grad():
            _, _, res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(res >= 0.5, 1, 0)
        
        full_image_roi = np.ones_like(gt_mask_np, dtype=bool)
        if calculate_iou_score(pred_mask, gt_mask_np, full_image_roi) < 70: continue

        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)

        # segments_slic = slic(orig_image_np_01, n_segments=args.superpixel_size, compactness=10, sigma=1)
        
        ## for quickshift
        # segments_slic = quickshift(orig_image_np_01, kernel_size=5, max_dist=10, ratio=0.5)
        
        ## for watershed
        # gradient = sobel(color.rgb2gray(image.squeeze(0).permute(1, 2, 0).cpu().numpy()))
        # grid = regular_grid(orig_image_np_01.shape[:2], n_points=500)
        # markers = np.zeros(orig_image_np_01.shape[:2], dtype=int)
        
        # # Safely reshape the arange array to match the N-dimensional grid space
        # target_shape = markers[grid].shape
        # markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
        
        # segments_slic = watershed(gradient, markers, compactness=0.001)

        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask_np, ic_threshold=1.0, oc_threshold=0.0)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_score(pred_mask, gt_mask_np, segments_slic == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue

        qualified_images.append({
            'name': name, 'image_tensor': image_tensor, 'orig_image_np_01': orig_image_np_01, 
            'gt_mask_np': gt_mask_np, 'segments_slic': segments_slic, 'pred_mask': pred_mask, 
            'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(gt_mask_np, pred_mask)
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
        ssim_data_collector = []

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0):
            name = q_data['name']
            orig_image_np_01 = q_data['orig_image_np_01']
            image_tensor = q_data['image_tensor']
            gt_mask_np = q_data['gt_mask_np']
            segments_slic = q_data['segments_slic']
            pred_mask = q_data['pred_mask']
            qualified_ic_sp = q_data['qualified_ic_sp']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, name.rsplit('.', 1)[0])
            os.makedirs(current_image_output_dir, exist_ok=True)
            final_analysis = {"image_info": {"name": name.rsplit('.', 1)[0], "initial_dice_vs_gt": round(q_data['initial_dice_vs_gt'], 2)}}

            res_ic = run_ranking_analysis(
                'in_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, 
                segments_slic, qualified_ic_sp, qualified_ic_sp, args.noise_types, noise_char_map, 
                current_image_output_dir, ssim_data_collector, save_visuals=(run_idx == 0)
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


            num_oc_to_perturb = int(0.50 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments_slic)
                for sp_id in qualified_ic_sp: in_context_map[segments_slic == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments_slic)
                for sp_id in oc_sps_all: out_of_context_map[segments_slic == sp_id] = sp_id
                
                eigencam_heatmap = calculate_eigencam(model_instance, args.target_layer_name, image_tensor, (orig_image_np_01.shape[0], orig_image_np_01.shape[1]))
                
                gt_mask_for_sp = (np.array(Image.fromarray((gt_mask_np * 255).astype(np.uint8)).resize(segments_slic.shape[::-1], Image.NEAREST)) > 127).astype(np.uint8) if gt_mask_np.shape != segments_slic.shape else gt_mask_np.astype(np.uint8)
                
                visualize_segmentation_comparison(orig_image_np_01, gt_mask_for_sp, pred_mask, segments_slic, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, name.rsplit('.', 1)[0] + "_overview.png"))

            res_oc = run_ranking_analysis(
                'out_of_context', args, device, model_instance, orig_image_np_01, pred_mask, gt_mask_np, 
                segments_slic, out_of_context_sp_to_perturb, qualified_ic_sp, args.noise_types, noise_char_map, 
                current_image_output_dir, ssim_data_collector, save_visuals=(run_idx == 0)
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

            with open(os.path.join(current_image_output_dir, f"{name.rsplit('.', 1)[0]}_analysis.json"), 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

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
import os
import argparse
import json
import logging
import random
import math
from math import exp
import sys
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter, distance_transform_edt, center_of_mass, label
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage.segmentation import slic, mark_boundaries
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Import the specific, up-to-date functions for SAM-1
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'segment_anything' library.")
    print("Please ensure you have cloned the official 'segment-anything' repository")
    print("and that its location is accessible to your Python environment.")
    sys.exit()

random_seed = 42
random.seed(random_seed)
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# --- 1. Scoring and Helper Functions
# --------------------------------------------------------------------------
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
        pil_img = Image.fromarray(mask1.astype(np.uint8) * 255)
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img) / 255
    if mask1.shape != roi_mask.shape:
        pil_img_roi = Image.fromarray(roi_mask.astype(np.uint8) * 255)
        resized_img_roi = pil_img_roi.resize(mask1.shape[::-1], Image.NEAREST)
        roi_mask = (np.array(resized_img_roi) / 255).astype(bool)
    mask1_roi, mask2_roi = mask1[roi_mask], mask2[roi_mask]
    m1_roi, m2_roi = (mask1_roi == label), (mask2_roi == label)
    intersect = np.sum(m1_roi & m2_roi)
    union = np.sum(m1_roi | m2_roi)
    if union == 0: return 100.0
    return (float(intersect) / float(union)) * 100

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

def perturb_superpixels_with_level(original_image_scaled_01, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled_01)
    noise_char_value = 0.0
    
    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask):
        return perturbed_image, noise_char_value

    H, W, C = perturbed_image.shape

    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] += noise[region_mask]
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, 0.05 * noise_level)
        rows, cols = np.where(region_mask)
        num_pixels = len(rows)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            for c in range(C):
                indices = np.random.choice(num_pixels, num_noise, replace=False)
                salt_pepper_vals = np.random.choice([0.0, 1.0], num_noise)
                perturbed_image[rows[indices], cols[indices], c] = salt_pepper_vals

    elif noise_type == 'poisson':
        target_std = 0.1 * noise_level          # Poisson std on mid-grey matches the Gaussian std
        scale_factor = 0.5 / max(target_std, 1e-6) ** 2
        noise_char_value = target_std
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = np.clip(channel_data[region_mask], 0.0, 1.0)
            noisy_pixels = np.random.poisson(region_pixels * scale_factor) / scale_factor
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] *= (1.0 + noise[region_mask])
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = 10 * noise_level
        for c in range(C):
            blurred_channel = gaussian_filter(original_image_scaled_01[:, :, c], sigma=noise_char_value)
            perturbed_image[region_mask, c] = blurred_channel[region_mask]
        
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = channel_data[region_mask]
            n1 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            n2 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            noisy_pixels = np.sqrt(np.square(region_pixels + n1) + np.square(n2))
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

def save_downsampled_image(arr, path, is_mask=False, size=(256, 256)):
    """Saves a numpy array as an image, forced to 256x256 resolution to save disk space."""
    if is_mask:
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode='RGB')

    # Use Nearest Neighbor for binary masks, Bilinear for RGB images
    img = img.resize(size, Image.NEAREST if is_mask else Image.BILINEAR)
    img.save(path)

# --------------------------------------------------------------------------
# --- 2.Dataset and Model Wrappers
# --------------------------------------------------------------------------
class GenericSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(image_size,  interpolation=Image.NEAREST), transforms.ToTensor()])
    
    def __len__(self): return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))).convert('L')
        mask_resized = self.mask_transform(mask).squeeze().numpy()

        prompt_data = {'points': [[0, 0]], 'labels': [1]} # Fallback

        labeled_mask, num_features = label(mask_resized)
        if num_features > 0:
            sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
            largest_cc = np.argmax(sizes) + 1 
            cc_mask = (labeled_mask == largest_cc)
            
            coords = np.argwhere(cc_mask)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            cropped_cc_mask = cc_mask[min_y:max_y+1, min_x:max_x+1]
            dist_transform = distance_transform_edt(cropped_cc_mask)
            
            h, w = dist_transform.shape
            y, x = np.ogrid[:h, :w]
            
            local_com_y, local_com_x = center_of_mass(cropped_cc_mask)
            sigma_y, sigma_x = max(h / 3.0, 1e-5), max(w / 3.0, 1e-5)
            spatial_weight = np.exp(-(((y - local_com_y)**2) / (2 * sigma_y**2) + ((x - local_com_x)**2) / (2 * sigma_x**2)))
            
            H_img, W_img = mask_resized.shape
            margin_y, margin_x = 0.20 * H_img, 0.20 * W_img
            
            Y_global = min_y + y
            X_global = min_x + x
            
            dist_y = np.maximum(0, np.maximum(margin_y - Y_global, Y_global - (H_img - margin_y)))
            dist_x = np.maximum(0, np.maximum(margin_x - X_global, X_global - (W_img - margin_x)))
            
            global_penalty = np.exp(-(dist_y**2 / (2 * (margin_y / 2.0)**2) + dist_x**2 / (2 * (margin_x / 2.0)**2)))
            
            weighted_edt = dist_transform * spatial_weight * global_penalty
            local_cy, local_cx = np.unravel_index(np.argmax(weighted_edt), weighted_edt.shape)
            
            cy = min_y + local_cy
            cx = min_x + local_cx
            
            cy = int(np.clip(cy, 0, mask_resized.shape[0] - 1))
            cx = int(np.clip(cx, 0, mask_resized.shape[1] - 1))
            
            prompt_data = {'points': [[cx, cy]], 'labels': [1]}
            
        return {
            'image_tensor': self.transform(image), 
            'mask_tensor': self.mask_transform(mask), 
            'path': os.path.join(self.image_dir, img_name), 
            'prompt_points': torch.tensor(prompt_data['points'], dtype=torch.float32), 
            'prompt_labels': torch.tensor(prompt_data['labels'], dtype=torch.float32), 
            'img_name': img_name
        }

def create_generic_dataloader(data_dir, batch_size=1, image_size=(1024, 512)):
    return DataLoader(GenericSegmentationDataset(root_dir=data_dir, image_size=image_size), batch_size=1, shuffle=False, num_workers=2)

class BaseModelWrapper(ABC):
    def __init__(self, model, device): self.model, self.device = model, device
    @abstractmethod
    def predict(self, batch: dict): pass

class SAM1ModelWrapper(BaseModelWrapper):
    def predict(self, batch: dict):
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = np.ascontiguousarray((image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        points, labels = batch['prompt_points'], batch['prompt_labels']
        points_np = points.squeeze(0).cpu().numpy()
        labels_np = labels.squeeze(0).cpu().numpy()

        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                self.model.reset_image() 
            except AttributeError:
                pass 
                
            self.model.set_image(image_np_rgb)
            
            masks, _, logits = self.model.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True,
            )

            # Upscale low-res logits with SAM's own postprocessing (removes the square padding)
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            if logits_tensor.dim() == 3:
                logits_tensor = logits_tensor.unsqueeze(0)

            logits_high_res = self.model.model.postprocess_masks(
                logits_tensor.to(self.model.device), self.model.input_size, self.model.original_size
            ).float()
            probs = torch.sigmoid(logits_high_res).squeeze(0).cpu().numpy()
            
        return masks.astype(np.uint8), probs

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> BaseModelWrapper:
    logger = logging.getLogger('base')
    logger.info(f"Loading SAMv1 model: type='{args.model_type}', checkpoint='{args.model_path}'")
    
    if model_name.startswith('sam_'):
        sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return SAM1ModelWrapper(predictor, device)
    else:
        raise ValueError(f"Model type '{model_name}' is not a supported SAMv1 model.")

# --------------------------------------------------------------------------
# --- 3. Core Step-Wise Robustness Analysis (Upgraded for Mean Probability)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --- 3. Core Step-Wise Robustness Analysis (Upgraded with Global & Micro IoU)
# --------------------------------------------------------------------------
def run_ranking_analysis(
    analysis_type, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, current_image_output_dir, 
    initial_pred_mask, initial_probs, best_channel_index, 
    ic_sps_all, oc_sps_all, noise_step=0.05, max_level=5.0, save_visuals=False, alpha=1.0, beta=1.0
):
    logger = logging.getLogger('base')
    analysis_results = {
        "rank_distribution": [],
        "level_data_per_noise": {}
    }
    to_tensor_transform = transforms.ToTensor()
    
    sp_areas = {int(sp_id): int(np.sum(segments_slic == sp_id)) for sp_id in superpixels_to_measure}
    max_id = int(np.max(segments_slic)) + 1

    # --- BG-IoU Spillover Baseline Calculation ---
    bg_gt = (gt_mask == 0)
    bg_pred_init = (initial_pred_mask == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0

    # --- Global IoU Baseline ---
    base_intersect = np.sum((initial_pred_mask == 1) & (gt_mask > 0))
    base_union = np.sum((initial_pred_mask == 1) | (gt_mask > 0))
    base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0

    # --- Micro IoU Baseline Arrays ---
    intersect_mask_0 = (initial_pred_mask == 1) & (gt_mask > 0)
    union_mask_0 = (initial_pred_mask == 1) | (gt_mask > 0)
    int_counts_0 = np.bincount(segments_slic[intersect_mask_0].ravel(), minlength=max_id)
    uni_counts_0 = np.bincount(segments_slic[union_mask_0].ravel(), minlength=max_id)

    # --- Step 0 Baseline Probability Ranking ---
    initial_sp_data = []
    for sp_id in superpixels_to_measure:
        mask = (segments_slic == sp_id)
        mean_p = float(np.mean(initial_probs[mask]))
        
        u_val = uni_counts_0[sp_id]
        i_val = int_counts_0[sp_id]
        sp_iou = (i_val / u_val * 100.0) if u_val > 0 else 100.0
        
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

    N_measured = len(superpixels_to_measure)
    max_possible_jumps = math.floor((N_measured**2) / 2.0) if N_measured > 1 else 1.0

    base_raw_mean_prob = np.mean([sp['mean_prob'] for sp in sorted_initial_sp_data]) if sorted_initial_sp_data else 0.0
    base_utility_e0 = base_global_iou          # utility is IoU-based; probability is only used to rank superpixels

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
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_image_01, _ = perturb_superpixels_with_level(
                orig_image_01, segments_slic, superpixels_to_perturb, noise_type, noise_level
            )
            ssim_score = structural_similarity(orig_image_01, perturbed_image_01, data_range=1.0, channel_axis=-1)

            perturbed_batch = batch.copy()
            perturbed_batch['image_tensor'] = to_tensor_transform(perturbed_image_01).unsqueeze(0).float()
            
            all_perturbed_masks, all_perturbed_probs = model_wrapper.predict(perturbed_batch)
            perturbed_pred_mask = all_perturbed_masks[best_channel_index]
            perturbed_probs = all_perturbed_probs[best_channel_index]

            if save_visuals and int(round(noise_level, 2) * 100) % 50 == 0:
                save_prefix = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}")
                save_downsampled_image(perturbed_image_01, f"{save_prefix}.png")
                save_downsampled_image(perturbed_pred_mask, f"{save_prefix}_mask.png", is_mask=True)
                overlay_image = mark_boundaries(perturbed_image_01, perturbed_pred_mask, color=(1, 0, 0), mode='thick')
                save_downsampled_image(overlay_image, f"{save_prefix}_overlay.png")

            # --- Global IoU at Step k ---
            intersect_k = np.sum((perturbed_pred_mask == 1) & (gt_mask > 0))
            union_k = np.sum((perturbed_pred_mask == 1) | (gt_mask > 0))
            global_iou_k = (intersect_k / union_k * 100.0) if union_k > 0 else 100.0

            # --- Instant Background Spillover Ratio ---
            bg_pred_pert = (perturbed_pred_mask == 0)
            intersection_bg_pert = np.sum(bg_gt & bg_pred_pert)
            union_bg_pert = np.sum(bg_gt | bg_pred_pert)
            iou_bg_pert = (intersection_bg_pert / union_bg_pert * 100.0) if union_bg_pert > 0 else 100.0
            
            delta_iou_bg = iou_bg_base - iou_bg_pert
            if delta_iou_bg > 0:
                spillover_ratio_k = delta_iou_bg / 100.0
                cumulative_spillover_ratio += spillover_ratio_k
                conditional_bleed_count += 1
            else:
                spillover_ratio_k = 0.0

            # --- Micro IoU Vectors at Step k ---
            intersect_mask_k = (perturbed_pred_mask == 1) & (gt_mask > 0)
            union_mask_k = (perturbed_pred_mask == 1) | (gt_mask > 0)
            int_counts_k = np.bincount(segments_slic[intersect_mask_k].ravel(), minlength=max_id)
            uni_counts_k = np.bincount(segments_slic[union_mask_k].ravel(), minlength=max_id)

            # --- Dynamic Step-Wise Probability & IoU Extraction ---
            current_level_sp_data = []
            for sp_id in superpixels_to_measure:
                mask = (segments_slic == sp_id)
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
            total_rank_jumps = sum(abs(current_ranks_map[sp_id] - initial_ranks_map[sp_id]) for sp_id in superpixels_to_measure)
            norm_rp_k = total_rank_jumps / max_possible_jumps if max_possible_jumps > 0 else 0.0
            peak_rp = max(peak_rp, norm_rp_k)

            # --- Step-Wise Discounted Utility (E_k) ---
            raw_prob_mean_k = np.mean([sp['mean_prob'] for sp in sorted_superpixels]) if sorted_superpixels else 0.0
            e_k = global_iou_k * (1.0 - alpha * min(1.0, norm_rp_k)) * max(0.0, 1.0 - beta * spillover_ratio_k)

            level_data_list.append({
                "step_idx": step_idx + 1,
                "noise_level": round(float(noise_level), 3),
                "ssim": round(float(ssim_score), 4),
                "global_iou": round(float(global_iou_k), 2),
                "mean_raw_prob": round(float(raw_prob_mean_k * 100.0), 2),
                "rank_perplexity": round(float(norm_rp_k), 4),
                "instant_spillover_ratio": round(float(spillover_ratio_k), 4),
                "discounted_utility": round(float(e_k), 2),
                "superpixels": sorted_superpixels
            })
        
        avg_spillover = (cumulative_spillover_ratio / conditional_bleed_count) if conditional_bleed_count > 0 else 0.0
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

# --------------------------------------------------------------------------
# --- 4. Main Inference Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM IC/OC Continuous Noise Engine - SAM1")
    parser.add_argument('--data_dir', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam_vit_h')
    parser.add_argument('--model_type', type=str, default='vit_h')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM1/sam_vit_h_4b8939.pth')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/results/sam1/prism_out_icoc')
    parser.add_argument('--txtfile', type=str, default='./img_sel.txt', help="Path to frozen image cohort text file.")    
    parser.add_argument('--num_images', type=int, default=50, help="Target qualified images.")
    parser.add_argument('--num_dataset_runs', type=int, default=1, help="Dataset iterations.")
    parser.add_argument('--noise_step', type=float, default=0.1, help="Step size for noise perturbation loop.")
    parser.add_argument('--max_level', type=float, default=5.0, help="Highest noise level of the grid (inclusive).")
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--target_class', type=str, default='person')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()]
    )
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_wrapper = get_model_wrapper(args.model_name, args, device)
    val_loader = create_generic_dataloader(data_dir=os.path.join(args.data_dir, args.target_class), image_size=tuple(args.image_size))
   
    noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
    noise_char_map = {
        'salt_pepper': 'density', 'gaussian': 'std_dev', 'speckle': 'std_dev', 
        'poisson': 'std_at_midgrey', 'gaussian_blur': 'sigma', 'rician': 'sigma'
    }

    # =========================================================================
    # PHASE 1: FROZEN COHORT SCANNING & SUPERPIXEL QUALIFICATION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Strict Anatomical Qualifier Scan...")
    if not os.path.exists(args.txtfile):
        logger.error(f"FATAL: Cohort file '{args.txtfile}' not found.")
        sys.exit()
    
    with open(args.txtfile, 'r') as f:
        frozen_image_names = set([line.strip() for line in f.readlines()])

    qualified_images = []
    for idata, batch in enumerate(tqdm(val_loader, desc="Scanning Dataset")):
        if len(qualified_images) >= args.num_images: break
        
        img_name = batch['img_name'][0]
        if img_name not in frozen_image_names: continue
            
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()

        all_pred_masks, all_probs = model_wrapper.predict(batch)
        full_roi = np.ones_like(gt_mask, dtype=bool) 
        iou_scores = [calculate_iou_score(mask, gt_mask, full_roi) for mask in all_pred_masks]
        best_channel_index = np.argmax(iou_scores)
        
        pred_mask = all_pred_masks[best_channel_index]
        initial_probs = all_probs[best_channel_index]

        segments_slic = slic(orig_image_01, n_segments=100, compactness=10, sigma=1, channel_axis=-1)

        # Absolute Anchoring (No prediction threshold filters used)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask, ic_threshold=1.0, oc_threshold=0.00)
        
        if not ic_sps_all: continue
        dice_scores = [calculate_dice_score(mask, gt_mask) for mask in all_pred_masks]

        qualified_images.append({
            'batch': batch, 'img_name': img_name, 'orig_image_01': orig_image_01, 'gt_mask': gt_mask,
            'segments_slic': segments_slic, 'pred_mask': pred_mask, 'initial_probs': initial_probs,
            'best_channel_index': best_channel_index, 'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': dice_scores[best_channel_index]
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: MULTI-RUN ROBUSTNESS INFERENCE (WITH GLOBAL SOTA)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Extractions...")
    
    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0, leave=True):
            batch, img_name = q_data['batch'], q_data['img_name']
            orig_image_01, gt_mask = q_data['orig_image_01'], q_data['gt_mask']
            segments_slic = q_data['segments_slic']
            pred_mask, initial_probs = q_data['pred_mask'], q_data['initial_probs']
            best_channel_index = q_data['best_channel_index']
            ic_sps_all, oc_sps_all = q_data['ic_sps_all'], q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, img_name.rsplit('.', 1)[0])
            os.makedirs(current_image_output_dir, exist_ok=True)
            
            final_analysis = {
                "image_info": {
                    "name": img_name.rsplit('.', 1)[0], 
                    "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)
                }
            }

            # --- 1. In-Context Analysis ---
            res_ic = run_ranking_analysis(
                'in_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, 
                ic_sps_all, ic_sps_all, noise_types, noise_char_map, 
                current_image_output_dir, pred_mask, initial_probs, best_channel_index, 
                ic_sps_all, oc_sps_all, noise_step=args.noise_step, max_level=args.max_level, save_visuals=(run_idx == 0)
            )
            final_analysis['in_context_analysis'] = res_ic

            # --- 2. Out-of-Context Analysis ---
            num_oc_to_perturb = int(0.70 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)

            res_oc = run_ranking_analysis(
                'out_of_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, 
                out_of_context_sp_to_perturb, ic_sps_all, noise_types, noise_char_map, 
                current_image_output_dir, pred_mask, initial_probs, best_channel_index, 
                ic_sps_all, oc_sps_all, noise_step=args.noise_step, max_level=args.max_level, save_visuals=(run_idx == 0)
            )
            final_analysis['out_of_context_analysis'] = res_oc

            json_save_path = os.path.join(current_image_output_dir, f"{img_name.rsplit('.', 1)[0]}_analysis.json")
            with open(json_save_path, 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
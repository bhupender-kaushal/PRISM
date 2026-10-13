import os
import argparse
import json
import logging
import random
import math
from abc import ABC, abstractmethod
from math import exp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from PIL import Image

from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.ndimage import center_of_mass, label
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'sam2' library.")
    sys.exit()

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

def calculate_rbo(list1, list2, p=0.9):
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    if not list1 or not list2: return 0.0
    sl, ll = sorted([(len(list1), list1), (len(list2), list2)])
    s, S = sl; l, L = ll
    agreement = 0.0
    for d in range(1, s + 1):
        set1, set2 = set(S[:d]), set(L[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
    for d in range(s + 1, l + 1):
        set1, set2 = set(S), set(L[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
    rbo = (1 - p) * agreement
    return max(0.000, min(1.000, rbo))

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

# --- PRISM Regional Perturbation (Color-Safe) ---
def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0
    
    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask): return perturbed_image, noise_char_value
        
    region_mask_3c = np.stack([region_mask]*3, axis=-1)

    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] += noise[region_mask_3c]
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, 0.05 * noise_level)
        rows, cols = np.where(region_mask)
        num_pixels_in_region = len(rows)
        num_noise = int(noise_char_value * num_pixels_in_region)
        if num_noise > 0:
            indices = np.random.choice(num_pixels_in_region, num_noise, replace=False)
            salt_or_pepper = np.random.choice([0.0, 1.0], num_noise)
            perturbed_image[rows[indices], cols[indices], :] = salt_or_pepper[:, np.newaxis]
            
    elif noise_type == 'poisson':
        intensity_scale = max(0.05, 3.0 - noise_level)
        for c in range(perturbed_image.shape[2]):
            channel_view = perturbed_image[:, :, c]
            channel_mask = region_mask_3c[:, :, c]
            region_pixels = channel_view[channel_mask]
            raw_noise = np.random.poisson(np.clip(region_pixels, 0, 1) * 300 * intensity_scale) / (300 * intensity_scale)
            perturbed_image[:, :, c][channel_mask] = np.clip(raw_noise, 0, 1)
            
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] *= (1 + noise[region_mask_3c])
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = 10 * noise_level
        blurred_image = np.zeros_like(original_image_scaled)
        for c in range(3):
            blurred_image[:, :, c] = gaussian_filter(original_image_scaled[:, :, c], sigma=noise_char_value)
        perturbed_image[region_mask_3c] = blurred_image[region_mask_3c]
        
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        region = perturbed_image[region_mask_3c]
        n1 = np.random.normal(0, noise_char_value, region.shape)
        n2 = np.random.normal(0, noise_char_value, region.shape)
        noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
        perturbed_image[region_mask_3c] = noisy_region

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

# --- SOTA Global Perturbation (Color-Safe) ---
def perturb_global_image(original_image_scaled_01, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled_01)
    
    if noise_type == 'gaussian':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image += noise
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.3, 0.03 * noise_level)
        mask = np.random.rand(*perturbed_image.shape[:2]) < noise_char_value
        val = np.random.choice([0.0, 1.0], np.sum(mask))
        for c in range(3):
            perturbed_image[mask, c] = val
            
    elif noise_type == 'poisson':
        intensity_scale = max(0.05, 3.2 - noise_level)
        noisy_img = np.zeros_like(perturbed_image, dtype=np.float32)
        for c in range(perturbed_image.shape[2]):
            channel_data = np.clip(perturbed_image[:, :, c], 0, 1)
            scaled_counts = channel_data * 300 * intensity_scale
            noisy_img[:, :, c] = np.random.poisson(scaled_counts) / (300 * intensity_scale)
        perturbed_image = np.clip(noisy_img, 0, 1)
            
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image *= (1 + noise)
        
    elif noise_type == 'gaussian_blur':
        sigma_val = (noise_level * 10)
        for c in range(3):
            perturbed_image[:, :, c] = gaussian_filter(perturbed_image[:, :, c], sigma=sigma_val)
            
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        n1 = np.random.normal(0, noise_char_value, perturbed_image.shape)
        n2 = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))

    return np.clip(perturbed_image, 0.0, 1.0)


# --------------------------------------------------------------------------
# --- 2. Generic Data Loading and SAM-2 Model Wrappers
# --------------------------------------------------------------------------
class GenericSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
    
    def __len__(self): return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))).convert('L')
        mask_resized = self.mask_transform(mask).squeeze().numpy()

        prompt_data = {'points': [[0, 0]], 'labels': [1]} 

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
    def predict(self, batch: dict) -> np.ndarray: pass
    @abstractmethod
    def get_target_layer(self, name: str) -> nn.Module: pass

class SAM2ModelWrapper(BaseModelWrapper):
    def predict(self, batch: dict) -> np.ndarray:
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image_np_rgb = np.ascontiguousarray(image_np_rgb)
        
        points, labels = batch['prompt_points'], batch['prompt_labels']
        if isinstance(points, torch.Tensor) and points.dim() == 3 and points.shape[0] == 1: 
            points = points.squeeze(0)
        if isinstance(labels, torch.Tensor) and labels.dim() == 2 and labels.shape[0] == 1: 
            labels = labels.squeeze(0)
        
        points, labels = points.to(self.device), labels.to(self.device)
        points_in = points.unsqueeze(0) if points.dim() == 2 else points
        labels_in = labels.unsqueeze(0) if labels.dim() == 1 else labels

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.model.set_image(image_np_rgb)
            masks, _, _ = self.model.predict(
                point_coords=points_in, 
                point_labels=labels_in, 
                multimask_output=True
            )
        return masks.astype(np.uint8)

    def get_target_layer(self, name: str) -> nn.Module:
        base = self.model.model.image_encoder
        try:
            curr_obj = base
            for part in name.split('.'): curr_obj = getattr(curr_obj, part)
            return curr_obj
        except Exception:
            for full_name, module in self.model.model.named_modules():
                if full_name.endswith(name) or name in full_name: return module
            raise AttributeError(f"Could not find target layer '{name}'.")

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> BaseModelWrapper:
    logger = logging.getLogger('base')
    logger.info(f"Loading model: {model_name}")
    if model_name.startswith('sam2_'):
        model_cfg_path = os.path.abspath(args.model_config)
        cfg_dir = os.path.dirname(model_cfg_path)
        cfg_name = os.path.basename(model_cfg_path)
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
            from hydra import initialize_config_dir, compose
            initialize_config_dir(config_dir=cfg_dir)
            _ = compose(config_name=cfg_name)
            sam_model = build_sam2(cfg_name, args.model_path, device=device)
        except Exception as e:
            sam_model = build_sam2(model_cfg_path, args.model_path, device=device)
        predictor = SAM2ImagePredictor(sam_model)
        return SAM2ModelWrapper(predictor, device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


# --------------------------------------------------------------------------
# --- 3. Core Analysis and Visualization
# --------------------------------------------------------------------------

def run_ranking_analysis(
    analysis_type, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, sota_max_valid_levels,
    current_image_output_dir, initial_pred_mask, best_channel_index, 
    noise_step=0.1, save_visuals=False
):
    logger = logging.getLogger('base')
    analysis_results = {}
    to_tensor_transform = transforms.ToTensor()
    
    lambda_param = 1.0
    beta_param = 1.0
    max_id = segments_slic.max() + 1 

    bg_gt = (gt_mask == 0)
    bg_pred_init = (initial_pred_mask == 0)
    intersection_bg_base = np.sum(bg_gt & bg_pred_init)
    union_bg_base = np.sum(bg_gt | bg_pred_init)
    iou_bg_base = (intersection_bg_base / union_bg_base * 100.0) if union_bg_base > 0 else 100.0

    intersect_mask_init = (initial_pred_mask == 1) & (gt_mask > 0)
    union_mask_init = (initial_pred_mask == 1) | (gt_mask > 0)
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
        level_0_entry = {noise_char_map.get(noise_type, 'noise_value'): 0.0, "ssim_score": 1.0, "image_path": "original", "superpixels": sorted_initial_sp_data, "instant_spillover_ratio": 0.0}
        level_data_list = [level_0_entry]
        rbo_scores_list, sp_iou_history, sp_rank_history = [], {sp_id: [] for sp_id in superpixels_to_measure}, {sp_id: [] for sp_id in superpixels_to_measure}
        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0

        current_pred_mask = initial_pred_mask

        # --- OPTIMIZATION: Dynamically cap PRISM inference to SOTA's boundary ---
        max_level = sota_max_valid_levels.get(noise_type, 0.0)
        
        if max_level < noise_step:
            noise_levels_arr = [] # SOTA failed on first step, skip loop entirely
        else:
            noise_levels_arr = np.arange(noise_step, max_level + 1e-5, noise_step)

        for step_idx, noise_level in enumerate(noise_levels_arr): 
            perturbed_image_01, noise_char_value = perturb_superpixels_with_level(orig_image_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_image_01, data_range=1.0, channel_axis=-1)
            
            # --- SSIM FREEZE LOGIC (Secondary safety check) ---
            if ssim_score >= 0.70:
                perturbed_batch = batch.copy()
                perturbed_batch['image_tensor'] = to_tensor_transform(perturbed_image_01).unsqueeze(0).float()
                
                all_perturbed_masks = model_wrapper.predict(perturbed_batch)
                current_pred_mask = all_perturbed_masks[best_channel_index]

            else:
                logger.info(f"[PRISM] SSIM below threshold for noise_type: {noise_type}, noise_level: {noise_level:.2f}. Breaking.")
                break

            bg_pred_pert = (current_pred_mask == 0)
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

            intersect_mask_pert = (initial_pred_mask == 1) & (current_pred_mask == 1)
            union_mask_pert = (initial_pred_mask == 1) | (current_pred_mask == 1)
            
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
            
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            rbo_scores_list.append(round(calculate_rbo(initial_rank_list, current_rank_list, p=0.9), 4))
            
            level_data_list.append({"noise_level": round(noise_level, 2), "instant_spillover_ratio": round(spillover_ratio_at_j, 4), "superpixels": sorted_superpixels})


        avg_spillover = cumulative_spillover_ratio / conditional_bleed_count if conditional_bleed_count > 0 else 0.0
        spillover_discount = math.exp(-beta_param * avg_spillover)
        
        reliability_results = []
        for sp_id in superpixels_to_measure:
            avg_iou = np.mean(sp_iou_history[sp_id]) if len(sp_iou_history[sp_id]) > 0 else 0.0
            initial_iou = initial_iou_map.get(sp_id, 0.0)
            rank_list = [r for r in sp_rank_history[sp_id] if r is not None]
            num_changes = sum(1 for i in range(1, len(rank_list)) if rank_list[i] != rank_list[i-1])
            sum_of_jumps = sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list)))
            rank_perplexity = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0
            rank_perplexity = rank_perplexity / len(superpixels_to_measure) if len(superpixels_to_measure) > 0 else 0.0
            confidence_weight = exp(-lambda_param * rank_perplexity)
            weighted_iou = avg_iou * confidence_weight
            base_reliability_score = (weighted_iou / 100.0) if initial_iou > 1e-9 else 0.0

            reliability_results.append({
                "sp_id": int(sp_id),
                "base_reliability_score": base_reliability_score,  
                "weighted_iou": round(weighted_iou, 4), "avg_iou": round(avg_iou, 4),
                "initial_iou": round(initial_iou, 4), "rank_perplexity": round(rank_perplexity, 4)
            })

        analysis_results[noise_type] = {
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "spillover_metrics": {"conditional_average_spillover": round(avg_spillover, 4), "global_spillover_discount": round(spillover_discount, 4)},
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['base_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results


# --------------------------------------------------------------------------
# --- 4. Main Two-Phase Pipeline + Ablation
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="High-Speed Two-Phase Model Reliability Inference + Rank Ablation (SAM 2)")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large')
    parser.add_argument('--model_path', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_config', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--target_layer_name', type=str, default='sam_mask_decoder.output_upscaling.4')
    parser.add_argument('--output_dir', type=str, default='./sota_prism_ablation_results')
    parser.add_argument('--num_images', type=int, default=50, help="Strictly process exactly {args.num_images} qualified images for ablation study.")
    parser.add_argument('--noise_step', type=float, default=0.01, help="Step size for noise loop (0.1 for Ablation).")
    parser.add_argument('--noise_max', type=float, default=3.0, help="Maximum noise level for perturbation sweeps.")
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--target_class', type=str, default='person')
    parser.add_argument('--txtfile', type=str, default='./img_sel.txt', help="Path to the selected images text file containing qualified image names.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_wrapper = get_model_wrapper(args.model_name, args, device)
    val_loader = create_generic_dataloader(data_dir=args.data_dir+'/'+args.target_class, image_size=tuple(args.image_size))
   
    # noise_types = [ 'gaussian_blur']
    noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

    
    noise_char_map = {'salt_pepper': 'density', 'gaussian': 'std_dev', 'speckle': 'std_dev', 'poisson': 'intensity_scale', 'gaussian_blur': 'sigma', 'rician': 'sigma'}

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION (Strictly {args.num_images} Images)
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Dataset to Qualify EXACTLY {args.num_images} Images for Ablation...")
    qualified_images = []

    cohort_file = args.txtfile

    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Frozen cohort file '{cohort_file}' not found.")
        sys.exit()

    with open(cohort_file, 'r') as f:
        frozen_image_names = set([line.strip() for line in f.readlines()])
    
    for idata, batch in enumerate(tqdm(val_loader, desc="Scanning Dataset")):
        if len(qualified_images) >= args.num_images:
            break
        img_name = batch['img_name'][0]
        if img_name not in frozen_image_names:
            continue
        
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()

        all_pred_masks = model_wrapper.predict(batch)
        full_roi = np.ones_like(gt_mask, dtype=bool) 
        iou_scores = [calculate_iou_score(mask, gt_mask, full_roi) for mask in all_pred_masks]
        best_channel_index = np.argmax(iou_scores)

        # if iou_scores[best_channel_index] <= 60.0: continue

        pred_mask = all_pred_masks[best_channel_index]

        segments_slic = slic(orig_image_01, n_segments=500, compactness=10, sigma=1, channel_axis=-1)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask, ic_threshold=1.0, oc_threshold=0.00)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_score(pred_mask, gt_mask, segments_slic == sp_id) > 50: qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue
        
        qualified_images.append({
            'batch': batch, 'img_name': img_name, 'orig_image_01': orig_image_01, 'gt_mask': gt_mask,
            'segments_slic': segments_slic, 'pred_mask': pred_mask, 'all_pred_masks': all_pred_masks,
            'best_channel_index': best_channel_index, 'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all
        })

        

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) < args.num_images:
        logger.warning(f"Only found {len(qualified_images)} qualified images. Continuing with available cohort.")

    # =========================================================================
    # PHASE 2: SOTA (First) -> PRISM (Truncated) SWEEPS
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing SOTA (Upper Bound) & PRISM Noise Analysis ({args.noise_step} to {args.noise_max})...")
    
    all_images_analysis = {}
    to_tensor_transform = transforms.ToTensor()
    
    for q_data in tqdm(qualified_images, desc="Processing Image Cohort"):

        logger.info(f"\nProcessing Image: {q_data['img_name']} ({len(all_images_analysis)+1}/{len(qualified_images)})")

        batch, img_name = q_data['batch'], q_data['img_name']
        orig_image_01, gt_mask = q_data['orig_image_01'], q_data['gt_mask']
        segments_slic, pred_mask = q_data['segments_slic'], q_data['pred_mask']

        best_channel_index, qualified_ic_sp, oc_sps_all = q_data['best_channel_index'], q_data['qualified_ic_sp'], q_data['oc_sps_all']

        final_analysis = {"image_info": {"name": img_name.rsplit('.', 1)[0]}}

        # --- 1. SOTA Pipeline (Run FIRST to Establish Truncation Boundary) ---
        
        full_roi = np.ones_like(gt_mask, dtype=bool)
        Af_c = calculate_iou_score(pred_mask, gt_mask, full_roi) / 100.0
        if Af_c == 0: Af_c = 1e-6 

        sota_results = {}
        sota_max_valid_levels = {}
        
        for noise_type in noise_types:
            sota_results[noise_type] = []
            max_level_for_this_noise = 0.0
            
            for noise_level in np.arange(args.noise_step, args.noise_max + 1e-5, args.noise_step):
                pert_img = perturb_global_image(orig_image_01, noise_type, noise_level)
                ssim_score=structural_similarity(orig_image_01, pert_img, data_range=1.0, channel_axis=-1)

                if ssim_score >= 0.70:
                    max_level_for_this_noise = noise_level 
                    pert_batch = batch.copy()
                    pert_batch['image_tensor'] = to_tensor_transform(pert_img).unsqueeze(0).float()
                    all_pert_masks = model_wrapper.predict(pert_batch)
                else:
                    logger.info(f"[SOTA] Image: {img_name}, Noise: {noise_type}, Level: {round(noise_level, 2)}, SSIM: {round(ssim_score, 4)} - Skipping Prediction (SSIM < 0.70)")
                    break

                pert_mask = all_pert_masks[best_channel_index]
                
                Af_ps = calculate_iou_score(pert_mask, gt_mask, full_roi) / 100.0
                gamma_r = Af_ps / Af_c  # SOTA mathematical formulation
                
                sota_results[noise_type].append({'noise_level': round(noise_level, 2), 'gamma_r': gamma_r})
                
            sota_max_valid_levels[noise_type] = max_level_for_this_noise

        final_analysis['sota_analysis'] = sota_results

        # --- 2. PRISM Pipeline (Dynamically Capped via sota_max_valid_levels) ---
        res_ic = run_ranking_analysis('in_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, qualified_ic_sp, qualified_ic_sp, noise_types, noise_char_map, sota_max_valid_levels, '', pred_mask, best_channel_index, noise_step=args.noise_step, save_visuals=False)
        final_analysis['in_context_analysis'] = res_ic
        
        num_oc_to_perturb = int(0.70 * len(oc_sps_all))
        out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
        res_oc = run_ranking_analysis('out_of_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, out_of_context_sp_to_perturb, qualified_ic_sp, noise_types, noise_char_map, sota_max_valid_levels, '', pred_mask, best_channel_index, noise_step=args.noise_step, save_visuals=False)
        final_analysis['out_of_context_analysis'] = res_oc

        all_images_analysis[img_name] = final_analysis

    # =========================================================================
    # PHASE 3: RANK ENTROPY ABLATION (STRICT COMMON GROUND)
    # =========================================================================
    logger.info("\n=== [PHASE 3] Rank Entropy Ablation: Strict Common Ground Evaluation ===")
    
    entropy_results = []
    img_names_ordered = [q['img_name'] for q in qualified_images] 
    
    for noise_type in noise_types:
        min_valid_steps = 999 
        
        for img_name in img_names_ordered:
            data = all_images_analysis[img_name]
            sota_len = len(data['sota_analysis'][noise_type])
            ic_len = len(data['in_context_analysis'][noise_type]['level_data']) - 1
            oc_len = len(data['out_of_context_analysis'][noise_type]['level_data']) - 1
            
            min_valid_steps = min(min_valid_steps, sota_len, ic_len, oc_len)
            
        logger.info(f"[{noise_type.upper()}] Common Ground strictly established at {min_valid_steps} perturbation steps.")
        
        if min_valid_steps == 0:
            logger.warning(f"[{noise_type.upper()}] An image failed SSIM instantly. Cannot compute entropy for this noise.")
            continue
            
        sota_rank_matrix = {img: [] for img in img_names_ordered}
        prism_rank_matrix = {img: [] for img in img_names_ordered}
        
        for step_idx in range(min_valid_steps):
            sota_level_scores = []
            prism_level_scores = []
            
            for img_name in img_names_ordered:
                data = all_images_analysis[img_name]
                
                sota_gamma = data['sota_analysis'][noise_type][step_idx]['gamma_r']
                sota_level_scores.append(sota_gamma)
                
                ic_lvl_data = data['in_context_analysis'][noise_type]['level_data'][step_idx + 1]
                oc_lvl_data = data['out_of_context_analysis'][noise_type]['level_data'][step_idx + 1]
                
                ic_iou = np.mean([sp['iou'] for sp in ic_lvl_data['superpixels']]) / 100.0 if ic_lvl_data['superpixels'] else 0.0
                oc_iou = np.mean([sp['iou'] for sp in oc_lvl_data['superpixels']]) / 100.0 if oc_lvl_data['superpixels'] else 0.0
                
                R_ic = ic_iou * math.exp(-1.0 * ic_lvl_data['instant_spillover_ratio'])
                R_oc = oc_iou * math.exp(-1.0 * oc_lvl_data['instant_spillover_ratio'])
                
                R_prism = (2 * R_ic * R_oc) / (R_ic + R_oc) if (R_ic + R_oc) > 0 else 0.0
                prism_level_scores.append(R_prism)
            
            jitter_s = np.random.normal(0, 1e-9, len(sota_level_scores))
            sota_ranks = stats.rankdata(-(np.array(sota_level_scores) + jitter_s), method='ordinal')
            
            jitter_p = np.random.normal(0, 1e-9, len(prism_level_scores))
            prism_ranks = stats.rankdata(-(np.array(prism_level_scores) + jitter_p), method='ordinal')
            
            for i, img_name in enumerate(img_names_ordered):
                sota_rank_matrix[img_name].append(sota_ranks[i])
                prism_rank_matrix[img_name].append(prism_ranks[i])
                
        for img_name in img_names_ordered:
            # Dynamically adapt bin size to number of images
            s_counts = np.bincount(sota_rank_matrix[img_name], minlength=len(img_names_ordered)+1)[1:]
            p_counts = np.bincount(prism_rank_matrix[img_name], minlength=len(img_names_ordered)+1)[1:]
            
            H_sota = stats.entropy(s_counts, base=2)
            H_prism = stats.entropy(p_counts, base=2)
            
            entropy_results.append({'Noise Type': noise_type, 'Image': img_name, 'SOTA Entropy': H_sota, 'PRISM Entropy': H_prism})

    # =========================================================================
    # PHASE 4: ABLATION VISUALIZATION, STATISTICAL PROOF & JSON EXPORT
    # =========================================================================
    df_entropy = pd.DataFrame(entropy_results)
    
    logger.info("\n" + "="*70)
    logger.info("FORMAL STATISTICAL PROOFS (SHAPIRO-WILK, T-TEST, WILCOXON)")
    logger.info("="*70)
    
    global_summary_data = []
    
    for noise_type in noise_types:
        if df_entropy.empty or noise_type not in df_entropy['Noise Type'].unique():
            logger.warning(f"Skipping statistical proof for {noise_type.upper()}: No valid common ground data found.")
            continue

        df_sub = df_entropy[df_entropy['Noise Type'] == noise_type].copy()
        
        sota_entropies = df_sub['SOTA Entropy'].values
        prism_entropies = df_sub['PRISM Entropy'].values
        
        # 1. Normality Test (Shapiro-Wilk)
        differences = sota_entropies - prism_entropies
        if np.allclose(sota_entropies, prism_entropies) or len(differences) < 3:
            shapiro_stat, shapiro_p = 1.0, 1.0
            is_normal = True
        else:
            shapiro_stat, shapiro_p = stats.shapiro(differences)
            is_normal = shapiro_p > 0.05
            
        # 2. Parametric: Paired T-Test
        if np.allclose(sota_entropies, prism_entropies):
            t_stat, t_p_val = 0.0, 1.0 
        else:
            t_stat, t_p_val = stats.ttest_rel(sota_entropies, prism_entropies, alternative='greater')
            
        # 3. Non-Parametric: Wilcoxon Signed-Rank Test
        if np.allclose(sota_entropies, prism_entropies):
            w_stat, w_p_val, cles, r_rb = 0.0, 1.0, 0.0, 0.0 
        else:
            w_stat, w_p_val = stats.wilcoxon(sota_entropies, prism_entropies, alternative='greater')
            cles = np.mean(sota_entropies > prism_entropies)
            diff_nz = differences[differences != 0]
            if len(diff_nz) == 0:
                r_rb = 0.0
            else:
                ranks = stats.rankdata(np.abs(diff_nz))
                w_plus = np.sum(ranks[diff_nz > 0])
                w_minus = np.sum(ranks[diff_nz < 0])
                r_rb = (w_plus - w_minus) / (w_plus + w_minus)
            
        mean_sota, std_sota = np.mean(sota_entropies), np.std(sota_entropies)
        mean_prism, std_prism = np.mean(prism_entropies), np.std(prism_entropies)
        
        logger.info(f"\n[STATISTICAL PROOF] {noise_type.upper()} NOISE:")
        logger.info(f"    -> PRISM Mean Entropy: {mean_prism:.4f} bits | SOTA Mean Entropy: {mean_sota:.4f} bits")
        
        logger.info(f"    -> Shapiro-Wilk (Normality): W = {shapiro_stat:.4f}, p = {shapiro_p:.4e} (Normal: {is_normal})")
        logger.info(f"    -> Paired t-test: t = {t_stat:.4f}, p = {t_p_val:.4e}")
        logger.info(f"    -> Wilcoxon Signed-Rank: W = {w_stat:.1f}, p = {w_p_val:.4e}")
        
        if is_normal:
            final_p_val = t_p_val
            test_used = "Paired t-test"
        else:
            final_p_val = w_p_val
            test_used = "Wilcoxon Signed-Rank"
            
        logger.info(f"    -> Selected Test: {test_used} (based on Shapiro-Wilk result)")
        
        if final_p_val < 0.05:
            logger.info(f"    -> Conclusion: Null hypothesis rejected ({test_used}). PRISM is significantly more stable.")
            sig_marker = "***" if final_p_val < 0.001 else "**" if final_p_val < 0.01 else "*"
        else:
            logger.info(f"    -> Conclusion: Failed to reject null hypothesis at alpha=0.05 ({test_used}).")
            sig_marker = "ns"
            
        global_summary_data.append({
            'Noise_Type': noise_type.upper(),
            'SOTA_Mean_Entropy': float(mean_sota), 'SOTA_Std_Entropy': float(std_sota),
            'PRISM_Mean_Entropy': float(mean_prism), 'PRISM_Std_Entropy': float(std_prism),
            'shapiro_p': float(shapiro_p), 'is_normal': is_normal,
            't_statistic': float(t_stat), 't_p_value': float(t_p_val),
            'w_statistic': float(w_stat), 'w_p_value': float(w_p_val),
            'selected_test': test_used, 'final_p_value': float(final_p_val),
            'CLES': float(cles), 'Rank_Biserial': float(r_rb), 'significance_marker': sig_marker
        })
            
        plt.figure(figsize=(14, 6))
        x = np.arange(len(df_sub))
        width = 0.35
        
        plt.bar(x - width/2, df_sub['SOTA Entropy'], width, label=r'SOTA (Global $\gamma_r$)', color='salmon', edgecolor='black')
        plt.bar(x + width/2, df_sub['PRISM Entropy'], width, label=r'PRISM (Regional $R$)', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Cohort Image Identifiers', fontsize=14)
        plt.ylabel('Rank Entropy ($H_i$) [bits]', fontsize=14)
        plot_title = f'Discriminative Rank Stability Ablation: {noise_type.upper()} Noise\nLower Entropy = Stable Ranking | SOTA vs PRISM: p = {final_p_val:.4e} ({sig_marker}) [{test_used}]'
        plt.title(plot_title, fontsize=16, fontweight='bold')
        
        plt.xticks(x, [f"Img {i+1}" for i in range(len(df_sub))], fontsize=12, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"Ablation_RankEntropy_{noise_type}.png"), dpi=300)
        plt.close()
        
    if global_summary_data:
        df_summary = pd.DataFrame(global_summary_data)
        plt.figure(figsize=(12, 7))
        x_summary = np.arange(len(df_summary))
        width_summary = 0.35
        
        plt.bar(x_summary - width_summary/2, df_summary['SOTA_Mean_Entropy'], width_summary, yerr=df_summary['SOTA_Std_Entropy'], capsize=5, label=r'SOTA (Global $\gamma_r$)', color='salmon', edgecolor='black')
        plt.bar(x_summary + width_summary/2, df_summary['PRISM_Mean_Entropy'], width_summary, yerr=df_summary['PRISM_Std_Entropy'], capsize=5, label=r'PRISM (Regional $R$)', color='dodgerblue', edgecolor='black')
        
        plt.xlabel('Perturbation Types', fontsize=14, fontweight='bold')
        plt.ylabel(r'Average Rank Entropy ($\bar{H}$) [bits]', fontsize=14, fontweight='bold')
        plt.title('Global Rank Stability Comparison Across All Perturbations\nLower Average Entropy = Superior Rank Stability', fontsize=16, fontweight='bold')
        plt.xticks(x_summary, df_summary['Noise_Type'], fontsize=12, rotation=15)
        
        for idx, row in df_summary.iterrows():
            max_y = max(row['SOTA_Mean_Entropy'] + row['SOTA_Std_Entropy'], row['PRISM_Mean_Entropy'] + row['PRISM_Std_Entropy'])
            test_abbr = "T" if row['selected_test'] == "Paired t-test" else "W"
            plt.text(idx, max_y + 0.1, f"{row['significance_marker']} ({test_abbr})", ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
        
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "Ablation_RankEntropy_ALL_SUMMARY.png"), dpi=300)
        plt.close()

    json_export_data = {"statistical_summary": global_summary_data, "raw_image_entropies": entropy_results}
    with open(os.path.join(args.output_dir, "Ablation_Statistical_Results.json"), 'w') as f:
        json.dump(json_export_data, f, indent=4, cls=NpEncoder)
        
    logger.info(f"\n--- Ablation Finished. Visuals and JSON Results saved to {args.output_dir} ---")

if __name__ == "__main__":
    main()
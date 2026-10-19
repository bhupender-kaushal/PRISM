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
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
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

def perturb_superpixels_with_level(original_image_scaled_01, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    perturbed_image = np.copy(original_image_scaled_01)
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

    # elif noise_type == 'poisson':
    #     # Normalized Intensity: Ensure lambda parameter is positive and scaled
    #     intensity_scale = max(0.05, 3.0 - noise_level)

    #     for c in range(perturbed_image.shape[2]):
    #         channel_view = perturbed_image[:, :, c]
    #         channel_mask = region_mask_3c[:, :, c]
    #         region_pixels = channel_view[channel_mask]
    #         raw_noise = np.random.poisson(np.clip(region_pixels, 0, 1) * 100 * intensity_scale) / (100 * intensity_scale)

    #         noisy_pixels = np.clip(raw_noise, 0, 1)
    #         perturbed_image[:, :, c][region_mask_3c[:, :, c]] = noisy_pixels
        
    elif noise_type == 'speckle':
        noise_char_value = 0.1 * noise_level
        noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
        perturbed_image[region_mask_3c] *= (1 + noise[region_mask_3c])
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = 0.5 * noise_level
        blurred_image = gaussian_filter(original_image_scaled_01, sigma=(noise_char_value, noise_char_value, 0))
        perturbed_image[region_mask_3c] = blurred_image[region_mask_3c]
        
    elif noise_type == 'rician':
        noise_char_value = 0.1 * noise_level
        region = perturbed_image[region_mask_3c]
        n1 = np.random.normal(0, noise_char_value, region.shape)
        n2 = np.random.normal(0, noise_char_value, region.shape)
        noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
        perturbed_image[region_mask_3c] = noisy_region

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

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
def calculate_eigencam(model_wrapper: BaseModelWrapper, target_layer_name: str, batch: dict, image_size: tuple) -> np.ndarray:
    feature_maps = []
    try:
        target_layer = model_wrapper.get_target_layer(target_layer_name)
    except Exception as e:
        return np.zeros(image_size[:2], dtype=np.float32)
    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    _ = model_wrapper.predict(batch)
    hook.remove()
    
    if not feature_maps: return np.zeros(image_size[:2], dtype=np.float32)
    
    features = feature_maps[0].squeeze(0).to(torch.float32)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    pca = PCA(n_components=1).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    out_size = (int(image_size[0]), int(image_size[1]))
    saliency_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(saliency_tensor, size=out_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None, prompt_points=None):
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    display_image_rgb = np.array(Image.fromarray((display_image_rgb * 255).astype(np.uint8)).resize((512, 512))) / 255.0
    gt_mask = np.array(Image.fromarray(gt_mask.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    pred_mask = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    all_sp_map = np.array(Image.fromarray(all_sp_map.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    in_context_map = np.array(Image.fromarray(in_context_map.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    out_of_context_map = np.array(Image.fromarray(out_of_context_map.astype(np.uint8)).resize((512, 512), Image.NEAREST))
    eigencam_heatmap = np.array(Image.fromarray((eigencam_heatmap * 255).astype(np.uint8)).resize((512, 512))) / 255.0

    fig, axes = plt.subplots(1, 7, figsize=(42, 6))
    axes[0].imshow(display_image_rgb)
    if prompt_points is not None:
        for point in prompt_points:
            x, y = point
            axes[0].scatter(x * (512 / orig_image.shape[1]), y * (512 / orig_image.shape[0]), c='red', s=200, marker='*', edgecolors='yellow', linewidths=1)
    axes[0].set_title("Original w/ Prompts"); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
    axes[3].imshow(mark_boundaries(display_image_rgb, all_sp_map)); axes[3].set_title("All Superpixels"); axes[3].axis('off')
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map)); axes[4].set_title("In-Context SPs"); axes[4].axis('off')
    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map)); axes[5].set_title("Out-of-Context SPs"); axes[5].axis('off')
    axes[6].imshow(display_image_rgb)
    im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=0.5)
    axes[6].set_title("EigenCAM"); axes[6].axis('off')
    plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def run_ranking_analysis(
    analysis_type, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, ssim_data_collector,
    current_image_output_dir, initial_pred_mask, best_channel_index, 
    noise_step=0.1, save_visuals=False
):
    logger = logging.getLogger('base')
    analysis_results = {}
    to_tensor_transform = transforms.ToTensor()
    
    lambda_param = 1.0
    beta_param = 1.0
    max_id = segments_slic.max() + 1 

    # --- BG-IoU SPILLOVER BASELINE CALCULATION ---
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
        if save_visuals:
            noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
            os.makedirs(noise_specific_dir, exist_ok=True)
        
        level_0_entry = {noise_char_map.get(noise_type, 'noise_value'): 0.0, "ssim_score": 1.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
        level_data_list = [level_0_entry]

        rbo_scores_list, sp_iou_history, sp_rank_history = [], {sp_id: [] for sp_id in superpixels_to_measure}, {sp_id: [] for sp_id in superpixels_to_measure}

        cumulative_spillover_ratio = 0.0
        conditional_bleed_count = 0

        noise_levels_arr = np.arange(noise_step, 3.1, noise_step)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_image_01, noise_char_value = perturb_superpixels_with_level(orig_image_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_image_01, data_range=1.0, channel_axis=-1)
            
            is_final_step = False
            if ssim_score < 0.70: 
                logger.info(f"\n[SSIM TRIGGER] SSIM ({ssim_score:.4f}) dropped below 0.70 at {noise_type.upper()} level {noise_level:.2f}. Halting.")
                is_final_step = True

            ssim_data_collector.append({"Analysis Type": analysis_type, "Noise Type": noise_type, "Noise Level": noise_level, "SSIM": ssim_score})

            perturbed_batch = batch.copy()
            perturbed_batch['image_tensor'] = to_tensor_transform(perturbed_image_01).unsqueeze(0).float()
            
            all_perturbed_masks = model_wrapper.predict(perturbed_batch)
            perturbed_pred_mask = all_perturbed_masks[best_channel_index]

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
                perturbed_image_to_save = (perturbed_image_01 * 255).astype(np.uint8)
                Image.fromarray(perturbed_image_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                
                mask_to_save = (perturbed_pred_mask * 255).astype(np.uint8)
                Image.fromarray(mask_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                
                overlay_image = mark_boundaries(perturbed_image_01, perturbed_pred_mask, color=(1, 0, 0), mode='thick')
                overlay_to_save = (overlay_image * 255).astype(np.uint8)
                Image.fromarray(overlay_to_save).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            if is_final_step:
                break 

            intersect_mask_pert = (initial_pred_mask == 1) & (perturbed_pred_mask == 1)
            union_mask_pert = (initial_pred_mask == 1) | (perturbed_pred_mask == 1)
            
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
            
            for rank, sp_data in enumerate(sorted_superpixels, 1): sp_data['rank'] = rank
            
            level_data_list.append({
                "noise_level": round(noise_level, 2), 
                "instant_spillover_ratio": round(spillover_ratio_at_j, 4),
                "superpixels": sorted_superpixels
            })
        
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
# --- 4. Main Two-Phase Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="High-Speed Two-Phase Model Reliability Inference (SAM 2)")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large')
    parser.add_argument('--model_path', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_config', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--target_layer_name', type=str, default='sam_mask_decoder.output_upscaling.4')
    parser.add_argument('--output_dir', type=str, default='./new_expresults_50')

    parser.add_argument('--txtfile', type=str, default='./img_sel.txt', help="Path to the selected images text file containing qualified image names.")    

    parser.add_argument('--num_images', type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument('--num_dataset_runs', type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument('--noise_step', type=float, default=0.05, help="Step size for noise loop.")
    
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--perturb_ratio', type=float, default=1.0)
    parser.add_argument('--target_class', type=str, default='person')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_wrapper = get_model_wrapper(args.model_name, args, device)
    val_loader = create_generic_dataloader(data_dir=args.data_dir+'/'+args.target_class, image_size=tuple(args.image_size))
   
    noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
    noise_char_map = {'salt_pepper': 'density', 'gaussian': 'std_dev', 'speckle': 'std_dev', 'poisson': 'intensity_scale', 'gaussian_blur': 'sigma', 'rician': 'sigma'}

    # =========================================================================
    # PHASE 1: IMAGE QUALIFICATION SELECTION
    # =========================================================================

    logger.info(f"\n[PHASE 1] Scanning Dataset to Qualify {args.num_images} Images...")
    qualified_images = []

    cohort_file = args.txtfile

    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Frozen cohort file '{cohort_file}' not found.")
        sys.exit()
    
    with open(cohort_file, 'r') as f:
        frozen_image_names = set([line.strip() for line in f.readlines()])


    for idata, batch in enumerate(tqdm(val_loader, desc="Scanning Dataset")):
        if len(qualified_images) >= args.num_images: break
        
        img_name = batch['img_name'][0]
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()

        all_pred_masks = model_wrapper.predict(batch)
        full_roi = np.ones_like(gt_mask, dtype=bool) 
        iou_scores = [calculate_iou_score(mask, gt_mask, full_roi) for mask in all_pred_masks]
        best_channel_index = np.argmax(iou_scores)

        # if iou_scores[best_channel_index] <= 60.0: continue

        pred_mask = all_pred_masks[best_channel_index]

        # segments_slic = slic(orig_image_01, n_segments=500, compactness=10, sigma=1, channel_axis=-1)


        ## for quickshift.
        segments_slic = quickshift(orig_image_01, kernel_size=5, max_dist=10, ratio=0.5)
        
        ## for Watershed. 
        # gradient = sobel(color.rgb2gray(orig_image_01))
        # grid = regular_grid(orig_image_01.shape[:2], n_points=500) # Assuming target_segments is 500
        # markers = np.zeros(orig_image_01.shape[:2], dtype=int)

        # target_shape = markers[grid].shape
        # markers[grid] = np.arange(1, markers[grid].size + 1).reshape(target_shape)
        
        # segments_slic = watershed(gradient, markers, compactness=0.001)
        
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask, ic_threshold=1.0, oc_threshold=0.00)
        
        qualified_ic_sp = []
        for sp_id in ic_sps_all:
            if calculate_iou_score(pred_mask, gt_mask, segments_slic == sp_id) > 50:
                qualified_ic_sp.append(sp_id)
        
        if not qualified_ic_sp: continue
        
        dice_scores = [calculate_dice_score(mask, gt_mask) for mask in all_pred_masks]

        qualified_images.append({
            'batch': batch, 'img_name': img_name, 'orig_image_01': orig_image_01, 'gt_mask': gt_mask,
            'segments_slic': segments_slic, 'pred_mask': pred_mask, 'all_pred_masks': all_pred_masks,
            'best_channel_index': best_channel_index, 'qualified_ic_sp': qualified_ic_sp, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': dice_scores[best_channel_index]
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: N DATASET ITERATIONS (PERTURBATION & MEASUREMENT)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Iterations...")
    dataset_run_harmonic_scores = []
    
    all_runs_summary = {
        'reliability': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}}, 
        'rbo': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}}
    }
    
    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        run_ic_scores, run_oc_scores = [], []
        ssim_vs_noise_data = [] 

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0):
            batch, img_name = q_data['batch'], q_data['img_name']
            orig_image_01, gt_mask = q_data['orig_image_01'], q_data['gt_mask']
            segments_slic, pred_mask = q_data['segments_slic'], q_data['pred_mask']
            best_channel_index = q_data['best_channel_index']
            qualified_ic_sp, oc_sps_all = q_data['qualified_ic_sp'], q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, img_name.rsplit('.', 1)[0])
            os.makedirs(current_image_output_dir, exist_ok=True)
            final_analysis = {"image_info": {"name": img_name.rsplit('.', 1)[0], "initial_dice_vs_gt": round(q_data['initial_dice_vs_gt'], 2)}}

            if run_idx == 0:
                sanity_mask_dir = os.path.join(current_image_output_dir, "sanity_masks")
                os.makedirs(sanity_mask_dir, exist_ok=True)
                fig, axes = plt.subplots(1, len(q_data['all_pred_masks']), figsize=(6 * len(q_data['all_pred_masks']), 6))
                if len(q_data['all_pred_masks']) == 1: axes = [axes]
                for i, mask in enumerate(q_data['all_pred_masks']):
                    axes[i].imshow(mask, cmap='gray'); axes[i].axis('off')
                plt.tight_layout(); plt.savefig(os.path.join(sanity_mask_dir, "all_masks.png"), bbox_inches='tight', dpi=200); plt.close(fig)

            # --- 1. In-Context Analysis ---
            res_ic = run_ranking_analysis(
                'in_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, 
                qualified_ic_sp, qualified_ic_sp, noise_types, noise_char_map, ssim_vs_noise_data, 
                current_image_output_dir, pred_mask, best_channel_index, noise_step=args.noise_step, save_visuals=(run_idx == 0)
            )
            final_analysis['in_context_analysis'] = res_ic

            img_ic_scores = []
            
            for nt in noise_types: 
                all_runs_summary['rbo']['in_context'][nt].append(res_ic[nt]['average_rank_biased_overlap'])
                nt_rels = [r['base_reliability_score'] for r in res_ic[nt]['reliability_analysis']]
                
                if nt_rels: 
                    base_img_score = np.mean(nt_rels)
                    discount = res_ic[nt]['spillover_metrics']['global_spillover_discount']
                    discounted_img_score = base_img_score * discount
                    all_runs_summary['reliability']['in_context'][nt].append(discounted_img_score)
                    img_ic_scores.append(discounted_img_score)
            
            if img_ic_scores: run_ic_scores.append(np.mean(img_ic_scores))

            # --- 2. Out-of-Context Analysis ---
            num_oc_to_perturb = int(0.70 * len(oc_sps_all))
            out_of_context_sp_to_perturb = random.sample(oc_sps_all, num_oc_to_perturb)
            
            if run_idx == 0:
                in_context_map = np.zeros_like(segments_slic)
                for sp_id in qualified_ic_sp: in_context_map[segments_slic == sp_id] = sp_id
                out_of_context_map = np.zeros_like(segments_slic)
                for sp_id in oc_sps_all: out_of_context_map[segments_slic == sp_id] = sp_id
                eigencam_heatmap = calculate_eigencam(model_wrapper, args.target_layer_name, batch, orig_image_01.shape)
                visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, segments_slic, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_image_output_dir, img_name.rsplit('.', 1)[0] + "_overview.png"), prompt_points=batch['prompt_points'].squeeze(0).cpu().numpy())

            res_oc = run_ranking_analysis(
                'out_of_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, 
                out_of_context_sp_to_perturb, qualified_ic_sp, noise_types, noise_char_map, ssim_vs_noise_data, 
                current_image_output_dir, pred_mask, best_channel_index, noise_step=args.noise_step, save_visuals=(run_idx == 0)
            )
            final_analysis['out_of_context_analysis'] = res_oc
            
            img_oc_scores = []
            for nt in noise_types: 
                all_runs_summary['rbo']['out_of_context'][nt].append(res_oc[nt]['average_rank_biased_overlap'])
                nt_rels = [r['base_reliability_score'] for r in res_oc[nt]['reliability_analysis']]
                
                if nt_rels: 
                    base_img_score = np.mean(nt_rels)
                    discount = res_oc[nt]['spillover_metrics']['global_spillover_discount']
                    discounted_img_score = base_img_score * discount
                    all_runs_summary['reliability']['out_of_context'][nt].append(discounted_img_score)
                    img_oc_scores.append(discounted_img_score)
            
            if img_oc_scores: run_oc_scores.append(np.mean(img_oc_scores))

            json_save_path = os.path.join(current_image_output_dir, f"{img_name.rsplit('.', 1)[0]}_analysis.json")
            with open(json_save_path, 'w') as f: json.dump(final_analysis, f, indent=4, cls=NpEncoder)

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

        g = sns.catplot(
            data=df_rbo, x='Noise Type', y='RBO Score', col='Analysis Type',
            kind='violin', palette='viridis', sharey=True, height=6, aspect=1.2
        )
        g.fig.suptitle(f'RBO Distribution Across {args.num_dataset_runs} Runs', y=1.03, fontsize=16)
        g.set_axis_labels("Noise Type", "RBO Score")
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles("{col_name} Analysis")
        for ax in g.axes.flat: ax.set_ylim(0, 1)
        plt.savefig(os.path.join(args.output_dir, "RBO_Distributions_ViolinPlot.png"), bbox_inches='tight', dpi=300)
        plt.close(g.fig)

    for analysis_type, noise_data in all_runs_summary['reliability'].items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7))
            bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='mako')
            plt.title(f"Expected Reliability ({analysis_type.replace('_', ' ').title()}) over {args.num_dataset_runs} Runs", fontsize=16)
            for bar in bars.patches: 
                bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{analysis_type}_final_reliability_summary.png"), dpi=300)
            plt.close()

    plt.figure(figsize=(10, 6))
    runs_x = np.arange(1, args.num_dataset_runs + 1)
    
    plt.plot(runs_x, dataset_run_harmonic_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Run Robustness ($R_k$)')
    plt.axhline(R_bar, color='r', linestyle='--', linewidth=2, label=f'Expected Mean ($\overline{{R}}$) = {R_bar:.4f}')
    plt.fill_between(runs_x, R_bar - sigma_R, R_bar + sigma_R, color='r', alpha=0.2, label=f'$\pm 1\sigma_R$ ({sigma_R:.4f})')
    
    plt.title(f'Intra-Dataset Robustness Variance over {args.num_dataset_runs} Runs', fontsize=16, pad=15)
    plt.xlabel('Dataset Iteration (N)', fontsize=14)
    plt.ylabel('Dataset Robustness Score (Harmonic)', fontsize=14)
    
    plt.xticks(runs_x)
    safe_min = max(0, min(dataset_run_harmonic_scores) - (2 * sigma_R) - 0.05)
    safe_max = min(1, max(dataset_run_harmonic_scores) + (2 * sigma_R) + 0.05)
    plt.ylim(safe_min, safe_max)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "Run_Variance_Plot.png"), dpi=300)
    plt.close()
    
    logger.info(f"--- Pipeline Finished. Results and Plots are in {args.output_dir} ---")

if __name__ == "__main__":
    main()
import os
import argparse
import json
import logging
import random
import math
from math import exp
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'sam2' library.")
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
# --- 2. Dataset & Model Wrappers
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

class SAM2ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, batch: dict):
        """UPGRADED: Returns binary masks AND high-res sigmoid probabilities."""
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image_np_rgb = np.ascontiguousarray(image_np_rgb)
        
        H, W = image_np_rgb.shape[:2]
        
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
            masks, _, logits = self.model.predict(
                point_coords=points_in, 
                point_labels=labels_in, 
                multimask_output=True
            )
            
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            if logits_tensor.dim() == 3: logits_tensor = logits_tensor.unsqueeze(0)
                
            logits_high_res = torch.nn.functional.interpolate(
                logits_tensor, size=(H, W), mode="bilinear", align_corners=False
            )
            probs = torch.sigmoid(logits_high_res).squeeze(0).cpu().numpy()

        return masks.astype(np.uint8), probs

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> SAM2ModelWrapper:
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
        except Exception:
            sam_model = build_sam2(model_cfg_path, args.model_path, device=device)
        predictor = SAM2ImagePredictor(sam_model)
        return SAM2ModelWrapper(predictor, device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

# --------------------------------------------------------------------------
# --- 3. Raw Step Records (probability is used only to rank the IC superpixels)
# --------------------------------------------------------------------------
def ic_ranks(probs, segments, ic_sps):
    """Rank IC superpixels by mean probability with integer ranks 1..N (rank 1 = most confident).
    Probabilities are rounded to 3 decimals; ties go to the larger superpixel, then the smaller sp_id."""
    means = np.array([probs[segments == sp].mean() for sp in ic_sps])
    areas = np.array([np.sum(segments == sp) for sp in ic_sps])
    order = np.lexsort((np.asarray(ic_sps), -areas, -np.round(means, 3)))
    ranks = np.empty(len(ic_sps), dtype=int)
    ranks[order] = np.arange(1, len(ic_sps) + 1)
    return ranks, means

def rank_perplexity(ranks_k, ranks_0):
    """Normalised Spearman footrule: sum |r_k - r_0| / floor(N^2 / 2); 0 = unchanged ranking."""
    n = len(ranks_0)
    if n < 2:
        return 0.0
    return float(np.abs(np.asarray(ranks_k) - np.asarray(ranks_0)).sum() / (n * n // 2))

def bg_iou(pred_mask, gt_mask):
    """IoU (%) of the predicted background with the GT background."""
    bg_gt, bg_pred = (gt_mask == 0), (pred_mask == 0)
    union = np.sum(bg_gt | bg_pred)
    return float(np.sum(bg_gt & bg_pred) / union * 100.0) if union > 0 else 100.0

def clean_ranking_record(segments, ic_sps, clean_ranks, clean_means):
    """Clean (step 0) ranking of the IC superpixels, stored once per image."""
    return [{"sp_id": int(sp), "mean_prob": round(float(p), 3), "rank": int(r), "area": int(np.sum(segments == sp))}
            for sp, p, r in zip(ic_sps, clean_means, clean_ranks)]

def step_record(step_idx, pos_key, pos, ssim_score, pred_mask, probs, gt_mask, segments,
                ic_sps, clean_ranks, clean_bg_iou):
    """Raw quantities of one step; no metric is computed here.
    csr = drop of the background IoU against the clean prediction (0 if it does not drop)."""
    inter = np.sum((pred_mask == 1) & (gt_mask > 0))
    union = np.sum((pred_mask == 1) | (gt_mask > 0))
    ranks_k, means_k = ic_ranks(probs, segments, ic_sps)
    return {
        "step_idx": int(step_idx),
        pos_key: round(float(pos), 4),
        "ssim": round(float(ssim_score), 4),
        "global_iou": round(float(inter / union * 100.0) if union > 0 else 100.0, 2),
        "csr": round(max(0.0, clean_bg_iou - bg_iou(pred_mask, gt_mask)) / 100.0, 4),
        "rank_perplexity": round(rank_perplexity(ranks_k, clean_ranks), 4),
        "sp_probs": {str(int(sp)): round(float(p), 3) for sp, p in zip(ic_sps, means_k)},
    }

# --------------------------------------------------------------------------
# --- 4. Core Step-Wise Robustness Analysis (raw quantities per noise level)
# --------------------------------------------------------------------------
def run_ranking_analysis(
    analysis_type, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    superpixels_to_perturb, noise_types, current_image_output_dir,
    initial_pred_mask, initial_probs, best_channel_index,
    ic_sps_all, clean_ranks, noise_step=0.1, max_level=5.0, save_visuals=False
):
    analysis_results = {"level_data_per_noise": {}}
    to_tensor_transform = transforms.ToTensor()
    clean_bg_iou = bg_iou(initial_pred_mask, gt_mask)
    noise_levels_arr = np.round(np.arange(1, int(round(max_level / noise_step)) + 1) * noise_step, 4)

    for noise_type in noise_types:
        if save_visuals:
            noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
            os.makedirs(noise_specific_dir, exist_ok=True)

        # Step 0: clean image
        level_data_list = [step_record(0, "noise_level", 0.0, 1.0, initial_pred_mask, initial_probs, gt_mask,
                                       segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)]

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

            level_data_list.append(step_record(step_idx + 1, "noise_level", noise_level, ssim_score,
                                               perturbed_pred_mask, perturbed_probs, gt_mask, segments_slic,
                                               ic_sps_all, clean_ranks, clean_bg_iou))

        analysis_results["level_data_per_noise"][noise_type] = {"level_data": level_data_list}
    return analysis_results
# --------------------------------------------------------------------------
# --- 5. Main Inference Pipeline
# -------------------------------------------------------------------------- /SAMv2/sam2/sam2/configs/sam2.1
def main():
    parser = argparse.ArgumentParser(description="PRISM Global Continuous Noise Engine - SAM2")
    parser.add_argument('--data_dir', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_config', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/results/sam2/prism_out_global')
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

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0):
            batch, img_name = q_data['batch'], q_data['img_name']
            orig_image_01, gt_mask = q_data['orig_image_01'], q_data['gt_mask']
            segments_slic = q_data['segments_slic']
            pred_mask, initial_probs = q_data['pred_mask'], q_data['initial_probs']
            best_channel_index = q_data['best_channel_index']
            ic_sps_all, oc_sps_all = q_data['ic_sps_all'], q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, img_name.rsplit('.', 1)[0])
            os.makedirs(current_image_output_dir, exist_ok=True)
            
            # Clean ranking of the IC superpixels, computed once per image and used for every step
            clean_ranks, clean_means = ic_ranks(initial_probs, segments_slic, ic_sps_all)
            final_analysis = {
                "image_info": {
                    "name": img_name.rsplit('.', 1)[0],
                    "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)
                },
                "clean_ranking": clean_ranking_record(segments_slic, ic_sps_all, clean_ranks, clean_means)
            }

            # Extract ALL unique superpixels for the pure SOTA baseline
            all_sps = np.unique(segments_slic)
            all_sps = all_sps[all_sps != 0]

            # --- Global Analysis: perturb all superpixels, measure the IC superpixels ---
            res_global = run_ranking_analysis(
                'global_sota', model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
                all_sps, noise_types,
                current_image_output_dir, pred_mask, initial_probs, best_channel_index,
                ic_sps_all, clean_ranks, noise_step=args.noise_step, max_level=args.max_level, save_visuals=(run_idx == 0)
            )
            final_analysis['global_sota_analysis'] = res_global

            json_save_path = os.path.join(current_image_output_dir, f"{img_name.rsplit('.', 1)[0]}_analysis.json")
            with open(json_save_path, 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
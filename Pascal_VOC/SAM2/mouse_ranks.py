import os
import argparse
import json
import logging
import random
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
from scipy.ndimage import gaussian_filter
from scipy.ndimage import center_of_mass, label
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage.segmentation import mark_boundaries, slic, quickshift
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

# Import the specific, up-to-date functions for SAM-2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'sam2' library.")
    print("Please ensure you have cloned the 'sam2' repository from GitHub")
    print("and that its location is accessible to your Python environment.")
    exit()

# --------------------------------------------------------------------------
# --- 2. Scoring and Helper Functions (Mimicking the Target Script)
# --------------------------------------------------------------------------
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
    """Calculates Rank-Biased Overlap (RBO) between two ranked lists."""
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
        if union > 0 and (intersection / union) >= iou_threshold:
            selected_sp_ids.append(sp_id)
    return selected_sp_ids

def perturb_superpixels_with_level(original_image_scaled_01, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """Perturbs 3-channel image regions and returns the image and the noise characteristic value used."""
    perturbed_image = np.copy(original_image_scaled_01)
    noise_char_value = 0.0
    for sp_id in selected_superpixel_ids:
        sp_mask = (superpixels_segments == sp_id)
        # Create a 3-channel mask for RGB images
        sp_mask_3c = np.stack([sp_mask]*3, axis=-1)

        if noise_type == 'gaussian':
            noise_char_value = 0.1 * noise_level
            noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
            perturbed_image[sp_mask_3c] += noise[sp_mask_3c]
        elif noise_type == 'salt_pepper':
            noise_char_value = min(0.5, 0.05 * noise_level)
            num_pixels_in_sp = np.sum(sp_mask)
            num_noise = int(noise_char_value * num_pixels_in_sp)
            if num_noise > 0:
                for c in range(3):
                    rows, cols = np.where(sp_mask)
                    indices = random.sample(range(len(rows)), num_noise)
                    for i in indices: perturbed_image[rows[i], cols[i], c] = random.choice([0.0, 1.0])
        elif noise_type == 'poisson':
            noise_char_value = max(0.05, 3.2 - noise_level)
            region = perturbed_image[sp_mask_3c]
            # Reshape for applying noise and then reshape back
            original_shape = region.shape
            region_flat = region.flatten()
            noisy_region_flat = np.random.poisson(region_flat * 25 * noise_char_value) / (25 * noise_char_value)
            perturbed_image[sp_mask_3c] = noisy_region_flat.reshape(original_shape)
        elif noise_type == 'speckle':
            noise_char_value = 0.1 * noise_level
            noise = np.random.normal(0, noise_char_value, perturbed_image.shape)
            perturbed_image[sp_mask_3c] *= (1 + noise[sp_mask_3c])
        elif noise_type == 'gaussian_blur':
            noise_char_value = 0.5 * noise_level
            blurred_image = gaussian_filter(original_image_scaled_01, sigma=(noise_char_value, noise_char_value, 0))
            perturbed_image[sp_mask_3c] = blurred_image[sp_mask_3c]
        elif noise_type == 'rician':
            noise_char_value = 0.1 * noise_level
            region = perturbed_image[sp_mask_3c]
            n1 = np.random.normal(0, noise_char_value, region.shape)
            n2 = np.random.normal(0, noise_char_value, region.shape)
            noisy_region = np.sqrt(np.square(region + n1) + np.square(n2))
            perturbed_image[sp_mask_3c] = noisy_region

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

# --------------------------------------------------------------------------
# --- 3. Generic Data Loading and Model Wrappers (Unchanged)
# --------------------------------------------------------------------------

class GenericSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))

        self.image_size, self.transform, self.mask_transform = image_size, transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]), transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
    def __len__(self): return len(self.image_filenames)
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))).convert('L')
        mask_resized = self.mask_transform(mask).squeeze().numpy()

        # Find the center of mass of the largest connected component in the mask

        labeled_mask, num_features = label(mask_resized)
        if num_features > 0:
            # Find the largest component
            # Find the largest connected component (white region)
            sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
            largest_cc = np.argmax(sizes) + 1  # +1 because label 0 is background
            cc_mask = (labeled_mask == largest_cc)
            cy, cx = center_of_mass(cc_mask)

            # Ensure the point is at least 50 pixels inside from the boundary of the largest region
            coords = np.argwhere(cc_mask)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            safe_min_y, safe_max_y = min_y + 50, max_y - 50
            safe_min_x, safe_max_x = min_x + 50, max_x - 50

            # Clamp cy, cx to be at least 50 pixels from the boundary
            cy = np.clip(cy, safe_min_y, safe_max_y)
            cx = np.clip(cx, safe_min_x, safe_max_x)

            prompt_data = {'points': [[cy, cx]], 'labels': [1]}
        return {'image_tensor': self.transform(image), 'mask_tensor': self.mask_transform(mask), 'path': os.path.join(self.image_dir, img_name), 'prompt_points': torch.tensor(prompt_data['points'], dtype=torch.float32), 'prompt_labels': torch.tensor(prompt_data['labels'], dtype=torch.float32), 'img_name': img_name}

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
        image_np_bgr = np.ascontiguousarray(image_np_rgb[:, :, ::-1])
        points, labels = batch['prompt_points'], batch['prompt_labels']
        if isinstance(points, torch.Tensor) and points.dim() == 3 and points.shape[0] == 1: points = points.squeeze(0)
        if isinstance(labels, torch.Tensor) and labels.dim() == 2 and labels.shape[0] == 1: labels = labels.squeeze(0)
        
        points, labels = points.to(self.device), labels.to(self.device)
        points_in, labels_in = (points.unsqueeze(0) if points.dim() == 2 else points), (labels.unsqueeze(0) if labels.dim() == 1 else labels)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.set_image(image_np_bgr)
            masks, _, _ = self.model.predict(point_coords=points_in, point_labels=labels_in, multimask_output=True)

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
    """Factory function to build and wrap the specified model."""
    logger = logging.getLogger('base')
    logger.info(f"Loading model: {model_name} from config: {args.model_config}")
    if model_name.startswith('sam2_'):
        model_cfg_path = os.path.abspath(args.model_config)
        cfg_dir = os.path.dirname(model_cfg_path)
        cfg_name = os.path.basename(model_cfg_path)
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            from hydra import initialize_config_dir, compose
            initialize_config_dir(config_dir=cfg_dir)
            _ = compose(config_name=cfg_name)
            sam_model = build_sam2(cfg_name, args.model_path, device=device)
        except Exception as e:
            logger.warning(f"Hydra init/compose warning: {e} -- falling back to passing absolute path to build_sam2")
            sam_model = build_sam2(model_cfg_path, args.model_path, device=device)

        predictor = SAM2ImagePredictor(sam_model)
        return SAM2ModelWrapper(predictor, device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

# --------------------------------------------------------------------------
# --- 4. Core Analysis and Visualization (Updated)
# --------------------------------------------------------------------------

def calculate_eigencam(model_wrapper: BaseModelWrapper, target_layer_name: str, batch: dict, image_size: tuple) -> np.ndarray:
    """ This function is from the original script as it's compatible with the BaseModelWrapper. """
    feature_maps = []
    logger = logging.getLogger('base')
    try:
        target_layer = model_wrapper.get_target_layer(target_layer_name)
    except Exception as e:
        logger.error(f"Could not find target layer '{target_layer_name}'. Error: {e}")
        return np.zeros(image_size[:2], dtype=np.float32)
    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    _ = model_wrapper.predict(batch)
    hook.remove()
    if not feature_maps:
        logger.error("EigenCAM hook did not capture any feature maps."); return np.zeros(image_size[:2], dtype=np.float32)
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
    """Visualizes the original image, GT, prediction, and various superpixel maps, with prompt points overlayed."""
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image

    # Resize all images to 512x512
    display_image_rgb = (display_image_rgb * 255).astype(np.uint8)
    display_image_rgb = np.array(
        Image.fromarray(display_image_rgb).resize((512, 512))
    ) / 255.0

    gt_mask = gt_mask.astype(np.uint8)
    gt_mask = np.array(
        Image.fromarray(gt_mask).resize((512, 512), Image.NEAREST)
    )

    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = np.array(
        Image.fromarray(pred_mask).resize((512, 512), Image.NEAREST)
    )

    all_sp_map = all_sp_map.astype(np.uint8)
    all_sp_map = np.array(
        Image.fromarray(all_sp_map).resize((512, 512), Image.NEAREST)
    )

    in_context_map = in_context_map.astype(np.uint8)
    in_context_map = np.array(
        Image.fromarray(in_context_map).resize((512, 512), Image.NEAREST)
    )

    out_of_context_map = out_of_context_map.astype(np.uint8)
    out_of_context_map = np.array(
        Image.fromarray(out_of_context_map).resize((512, 512), Image.NEAREST)
    )

    eigencam_heatmap = (eigencam_heatmap * 255).astype(np.uint8)
    eigencam_heatmap = np.array(
        Image.fromarray(eigencam_heatmap).resize((512, 512))
    ) / 255.0

    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb)
    if prompt_points is not None:
        for point in prompt_points:
            y, x = point
            axes[0].scatter(x * (512 / orig_image.shape[1]), y * (512 / orig_image.shape[0]), c='red', s=200, marker='*', edgecolors='yellow', linewidths=1)
        axes[0].legend(loc='upper right')
    axes[0].set_title("Original Image with Prompts")
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

    axes[6].imshow(display_image_rgb)
    im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1)
    axes[6].set_title("EigenCAM")
    axes[6].axis('off')
    plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

### MODIFIED FUNCTION ###
def run_ranking_analysis(
    analysis_type, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    superpixels_to_perturb, superpixels_to_measure,
    noise_types, noise_char_map, ssim_data_collector,
    current_image_output_dir,
    initial_pred_mask, best_channel_index
):
    """
    Runs the full ranking analysis with the advanced reliability score calculation.
    """
    logger = logging.getLogger('base')
    logger.info(f"--- Starting Ranking Analysis: {analysis_type.upper()} ---")
    logger.info(f"Perturbing {len(superpixels_to_perturb)} SPs, measuring stability effect on {len(superpixels_to_measure)} SPs.")
    analysis_results = {}
    to_tensor_transform = transforms.ToTensor()
    
    # Initial ranking based on IoU between initial prediction and ground truth.
    initial_sp_iou_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(initial_pred_mask, gt_mask, segments_slic == sp_id), 2)} for sp_id in superpixels_to_measure]
    sorted_initial_sp_data = sorted(initial_sp_iou_data, key=lambda x: x['iou'], reverse=True)
    for rank, sp_data in enumerate(sorted_initial_sp_data, 1): sp_data['rank'] = rank
    initial_rank_list = [d['sp_id'] for d in sorted_initial_sp_data]
    initial_iou_map = {item['sp_id']: item['iou'] for item in sorted_initial_sp_data}

    for noise_type in noise_types:
        logger.info(f"--- Analyzing Noise Type: {noise_type.upper()} for {analysis_type} ---")
        
        noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
        os.makedirs(noise_specific_dir, exist_ok=True)
        
        level_0_entry = {noise_char_map.get(noise_type, 'noise_value'): 0.0, "ssim_score": 1.0, "image_path": "original", "superpixels": sorted_initial_sp_data}
        level_data_list = [level_0_entry]

        rbo_scores_list, sp_iou_history, sp_rank_history = [], {sp_id: [] for sp_id in superpixels_to_measure}, {sp_id: [] for sp_id in superpixels_to_measure}
        
        for noise_level in np.arange(0.05, 3.1, 0.05): 
            perturbed_image_01, noise_char_value = perturb_superpixels_with_level(orig_image_01, segments_slic, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_image_01, data_range=1.0, channel_axis=-1)
            
            if ssim_score < 0.85:
                logger.warning(f"SSIM score {ssim_score:.4f} dropped below 0.85 for '{noise_type}'. Stopping noise application.")
                break
            
            ssim_data_collector.append({"Analysis Type": analysis_type, "Noise Type": noise_type, "Noise Level": noise_level, "SSIM": ssim_score})

            perturbed_image_to_save = (perturbed_image_01 * 255).astype(np.uint8)
            pil_img = Image.fromarray(perturbed_image_to_save)
            img_save_path = os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png")
            pil_img.save(img_save_path)
            
            perturbed_batch = batch.copy()
            perturbed_batch['image_tensor'] = to_tensor_transform(perturbed_image_01).unsqueeze(0).float()
            
            all_perturbed_masks = model_wrapper.predict(perturbed_batch)
            perturbed_pred_mask = all_perturbed_masks[best_channel_index]

            current_level_sp_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(initial_pred_mask, perturbed_pred_mask, segments_slic == sp_id), 2)} for sp_id in superpixels_to_measure]
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            for sp_id in superpixels_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))
            
            current_rank_list = [d['sp_id'] for d in sorted_superpixels]
            rbo_scores_list.append(round(calculate_rbo(initial_rank_list, current_rank_list, p=0.9), 4))
            
            for rank, sp_data in enumerate(sorted_superpixels, 1):
                sp_data['rank'] = rank
            level_data_list.append({
                "noise_level": round(noise_level, 2),
                "image_path": os.path.relpath(img_save_path, os.path.dirname(current_image_output_dir)),
                "superpixels": sorted_superpixels
            })
        
        ### MODIFIED: Replaced original reliability calculation with new multi-step logic
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
        
        analysis_results[noise_type] = {
            "average_rank_biased_overlap": round(np.mean(rbo_scores_list) if rbo_scores_list else 0.0, 4),
            "rbo_at_each_level": rbo_scores_list,
            "reliability_analysis": sorted(reliability_results, key=lambda x: x['final_reliability_score'], reverse=True),
            "level_data": level_data_list
        }
    return analysis_results

def select_superpixels_by_gt_mask(superpixels_segments, gt_mask):
    """Selects all superpixels that have any overlap with the ground truth mask."""
    gt_mask_bool = gt_mask > 0
    overlapping_labels = np.unique(superpixels_segments[gt_mask_bool])
    return [label for label in overlapping_labels if label != 0]

# --- This is the updated main function ---
def main():
    parser = argparse.ArgumentParser(description="Advanced Segmentation Model Reliability Pipeline for SAM-2")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/pascal_sam_dataset_final', help='Path to the prepared dataset directory.')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large', help='Name of the model (e.g., sam2_hiera_large).')
    parser.add_argument('--model_path', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt', help='Path to the model checkpoint file (.pt).')
    parser.add_argument('--model_config', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml', help='Path to the model config file (.yaml).')
    parser.add_argument('--target_layer_name', type=str, default='sam_mask_decoder.output_upscaling.4', help='Dot-separated path to the model layer for EigenCAM.')
    # parser.add_argument('--output_dir', type=str, default='./expresults_quickshift')
    parser.add_argument('--output_dir', type=str, default='./expresults')
    parser.add_argument('--num_iterations', type=int, default=1, help="Number of images to process from the dataset.")
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help="Image size as height width.")
    parser.add_argument('--perturb_ratio', type=float, default=1.0, help="Fraction of in-context superpixels to perturb.")
    parser.add_argument('--num_out_of_context_perturb', type=int, default=40, help="Number of out-of-context superpixels to perturb.")
    parser.add_argument('--target_class', type=str, default='person', help="Subdirectory within data_dir to assess (e.g., 'all_classes').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "analysis.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model_wrapper = get_model_wrapper(args.model_name, args, device)
    val_loader = create_generic_dataloader(data_dir=args.data_dir+'/'+args.target_class, image_size=tuple(args.image_size))
   
    noise_types = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
    noise_char_map = {'salt_pepper': 'density', 'gaussian': 'std_dev', 'speckle': 'std_dev', 'poisson': 'intensity_scale', 'gaussian_blur': 'sigma', 'rician': 'sigma'}
    all_images_summary = {'reliability': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}}, 'rbo': {'in_context': {nt: [] for nt in noise_types}, 'out_of_context': {nt: [] for nt in noise_types}}}
    ssim_vs_noise_data = []

    processed_valid_count = 0
    for idata, batch in enumerate(tqdm(val_loader, desc="Finding and Processing Valid Images")):
        if processed_valid_count >= args.num_iterations:
            logger.info(f"Target of {args.num_iterations} valid images reached. Stopping.")
            break
        
        img_name = batch['img_name'][0]
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()

        # --- Perform all predictions and checks upfront ---
        all_pred_masks = model_wrapper.predict(batch)
        full_roi = np.ones_like(gt_mask, dtype=bool) 
        iou_scores = [calculate_iou_score(mask, gt_mask, full_roi) for mask in all_pred_masks]
        best_channel_index = np.argmax(iou_scores)

        # Qualification Check 1: Overall IoU must be > 40
        if iou_scores[best_channel_index] <= 40.0:
            logger.warning(f"Image '{img_name}' failed qualification: IoU score is {iou_scores[best_channel_index]:.2f} (<= 40.0). Skipping.")
            continue

        pred_mask = all_pred_masks[best_channel_index]

        segments_slic = slic(orig_image_01, n_segments=300, compactness=10, sigma=1, channel_axis=-1)

        # if orig_image_01.ndim == 2:  # Single-channel image
        #     img_quick = np.stack([orig_image_01] * 3, axis=-1)  # Create a 3-channel version for Quickshift
        # else:
        #     img_quick = orig_image_01
        # segments_slic = quickshift(img_quick, kernel_size=5, max_dist=10, ratio=0.7)
        
        in_context_sp_all = select_superpixels_by_gt_mask(segments_slic, gt_mask)
        qualified_in_context_sp = []
        if in_context_sp_all:
            for sp_id in in_context_sp_all:
                iou = calculate_iou_score(pred_mask, gt_mask, segments_slic == sp_id)
                if iou > 50:
                    qualified_in_context_sp.append(sp_id)
        
        # Qualification Check 2: Must have qualified superpixels
        if not qualified_in_context_sp:
            logger.warning(f"Image '{img_name}' failed qualification: No superpixels met the >50 IoU threshold. Skipping.")
            continue

        # --- Image is confirmed valid, proceed with full analysis and saving ---
        processed_valid_count += 1
        logger.info(f"--- Processing valid image {processed_valid_count}/{args.num_iterations}: {img_name} ---")
        current_image_output_dir = os.path.join(args.output_dir, img_name.rsplit('.', 1)[0])
        os.makedirs(current_image_output_dir, exist_ok=True)

        # Save sanity check masks for the valid image
        sanity_mask_dir = os.path.join(args.output_dir, f"{img_name.rsplit('.', 1)[0]}_sanity_masks")
        os.makedirs(sanity_mask_dir, exist_ok=True)
        fig, axes = plt.subplots(1, len(all_pred_masks), figsize=(6 * len(all_pred_masks), 6))
        if len(all_pred_masks) == 1: axes = [axes]
        for i, mask in enumerate(all_pred_masks):
            axes[i].imshow(mask, cmap='gray')
            axes[i].set_title(f"Mask {i} (Selected)" if i == best_channel_index else f"Mask {i}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sanity_mask_dir, "all_masks.png"), bbox_inches='tight', dpi=200)
        plt.close(fig)
        
        dice_scores = [calculate_dice_score(mask, gt_mask) for mask in all_pred_masks]
        logger.info(f"Initial mask selection: IoUs were {[round(s, 2) for s in iou_scores]}. Selected channel {best_channel_index} as best.")
        logger.info(f"{len(qualified_in_context_sp)} superpixels qualified with IoU > 50.")
            
        all_sp_ids = np.unique(segments_slic)
        out_of_context_ids = [sp_id for sp_id in all_sp_ids if sp_id not in in_context_sp_all and sp_id != 0]
        out_of_context_sp_to_perturb = random.sample(out_of_context_ids, min(args.num_out_of_context_perturb, len(out_of_context_ids)))
        
        in_context_map = get_refined_superpixel_map(gt_mask, segments_slic)
        out_of_context_map = np.zeros_like(segments_slic)
        for sp_id in out_of_context_ids: out_of_context_map[segments_slic == sp_id] = sp_id
        
        eigencam_heatmap = calculate_eigencam(model_wrapper, args.target_layer_name, batch, orig_image_01.shape)
        prompt_points = batch['prompt_points'].squeeze(0).cpu().numpy()

        visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, segments_slic,in_context_map, out_of_context_map, eigencam_heatmap,save_path=os.path.join(current_image_output_dir,  img_name.rsplit('.', 1)[0] + "_overview.png"),prompt_points=prompt_points)

        final_analysis = {"image_info": {"name": img_name.rsplit('.', 1)[0], "initial_dice_vs_gt": round(dice_scores[best_channel_index], 2)}}
        
        ### MODIFIED: Calls to run_ranking_analysis now use the qualified list
        if qualified_in_context_sp:
            final_analysis['in_context_analysis'] = run_ranking_analysis('in_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, qualified_in_context_sp, qualified_in_context_sp, noise_types, noise_char_map, ssim_vs_noise_data, current_image_output_dir, pred_mask, best_channel_index)
        
        if out_of_context_sp_to_perturb:
            final_analysis['out_of_context_analysis'] = run_ranking_analysis('out_of_context', model_wrapper, batch, orig_image_01, gt_mask, segments_slic, out_of_context_sp_to_perturb, qualified_in_context_sp, noise_types, noise_char_map, ssim_vs_noise_data, current_image_output_dir, pred_mask, best_channel_index)
        
        json_save_path = os.path.join(current_image_output_dir, f"{img_name.rsplit('.', 1)[0]}_analysis.json")
        with open(json_save_path, 'w') as f:
            json.dump(final_analysis, f, indent=4, cls=NpEncoder)
        logger.info(f"Full analysis saved to {json_save_path}")

        ### MODIFIED: Key changed to 'final_reliability_score'
        for at in ['in_context', 'out_of_context']:
            if f'{at}_analysis' in final_analysis:
                for nt, res in final_analysis[f'{at}_analysis'].items():
                    all_images_summary['rbo'][at][nt].append(res['average_rank_biased_overlap'])
                    scores = [r['final_reliability_score'] for r in res.get('reliability_analysis', [])]
                    if scores: all_images_summary['reliability'][at][nt].append(np.mean(scores))

    logger.info("--- Generating Final Summary Plots ---")
    rbo_plot_data = [{'Analysis Type': at.replace('_', ' ').title(), 'Noise Type': nt, 'RBO Score': s} for at, nd in all_images_summary['rbo'].items() for nt, sc in nd.items() for s in sc]

    if rbo_plot_data:
        df_rbo = pd.DataFrame(rbo_plot_data)
        min_rbo = df_rbo['RBO Score'].min()
        max_rbo = df_rbo['RBO Score'].max()
        # Apply Min-Max Normalization
        if max_rbo > min_rbo:  # Avoid division by zero
            df_rbo['RBO Score'] = (df_rbo['RBO Score'] - min_rbo) / (max_rbo - min_rbo)
        else:
            df_rbo['RBO Score'] = 0.0  # If all values are the same, set them to 0

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

    for analysis_type, noise_data in all_images_summary['reliability'].items():
        plot_data = [{'Noise Type': nt, 'Average Reliability Score': np.mean(sc)} for nt, sc in noise_data.items() if sc]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 7)); bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='mako')
            plt.title(f"Final Reliability Score ({analysis_type.replace('_', ' ').title()})", fontsize=16)
            for bar in bars.patches: bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, f"{analysis_type}_final_reliability_summary.png"), dpi=300); plt.close()

    logger.info(f"--- Pipeline Finished. Results are in {args.output_dir} ---")

if __name__ == "__main__":
    main()
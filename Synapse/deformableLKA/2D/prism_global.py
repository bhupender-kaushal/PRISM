import os
import argparse
import json
import logging
import random
import math
import gc
from abc import ABC, abstractmethod
from math import exp
import sys

import numpy as np
import pandas as pd
from PIL import Image
import h5py

from scipy.ndimage import gaussian_filter, zoom
from scipy.ndimage import center_of_mass, label
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage import color, io
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.util import regular_grid
from sklearn.decomposition import PCA
import scipy.stats as stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn.functional as F
# Directly import the model from your project structure
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer

random_seed = 42
random.seed(random_seed)
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

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
        data = h5py.File(filepath, 'r')
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

# --- GRAYSCALE SAFE FIX: Noise Application ---
# Noise strength per unit noise level, tuned for Synapse. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'blur_sigma': 10.0,            # sigma = 10 * level
    'rician_sigma': 0.05,          # std = 0.05 * level
}

def perturb_image_slice(image_01, segments, sp_to_perturb, noise_type, noise_level):
    """
    Applies mathematically strict 2D perturbations (Grayscale safe). 
    No channel loops are used.
    """
    perturbed_image = np.copy(image_01)
    
    region_mask = np.isin(segments, sp_to_perturb)
    if not np.any(region_mask):
        return perturbed_image
        
    if noise_type == 'gaussian':
        noise = np.random.normal(0, NOISE_PARAMS['gaussian_sigma'] * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] += noise[region_mask]
        
    elif noise_type == 's&p' or noise_type == 'salt_pepper':
        noise_char_value = min(0.5, NOISE_PARAMS['salt_pepper_density'] * noise_level)
        num_pixels = np.sum(region_mask)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            rows, cols = np.where(region_mask)
            indices = np.random.choice(len(rows), num_noise, replace=False)
            perturbed_image[rows[indices], cols[indices]] = np.random.choice([0.0, 1.0], num_noise)
            
    elif noise_type == 'poisson':
        target_std = NOISE_PARAMS['gaussian_sigma'] * noise_level          # Poisson std on mid-grey matches the Gaussian std
        photons = 0.5 / max(target_std, 1e-6) ** 2
        region = perturbed_image[region_mask]
        noisy_region = np.random.poisson(np.clip(region, 0, 1) * photons) / photons
        perturbed_image[region_mask] = noisy_region
        
    elif noise_type == 'speckle':
        noise = np.random.normal(0, NOISE_PARAMS['speckle_sigma'] * noise_level, perturbed_image.shape)
        perturbed_image[region_mask] *= (1.0 + noise[region_mask])
        
    elif noise_type == 'gaussian_blur':
        blurred_image = gaussian_filter(image_01, sigma=NOISE_PARAMS['blur_sigma'] * noise_level)
        perturbed_image[region_mask] = blurred_image[region_mask]
        
    elif noise_type == 'rician':
        n1 = np.random.normal(0, NOISE_PARAMS['rician_sigma'] * noise_level, perturbed_image.shape)
        n2 = np.random.normal(0, NOISE_PARAMS['rician_sigma'] * noise_level, perturbed_image.shape)
        noisy_region = np.sqrt(np.square(perturbed_image + n1) + np.square(n2))
        perturbed_image[region_mask] = noisy_region[region_mask]

    return np.clip(perturbed_image, 0.0, 1.0)

def predict_slice_maxvit(model, image_slice_01, patch_size, target_class_index, device='cuda'):
    """MaxViT-specific inference routing extracting multiclass target probabilities."""
    h, w = image_slice_01.shape
    slice_rescaled = zoom(image_slice_01, (patch_size / h, patch_size / w), order=3) if (h != patch_size or w != patch_size) else image_slice_01
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = x_transforms(slice_rescaled).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs_tensor = torch.softmax(outputs, dim=1)
        
        out_mask = torch.argmax(probs_tensor, dim=1).squeeze(0).cpu().numpy()
        target_probs = probs_tensor[0, target_class_index, :, :].cpu().numpy()
        
    if h != patch_size or w != patch_size:
        out_mask = zoom(out_mask, (h / patch_size, w / patch_size), order=0)
        target_probs = zoom(target_probs, (h / patch_size, w / patch_size), order=1)
        
    pred_mask_bool = (out_mask == target_class_index)
    return pred_mask_bool, target_probs, input_tensor

def get_target_layer(model, layer_name):
    try:
        return model.get_submodule(layer_name)
    except AttributeError:
        for name, module in model.named_modules():
            if layer_name in name:
                return module
        raise AttributeError(f"Could not find target layer '{layer_name}'.")

def calculate_eigencam(model, input_tensor, target_layer_names, output_size):
    if not isinstance(target_layer_names, list):
        target_layer_names = [target_layer_names]
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

def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    display_image_rgb = gray2rgb(orig_image) if orig_image.ndim == 2 else orig_image
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    axes[0].imshow(display_image_rgb, cmap='gray'); axes[0].set_title("Original Image"); axes[0].axis('off')
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
    analysis_type, model, device, orig_image_01, segments, target_class_gt,
    superpixels_to_perturb, initial_pred_mask_bool,
    noise_types, patch_size, target_class_index, current_image_output_dir,
    initial_probs, ic_sps_all, clean_ranks,
    noise_step=0.1, max_level=5.0, save_visuals=False
):
    analysis_results = {"level_data_per_noise": {}}
    clean_bg_iou = bg_iou(initial_pred_mask_bool, target_class_gt)

    for noise_type in noise_types:
        if save_visuals:
            noise_specific_dir = os.path.join(current_image_output_dir, f"{analysis_type}_{noise_type}")
            os.makedirs(noise_specific_dir, exist_ok=True)

        # Step 0: clean image
        level_data_list = [step_record(0, "noise_level", 0.0, 1.0, initial_pred_mask_bool, initial_probs, target_class_gt,
                                       segments, ic_sps_all, clean_ranks, clean_bg_iou)]

        noise_levels_arr = np.round(np.arange(1, int(round(max_level / noise_step)) + 1) * noise_step, 4)
        pbar = tqdm(noise_levels_arr, desc=f"   -> [{analysis_type[:2].upper()}] {noise_type}", leave=False, position=1)
        
        for step_idx, noise_level in enumerate(pbar): 
            pbar.set_postfix({"Level": f"{noise_level:.2f}"})
            
            perturbed_img_gray = perturb_image_slice(orig_image_01, segments, superpixels_to_perturb, noise_type, noise_level)
            ssim_score = structural_similarity(orig_image_01, perturbed_img_gray, data_range=1.0)
            

            perturbed_pred_mask_bool, perturbed_probs, _ = predict_slice_maxvit(model, perturbed_img_gray, patch_size, target_class_index, device)

            if save_visuals and int(round(noise_level, 2) * 100) % 50 == 0:
                Image.fromarray((perturbed_img_gray * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}.png"))
                mask_to_save = (perturbed_pred_mask_bool * 255).astype(np.uint8)
                Image.fromarray(mask_to_save).resize((256, 256), Image.NEAREST).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_mask.png"))
                img_rgb = gray2rgb(perturbed_img_gray)
                overlay_image = mark_boundaries(img_rgb, perturbed_pred_mask_bool, color=(1, 0, 0), mode='thick')
                overlay_to_save = (overlay_image * 255).astype(np.uint8)
                Image.fromarray(overlay_to_save).resize((256, 256), Image.BILINEAR).save(os.path.join(noise_specific_dir, f"level_{noise_level:.2f}_ssim_{ssim_score:.4f}_overlay.png"))

            level_data_list.append(step_record(step_idx + 1, "noise_level", noise_level, ssim_score,
                                               perturbed_pred_mask_bool, perturbed_probs, target_class_gt, segments,
                                               ic_sps_all, clean_ranks, clean_bg_iou))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        analysis_results["level_data_per_noise"][noise_type] = {"level_data": level_data_list}
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
# 4. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler(os.path.join(args.output_dir, "dataset_inference.log")), logging.StreamHandler()]
    )
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Loading MaxViT model...")
    net = MaxViT_deformableLKAFormer(num_classes=args.num_classes).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    logger.info("Loading Synapse dataset...")
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    candidate_samples = find_samples_with_class(db_test, args.target_class_index)
    if not candidate_samples: sys.exit()
    random.shuffle(candidate_samples)

    cohort_file = args.img_cohort
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    for sample_info in tqdm(candidate_samples, desc="Qualifying Slices"):
        if len(qualified_images) >= args.num_images: break

        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        volume_data = db_test[vol_idx]
        image_slice, label_slice = volume_data['image'][slice_idx, :, :], volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"
        
        if case_name not in frozen_cohort:
            continue
        
        image_slice_01 = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        
        # Initial forward pass extracting specific probabilities
        initial_pred_mask_bool, initial_probs, input_tensor = predict_slice_maxvit(net, image_slice_01, args.img_size, args.target_class_index, device)
        target_class_gt = (label_slice == args.target_class_index).astype(np.uint8)
        
        segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        # Absolute Anchoring (No prediction threshold filters used)
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=1.0, oc_threshold=0.0)
        
        if not ic_sps_all: continue

        qualified_images.append({
            'case_name': case_name, 'image_slice_01': image_slice_01, 'input_tensor': input_tensor,
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'initial_probs': initial_probs,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool, target_class_gt)
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: MULTI-RUN ROBUSTNESS INFERENCE (GLOBAL)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Commencing {args.num_dataset_runs} Dataset Iterations...")

    for run_idx in range(args.num_dataset_runs):
        logger.info(f"\n==================== RUN {run_idx + 1}/{args.num_dataset_runs} ====================")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_idx+1}")
        os.makedirs(run_output_dir, exist_ok=True)

        for q_data in tqdm(qualified_images, desc=f"Processing Run {run_idx+1}", position=0, leave=True):
            case_name = q_data['case_name']
            image_slice_01 = q_data['image_slice_01']
            input_tensor = q_data['input_tensor']
            target_class_gt = q_data['target_class_gt']
            segments = q_data['segments']
            initial_pred_mask_bool = q_data['initial_pred_mask_bool']
            initial_probs = q_data['initial_probs']
            ic_sps_all = q_data['ic_sps_all']
            oc_sps_all = q_data['oc_sps_all']

            current_image_output_dir = os.path.join(run_output_dir, case_name)
            os.makedirs(current_image_output_dir, exist_ok=True)
            
            # Clean ranking of the IC superpixels, computed once per image and used for every step
            clean_ranks, clean_means = ic_ranks(initial_probs, segments, ic_sps_all)
            final_analysis = {
                "image_info": {
                    "name": case_name,
                    "initial_dice_vs_gt": round(float(q_data['initial_dice_vs_gt']), 2)
                },
                "clean_ranking": clean_ranking_record(segments, ic_sps_all, clean_ranks, clean_means)
            }


            # Extract ALL unique superpixels for the pure SOTA baseline
            all_sps = np.unique(segments)
            all_sps = all_sps[all_sps != 0]

            # --- Global Analysis: perturb all superpixels, measure the IC superpixels ---
            res_global = run_ranking_analysis(
                analysis_type='global_sota', 
                model=net, 
                device=device, 
                orig_image_01=image_slice_01, 
                segments=segments, 
                target_class_gt=target_class_gt, 
                superpixels_to_perturb=all_sps,
                initial_pred_mask_bool=initial_pred_mask_bool,
                noise_types=args.noise_types, 
                patch_size=args.img_size, 
                target_class_index=args.target_class_index, 
                current_image_output_dir=current_image_output_dir, 
                initial_probs=initial_probs, 
                ic_sps_all=ic_sps_all,
                clean_ranks=clean_ranks,
                noise_step=args.noise_step, max_level=args.max_level, 
                save_visuals=(run_idx == 0)
            )
            final_analysis['global_sota_analysis'] = res_global

            with open(os.path.join(current_image_output_dir, f"{case_name}_analysis.json"), 'w') as f: 
                json.dump(final_analysis, f, indent=4, cls=NpEncoder)

    logger.info(f"\n[COMPLETE] All {args.num_dataset_runs} runs processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxViT Model Reliability Inference + Rank Ablation")
    parser.add_argument("--volume_path", type=str, default="/cluster/home/bhupenk/bhupender/datasets/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--model_path", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/deformableLKA/2D/MaxViT_deform_LKA.pth")
    parser.add_argument("--output_dir", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/results/dlka/prism_out_global")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int,default=8)
    parser.add_argument("--img_cohort", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/img_sel.txt", help="pre compiled image list.")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_dataset_runs", type=int, default=1, help="Number of dataset iterations for variance calculation.")
    parser.add_argument("--noise_step", type=float, default=0.1, help="Step size for noise loop.")
    parser.add_argument("--max_level", type=float, default=5.0, help="Highest noise level of the grid (inclusive).")
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument("--noise_types", nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    parser.add_argument('--target_layer_names', nargs='+', default=['last_layer'])

    args = parser.parse_args()
    main(args)
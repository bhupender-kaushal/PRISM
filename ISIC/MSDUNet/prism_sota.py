import os, sys
import json
import argparse
import logging
import random
import time
import itertools
import math
import gc
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
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr
from skimage.color import gray2rgb, rgb2gray

from models.MSDUNet import MSDUFormer_Xnet_3463_FUESMS as model
from utils.dataloader import test_dataset

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

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    """Absolute Anatomical Anchoring: Ground Truth explicitly defines domains."""
    if superpixels_segments.shape != gt_mask.shape:
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

# --- RGB CHANNEL-WISE FIX: Noise Application ---
# Noise strength per unit noise level, tuned for ISIC. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'poisson_photons': 500.0,      # photons = 500 * max(0.05, 3.2 - level)
    'blur_sigma': 50.0,            # sigma = 50 * level
    'rician_sigma': 0.1,           # std = 0.1 * level
}

def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """
    Applies strict channel-wise perturbations over RGB channels for the selected superpixels.
    """
    perturbed_image = np.copy(original_image_scaled)
    noise_char_value = 0.0

    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask):
        return perturbed_image, noise_char_value

    H, W, C = perturbed_image.shape

    if noise_type == 'gaussian':
        noise_char_value = NOISE_PARAMS['gaussian_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] += noise[region_mask]
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'salt_pepper':
        noise_char_value = min(0.5, NOISE_PARAMS['salt_pepper_density'] * noise_level)
        rows, cols = np.where(region_mask)
        num_pixels = len(rows)
        num_noise = int(noise_char_value * num_pixels)
        if num_noise > 0:
            for c in range(C):
                indices = np.random.choice(num_pixels, num_noise, replace=False)
                salt_pepper_vals = np.random.choice([0.0, 1.0], num_noise)
                perturbed_image[rows[indices], cols[indices], c] = salt_pepper_vals

    elif noise_type == 'poisson':
        intensity_scale = max(0.05, 3.2 - noise_level)
        scale_factor = NOISE_PARAMS['poisson_photons'] * intensity_scale
        noise_char_value = intensity_scale
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = np.clip(channel_data[region_mask], 0.0, 1.0)
            noisy_pixels = np.random.poisson(region_pixels * scale_factor) / scale_factor
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'speckle':
        noise_char_value = NOISE_PARAMS['speckle_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            noise = np.random.normal(0, noise_char_value, size=(H, W))
            channel_data[region_mask] *= (1.0 + noise[region_mask])
            perturbed_image[:, :, c] = channel_data
        
    elif noise_type == 'gaussian_blur':
        noise_char_value = NOISE_PARAMS['blur_sigma'] * noise_level
        for c in range(C):
            blurred_channel = gaussian_filter(original_image_scaled[:, :, c], sigma=noise_char_value)
            perturbed_image[region_mask, c] = blurred_channel[region_mask]
        
    elif noise_type == 'rician':
        noise_char_value = NOISE_PARAMS['rician_sigma'] * noise_level
        for c in range(C):
            channel_data = perturbed_image[:, :, c]
            region_pixels = channel_data[region_mask]
            n1 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            n2 = np.random.normal(0, noise_char_value, size=region_pixels.shape)
            noisy_pixels = np.sqrt(np.square(region_pixels + n1) + np.square(n2))
            channel_data[region_mask] = noisy_pixels
            perturbed_image[:, :, c] = channel_data

    return np.clip(perturbed_image, 0.0, 1.0), noise_char_value

# --- Image Downsampling and Saving Helper (same as Pascal VOC) ---
def save_downsampled_image(arr, path, is_mask=False, size=(256, 256)):
    """Saves a numpy array as an image, forced to 256x256 resolution to save disk space."""
    if is_mask:
        img = Image.fromarray((np.asarray(arr) * 255).astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode='RGB' if arr.ndim == 3 else 'L')
    
    # Use Nearest Neighbor for binary masks, Bilinear for images
    img = img.resize(size, Image.NEAREST if is_mask else Image.BILINEAR)
    img.save(path)

def generate_baseline_canvas(orig_image_01, baseline_type):
    """Generates the Null-Information canvas (Black, Mean, or Max-Noise) for I/D."""
    if baseline_type == 'black':
        return np.zeros_like(orig_image_01)
    elif baseline_type == 'mean':
        mean_color = np.mean(orig_image_01, axis=(0, 1), keepdims=True)
        return np.ones_like(orig_image_01) * mean_color
    else:
        # Generate the fully saturated noisy background for the 6 noise variants
        dummy_all_mask = np.ones(orig_image_01.shape[:2], dtype=int)
        return perturb_superpixels_with_level(orig_image_01, dummy_all_mask, [1], baseline_type, noise_level=30.0)[0]

def predict_from_numpy(model_instance, img_np_01, device, gt_shape):
    """Helper to maintain precise ImageNet normalization during numpy array insertion/deletion steps."""
    tensor = torch.from_numpy(img_np_01).permute(2, 0, 1).float()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = normalize(tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        res = model_instance(input_tensor)
        res = F.upsample(res, size=gt_shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred_mask = np.where(probs >= 0.5, 1, 0)
    return pred_mask, probs

# --- Visualizations ---
def calculate_eigencam(model, target_layer_name, image_tensor, image_size):
    feature_maps = []
    try: target_layer = dict(model.named_modules())[target_layer_name]
    except KeyError: return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(image_tensor)
    hook.remove() 

    if not feature_maps: return np.zeros(image_size, dtype=np.float32)

    features = feature_maps[0].squeeze(0)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    
    n_samples, n_features = features_flat.shape
    n_components = min(1, n_samples, n_features)
    if n_components == 0: return np.zeros(image_size, dtype=np.float32)

    pca = PCA(n_components=n_components).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
    saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
    return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()

# --------------------------------------------------------------------------
# --- 2. Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_instance, device, orig_image_np_01, gt_mask_np, segments_slic,
    sps_to_perturb, sps_to_measure, initial_pred_mask, initial_probs,
    ic_sps_all, oc_sps_all, baselines, save_dir
):
    """Executes Insertion/Deletion ranked by probability, one superpixel per step, saving step-wise superpixel metrics on GT."""
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    gt_shape = gt_mask_np.shape
    
    if len(sps_to_perturb) == 0 or len(sps_to_measure) == 0: return analysis_results
    
    max_id = int(np.max(segments_slic)) + 1

    # --- Save Initial States and SP Map (same as Pascal VOC) ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)
    save_downsampled_image(mark_boundaries(orig_image_np_01, segments_slic), os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_np_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))
    sp_areas = {int(sp_id): int(np.sum(segments_slic == sp_id)) for sp_id in np.unique(segments_slic)}
    
    # 1. Map Areas, Domains, Probabilities, and EigenCAM Intensities
    rank_data = []
    for sp_id in sps_to_perturb:
        mask = (segments_slic == sp_id)
        area = sp_areas[int(sp_id)]
        mean_p = float(np.mean(initial_probs[mask]))
        
        if sp_id in ic_sps_all: gt_domain = "IC"
        elif sp_id in oc_sps_all: gt_domain = "OC"
        else: gt_domain = "Mixed"

        rank_data.append({
            "sp_id": int(sp_id), 
            "area": area, 
            "mean_prob": mean_p, 
            "gt_domain": gt_domain
        })
        
    # 2. Assign explicit Rank Integers
    rank_data.sort(key=lambda x: (x['mean_prob'], x['area']), reverse=True)
    for rank, item in enumerate(rank_data, 1):
        item['prob_rank'] = rank
    prob_ranked_ids = [item['sp_id'] for item in rank_data]
        
    analysis_results["rank_distribution"] = rank_data
    ranking_methods = {'prob_ranked': prob_ranked_ids}
    
    # 3. Dynamic Steps & Helpers
    N_perturb = len(sps_to_perturb)
    k_values = list(range(1, N_perturb + 1))  # one superpixel per step
    bg_gt = (gt_mask_np == 0)

    # Function to get individual GT superpixel tracking at each step
    def get_step_superpixels(current_pred_mask, current_probs):
        intersect_mask = (current_pred_mask == 1) & (gt_mask_np > 0)
        union_mask = (current_pred_mask == 1) | (gt_mask_np > 0)
        
        intersect_counts = np.bincount(segments_slic[intersect_mask].ravel(), minlength=max_id)
        union_counts = np.bincount(segments_slic[union_mask].ravel(), minlength=max_id)
        
        step_sps = []
        for sp_id in sps_to_measure:
            u_val = union_counts[sp_id]
            i_val = intersect_counts[sp_id]
            iou_val = (i_val / u_val * 100.0) if u_val > 0 else 100.0
            mask = (segments_slic == sp_id)
            mean_p = float(np.mean(current_probs[mask]))
            
            step_sps.append({
                "sp_id": int(sp_id),
                "iou": round(float(iou_val), 2),
                "mean_prob": round(mean_p, 4),
                "area": sp_areas[int(sp_id)]
            })
            
        step_sps.sort(key=lambda x: (x['mean_prob'], x['area']), reverse=True)
        for rank, item in enumerate(step_sps, 1):
            item['step_prob_rank'] = rank
        return step_sps

    base_sps = get_step_superpixels(initial_pred_mask, initial_probs)
    initial_ranks_map = {item['sp_id']: item['step_prob_rank'] for item in base_sps}
    max_possible_jumps = math.floor((len(sps_to_measure)**2) / 2.0) if len(sps_to_measure) > 1 else 1.0

    def calc_perplexity(step_sps):
        current_ranks_map = {item['sp_id']: item['step_prob_rank'] for item in step_sps}
        total_jumps = sum(abs(current_ranks_map[sp_id] - initial_ranks_map[sp_id]) for sp_id in sps_to_measure)
        return round(total_jumps / max_possible_jumps, 4) if max_possible_jumps > 0 else 0.0

    def create_entry(frac, global_iou, spill, mean_f_prob, step_sps): 
        return {
            "fraction_perturbed": frac, 
            "global_iou": round(float(global_iou), 2), 
            "spillover_ratio": round(float(spill), 4), 
            "mean_foreground_prob": round(float(mean_f_prob), 4),
            "rank_perplexity": calc_perplexity(step_sps),
            "superpixels": step_sps
        }

    def calculate_auc(traj):
        if len(traj) < 2: return 0.0
        x = [e['fraction_perturbed'] for e in traj]
        y = [e['global_iou'] / 100.0 for e in traj]
        return float(auc(x, y))

    # --- EXECUTE TRAJECTORIES OVER THE 8 BASELINES ---
    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_np_01, baseline)
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        
        # Base Deletion Metrics
        base_intersect = np.sum((initial_pred_mask == 1) & (gt_mask_np > 0))
        base_union = np.sum((initial_pred_mask == 1) | (gt_mask_np > 0))
        base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0
        base_f_prob = float(np.mean(initial_probs[initial_pred_mask == 1])) if np.any(initial_pred_mask == 1) else 0.0

        # Base Insertion Metrics (Fully destroyed image)
        pred_n, prob_n = predict_from_numpy(model_instance, bg_canvas, device, gt_shape)
        
        ins_union0 = np.sum((pred_n == 1) | (gt_mask_np > 0))
        ins_base_iou = (np.sum((pred_n == 1) & (gt_mask_np > 0)) / ins_union0 * 100.0) if ins_union0 > 0 else 100.0
        ins_base_prob = float(np.mean(prob_n[pred_n == 1])) if np.any(pred_n == 1) else 0.0
        ins_base_sps = get_step_superpixels(pred_n, prob_n)

        for r_name, r_ids in ranking_methods.items():
            traj = {"deletion": [], "insertion": []}
            traj["deletion"].append(create_entry(0.0, base_global_iou, 0.0, base_f_prob, base_sps))
            traj["insertion"].append(create_entry(0.0, ins_base_iou, 0.0, ins_base_prob, ins_base_sps))
            
            baseline_vis_dir = os.path.join(vis_dir_root, baseline, r_name)
            os.makedirs(baseline_vis_dir, exist_ok=True)
            
            pbar = tqdm(k_values, desc=f"   -> [{domain_name[:2].upper()}] {baseline.upper()} | {r_name}", leave=False)
            for k in pbar:
                fraction = round(k / N_perturb, 4)
                mask_2d = np.isin(segments_slic, r_ids[:k])
                mask_3d = mask_2d[:, :, None]

                # --- 1. Deletion Step ---
                img_del = np.where(mask_3d, bg_canvas, orig_image_np_01)
                pred_del, prob_del = predict_from_numpy(model_instance, img_del, device, gt_shape)
                
                iou_del = (np.sum((pred_del == 1) & (gt_mask_np > 0)) / max(1, np.sum((pred_del == 1) | (gt_mask_np > 0)))) * 100.0
                spill_del = np.sum((pred_del == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_del = float(np.mean(prob_del[pred_del == 1])) if np.any(pred_del == 1) else 0.0
                del_sps = get_step_superpixels(pred_del, prob_del)
                traj["deletion"].append(create_entry(fraction, iou_del, spill_del, prob_f_del, del_sps))

                # --- 2. Insertion Step ---
                img_ins = np.where(mask_3d, orig_image_np_01, bg_canvas)
                pred_ins, prob_ins = predict_from_numpy(model_instance, img_ins, device, gt_shape)
                
                iou_ins = (np.sum((pred_ins == 1) & (gt_mask_np > 0)) / max(1, np.sum((pred_ins == 1) | (gt_mask_np > 0)))) * 100.0
                spill_ins = np.sum((pred_ins == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_ins = float(np.mean(prob_ins[pred_ins == 1])) if np.any(pred_ins == 1) else 0.0
                ins_sps = get_step_superpixels(pred_ins, prob_ins)
                traj["insertion"].append(create_entry(fraction, iou_ins, spill_ins, prob_f_ins, ins_sps))

                # --- Save the perturbed inputs and predicted masks (same as Pascal VOC) ---
                step_vis_dir = os.path.join(baseline_vis_dir, f"step_{k:03d}")
                os.makedirs(step_vis_dir, exist_ok=True)
                
                save_downsampled_image(img_del, os.path.join(step_vis_dir, "deletion_input.jpg"))
                save_downsampled_image(pred_del, os.path.join(step_vis_dir, "deletion_pred.png"), is_mask=True)
                
                save_downsampled_image(img_ins, os.path.join(step_vis_dir, "insertion_input.jpg"))
                save_downsampled_image(pred_ins, os.path.join(step_vis_dir, "insertion_pred.png"), is_mask=True)

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            traj["auc_del"] = calculate_auc(traj["deletion"])
            traj["auc_ins"] = calculate_auc(traj["insertion"])
            analysis_results["trajectories"][baseline][r_name] = traj

    return analysis_results

# --------------------------------------------------------------------------
# --- 3. Main Controller
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (MSDUNet)")
    parser.add_argument('--pth_path', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/MSDUNet/ISIC2016_MSDUNet.pth')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/cluster/home/bhupenk/bhupender/datasets/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--target_layer_name', type=str, default='FU_E.conv1')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/results/msdunet/prism_out_insertion_deletion')
    parser.add_argument('--txtfile', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/img_sel.txt')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--superpixel_size', type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading MSDUNet model...")
    model_instance = model(num_classes=args.num_classes).to(device)
    model_instance.load_state_dict(torch.load(args.pth_path, map_location=device))
    model_instance.eval()

    test_loader = test_dataset(os.path.join(args.dataroot_val, 'images/'), os.path.join(args.dataroot_val, 'masks/'), args.testsize)

    # 6+2 Baselines
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS & HEATMAP GENERATION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Strict Anatomical Qualifier Scan...")
    qualified_images = []
    
    with open(args.txtfile, 'r') as f: frozen_image_names = set([line.strip() for line in f.readlines()])
    
    for idx in tqdm(range(len(test_loader.images)), desc="Qualifying Slices"):
        if len(qualified_images) >= args.num_images: break
        
        image, gt, name = test_loader.load_data()
        base_name = os.path.splitext(name)[0] if '.' in name else name
        if base_name not in frozen_image_names: continue
            
        gt_mask_np = np.asarray(gt, np.float32)
        pil_gt = Image.fromarray(gt_mask_np.astype(np.uint8))
        gt_mask_np = np.array(pil_gt.resize((args.testsize, args.testsize), Image.NEAREST)).astype(np.float32)
        gt_mask_np /= (gt_mask_np.max() + 1e-8)
        
        image_tensor = image.to(device)
        
        # Initial forward pass to get base prediction and probability map
        with torch.no_grad():
            res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            initial_probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(initial_probs >= 0.5, 1, 0)
        
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)

        segments_slic = slic(orig_image_np_01, n_segments=args.superpixel_size, compactness=10, sigma=1)
        
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask_np, ic_threshold=1.0, oc_threshold=0.0)
        if not ic_sps_all: continue

        qualified_images.append({
            'name': name, 'image_tensor': image_tensor, 'orig_image_np_01': orig_image_np_01, 
            'gt_mask_np': gt_mask_np, 'segments_slic': segments_slic, 'pred_mask': pred_mask, 
            'initial_probs': initial_probs,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all
        })

    # =========================================================================
    # PHASE 2: GLOBAL XAI TRAJECTORY RUN
    # =========================================================================
    logger.info(f"\n[PHASE 2] Executing IC-Restricted Prob-Rank I/D Trajectories over 8 Baselines...")

    # Forced single run execution
    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images, desc=f"Processing Images", position=0):
        name, orig_image_np_01 = q_data['name'], q_data['orig_image_np_01']
        gt_mask_np, segments_slic = q_data['gt_mask_np'], q_data['segments_slic']
        pred_mask, initial_probs = q_data['pred_mask'], q_data['initial_probs']
        ic_sps_all, oc_sps_all = q_data['ic_sps_all'], q_data['oc_sps_all']

        current_dir = os.path.join(run_output_dir, name.rsplit('.', 1)[0])
        os.makedirs(current_dir, exist_ok=True)
        

        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided', 
            model_instance=model_instance, 
            device=device, 
            orig_image_np_01=orig_image_np_01, 
            gt_mask_np=gt_mask_np, 
            segments_slic=segments_slic, 
            sps_to_perturb=ic_sps_all,           # Perturbing only the GT anatomy (in-context superpixels)
            sps_to_measure=ic_sps_all,           # Strictly tracking the GT anatomy ranks
            initial_pred_mask=pred_mask, 
            initial_probs=initial_probs, 
            ic_sps_all=ic_sps_all, 
            oc_sps_all=oc_sps_all, 
            baselines=baselines,
            save_dir=current_dir         # Per-step 256x256 visuals
        )

        with open(os.path.join(current_dir, f"{name.rsplit('.', 1)[0]}_analysis.json"), 'w') as f: 
            json.dump({"prob_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
import os
import argparse
import json
import logging
import random
import math
import gc
import zlib
from abc import ABC, abstractmethod
from math import exp
import sys
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb, rgb2gray
from skimage.metrics import structural_similarity
from skimage import color
from skimage.filters import sobel
from skimage.util import regular_grid

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

# Direct imports for your actual modules
import data as Data
import model as Model
import core.logger as Logger

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
# 1. HELPER & ANALYSIS FUNCTIONS
# =================================================================================
def calculate_dice_score(mask1, mask2, label=1):
    if mask1.shape != mask2.shape:
        mask1 = np.resize(mask1, mask2.shape)
    mask1_bool = (mask1 == label)
    mask2_bool = (mask2 == label)
    intersection = np.sum(mask1_bool & mask2_bool)
    sum_of_masks = np.sum(mask1_bool) + np.sum(mask2_bool)
    if sum_of_masks == 0: return 100.0
    return ((2. * intersection) / sum_of_masks) * 100

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    """Absolute Anatomical Anchoring: Ground Truth explicitly defines domains."""
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
# Noise strength per unit noise level, tuned for XCAD. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'blur_sigma': 10.0,            # sigma = 10 * level
    'rician_sigma': 0.1,           # std = 0.1 * level
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

def generate_baseline_canvas(orig_image_01, baseline_type, img_name):
    """I/D canvas: 'black' (zeros), 'mean' (the image's mean intensity) or 'random'
    (uniform values in [0, 1] per pixel, seeded per image so all models get the same canvas)."""
    if baseline_type == 'black':
        return np.zeros_like(orig_image_01)
    elif baseline_type == 'mean':
        mean_color = np.mean(orig_image_01) # Scalar for 2D images
        return np.ones_like(orig_image_01) * mean_color
    elif baseline_type == 'random':
        rng = np.random.default_rng(zlib.crc32(img_name.encode()))
        return rng.uniform(0, 1, orig_image_01.shape).astype(orig_image_01.dtype)
    raise ValueError(f"Unknown baseline '{baseline_type}'")

# --- Raw step records (probability is used only to rank the IC superpixels) ---
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

def predict_slice_diffusion(model_diffusion, img_01, val_data_template, device):
    """Inference router mapping [0, 1] inputs to [-1, 1] tensors and extracting probabilities."""
    if img_01.ndim == 2:
        img_tensor = torch.from_numpy(img_01).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        img_tensor = torch.from_numpy(img_01).permute(2,0,1).unsqueeze(0).float().to(device)
        
    img_m11 = (img_tensor * 2.0) - 1.0
    
    val_data = val_data_template.copy()
    val_data['A'] = img_m11
    
    model_diffusion.feed_data(val_data)
    with torch.no_grad():
        model_diffusion.test()
        
    predseg_raw = model_diffusion.get_current_segment()['test_V'].squeeze().cpu().numpy()
    
    # Map [-1, 1] output safely to [0, 1] probabilities for SOTA ranking
    probs = np.clip((predseg_raw + 1.0) / 2.0, 0.0, 1.0)
    pred_mask_bool = (probs > 0.5)
    
    return pred_mask_bool, probs

def calculate_eigencam(diffusion, target_layer_name, val_data, image_size):
    feature_maps = []
    model_instance = diffusion.netG.segment_fn
    logger = logging.getLogger('base')

    try:
        target_layer = dict(model_instance.named_modules())[target_layer_name]
    except KeyError:
        logger.error(f"Target layer '{target_layer_name}' not found. Using blank heatmap.")
        return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)

    diffusion.feed_data(val_data)
    with torch.no_grad():
        diffusion.test() 

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

def visualize_segmentation_comparison(orig_image_01, gt_mask, pred_mask, all_sp_map, in_context_map, out_of_context_map, eigencam_heatmap, save_path=None):
    display_image_rgb = gray2rgb(orig_image_01) if orig_image_01.ndim == 2 else orig_image_01
    import matplotlib.pyplot as plt
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

# --------------------------------------------------------------------------
# --- 2. Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_diffusion, device, val_data, orig_image_01, target_class_gt, segments_slic,
    ic_sps_all, clean_ranks, initial_pred_mask_bool, initial_probs,
    baselines, img_name, save_dir
):
    """Insertion/Deletion over the IC superpixels, one superpixel per step, in clean probability-rank
    order (the allowed use of probability). Every step stores the raw record with fraction_perturbed."""
    analysis_results = {"trajectories": {}}

    if len(ic_sps_all) == 0: return analysis_results

    # --- Save Initial States and SP Map (same as Pascal VOC) ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)
    save_downsampled_image(mark_boundaries(orig_image_01, segments_slic), os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))

    # Deletion / insertion order: clean probability rank (ic_ranks); ties: larger area first, then sp_id
    sp_areas = {int(sp): int(np.sum(segments_slic == sp)) for sp in ic_sps_all}
    order = sorted(zip(clean_ranks, ic_sps_all), key=lambda t: (t[0], -sp_areas[int(t[1])], int(t[1])))
    ranking_methods = {'prob_ranked': [int(sp) for _, sp in order]}

    N_perturb = len(ic_sps_all)
    k_values = list(range(1, N_perturb + 1))  # one superpixel per step
    clean_bg_iou = bg_iou(initial_pred_mask_bool, target_class_gt)

    def record(step_idx, fraction, img_01, pred, probs):
        ssim_score = structural_similarity(orig_image_01, img_01, data_range=1.0)
        return step_record(step_idx, "fraction_perturbed", fraction, ssim_score, pred, probs, target_class_gt,
                           segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)

    # --- EXECUTE TRAJECTORIES OVER THE BASELINES ---
    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_01, baseline, img_name)
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        pred_n_bool, prob_n = predict_slice_diffusion(model_diffusion, bg_canvas, val_data, device)

        for r_name, r_ids in ranking_methods.items():
            # Step 0: deletion starts from the clean image, insertion from the full canvas
            traj = {
                "deletion": [step_record(0, "fraction_perturbed", 0.0, 1.0, initial_pred_mask_bool, initial_probs, target_class_gt,
                                         segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)],
                "insertion": [record(0, 0.0, bg_canvas, pred_n_bool, prob_n)],
            }

            baseline_vis_dir = os.path.join(vis_dir_root, baseline, r_name)
            os.makedirs(baseline_vis_dir, exist_ok=True)
            
            pbar = tqdm(k_values, desc=f"   -> [{domain_name[:2].upper()}] {baseline.upper()} | {r_name}", leave=False)
            for k in pbar:
                fraction = round(k / N_perturb, 4)
                mask_2d = np.isin(segments_slic, r_ids[:k])

                # --- 1. Deletion Step ---
                img_del = np.where(mask_2d, bg_canvas, orig_image_01)
                pred_del_bool, prob_del = predict_slice_diffusion(model_diffusion, img_del, val_data, device)
                traj["deletion"].append(record(k, fraction, img_del, pred_del_bool, prob_del))

                # --- 2. Insertion Step ---
                img_ins = np.where(mask_2d, orig_image_01, bg_canvas)
                pred_ins_bool, prob_ins = predict_slice_diffusion(model_diffusion, img_ins, val_data, device)
                traj["insertion"].append(record(k, fraction, img_ins, pred_ins_bool, prob_ins))

                # --- Save the perturbed inputs and predicted masks (same as Pascal VOC) ---
                step_vis_dir = os.path.join(baseline_vis_dir, f"step_{k:03d}")
                os.makedirs(step_vis_dir, exist_ok=True)
                
                save_downsampled_image(img_del, os.path.join(step_vis_dir, "deletion_input.jpg"))
                save_downsampled_image(pred_del_bool, os.path.join(step_vis_dir, "deletion_pred.png"), is_mask=True)
                
                save_downsampled_image(img_ins, os.path.join(step_vis_dir, "insertion_input.jpg"))
                save_downsampled_image(pred_ins_bool, os.path.join(step_vis_dir, "insertion_pred.png"), is_mask=True)

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            analysis_results["trajectories"][baseline][r_name] = traj

    return analysis_results

# =================================================================================
# 3. MAIN EXECUTION (TWO-PHASE ARCHITECTURE)
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (Diffusion)")
    
    # Diffusion Specific Logger / Config Args
    parser.add_argument('-c', '--config', type=str, default='config/test.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)

    # Core Args
    parser.add_argument("--output_dir", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/XCAD/results/cdarl/prism_out_insertion_deletion")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument('--target_layer_names', nargs='+', default=['conv_img'])
    parser.add_argument('--img_path', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/XCAD/img_sel.txt', help="Path to frozen images file.")

    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    if args.data_dir:
        if 'datasets' in opt and 'val' in opt['datasets']: opt['datasets']['val']['dataroot'] = args.data_dir
        elif 'datasets' in opt and 'test' in opt['datasets']: opt['datasets']['test']['dataroot'] = args.data_dir
        else: opt['datasets'] = {'test': {'name': 'xcad_test', 'mode': args.phase, 'dataroot': args.data_dir}}

    opt['phase'] = args.phase
    if args.gpu_ids: opt['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
    elif opt['gpu_ids'] is None: opt['gpu_ids'] = []

    if 'path' not in opt: opt['path'] = {}
    if 'log' not in opt['path']: opt['path']['log'] = './logs'
    if 'results' not in opt['path']: opt['path']['results'] = './results'

    final_log_dir = os.path.dirname(opt['path']['log']) if os.path.splitext(opt['path']['log'])[1] else opt['path']['log']
    os.makedirs(final_log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = logging.DEBUG if args.debug else logging.INFO
    Logger.setup_logger(None, final_log_dir, opt['phase'], level=log_level, screen=True)
    logger = logging.getLogger('base')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading Diffusion Dataset...")
    dataset_opt = opt['datasets']['test']
    val_set = Data.create_dataset_xcad(dataset_opt, 'test')
    val_loader = Data.create_dataloader(val_set, dataset_opt, 'test')

    logger.info("Loading Diffusion model...")
    diffusion = Model.create_model(opt)
    diffusion.netG.to(device)

    # I/D canvases
    baselines = ['black', 'mean', 'random']

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS & HEATMAP GENERATION
    # =========================================================================
    logger.info(f"\n[PHASE 1] Scanning Candidates to Qualify {args.num_images} Images...")
    qualified_images = []
    
    cohort_file = args.img_path
    if not os.path.exists(cohort_file):
        logger.error(f"FATAL: Cohort file '{cohort_file}' missing.")
        sys.exit(1)

    with open(cohort_file, 'r') as f:
        frozen_cohort = set([line.strip() for line in f.readlines() if line.strip()])
    
    for idata, val_data in enumerate(tqdm(val_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break

        case_name = os.path.basename(val_data['P'][0]).replace('.png', '')
        if case_name not in frozen_cohort: continue
        
        orig_image_np_raw = val_data['A'][0].cpu().numpy()
        if orig_image_np_raw.ndim == 3:
            orig_image_np = rgb2gray(orig_image_np_raw.transpose(1, 2, 0)) if orig_image_np_raw.shape[0] == 3 else orig_image_np_raw.squeeze()
        else:
            orig_image_np = orig_image_np_raw
            
        orig_image_01 = np.clip((orig_image_np + 1.0) / 2.0, 0.0, 1.0)

        # Initial forward pass extracting specific continuous probabilities
        initial_pred_mask_bool, initial_probs = predict_slice_diffusion(diffusion, orig_image_01, val_data, device)

        gt_mask_raw = val_data['F'][0].squeeze().cpu().numpy()
        target_class_gt = ((gt_mask_raw + 1.0) / 2.0 > 0.5)
        
        # Calculate EigenCAM for Dual-Ranking benchmark
        try:
            eigencam_heatmap = calculate_eigencam(diffusion, args.target_layer_names[0], val_data, orig_image_01.shape)
        except Exception:
            eigencam_heatmap = np.zeros_like(orig_image_01, dtype=np.float32)

        segments = slic(orig_image_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        # Absolute Anchoring
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=0.85, oc_threshold=0.0)
        
        if not ic_sps_all: continue

        qualified_images.append({
            'case_name': case_name, 'val_data': val_data, 'orig_image_01': orig_image_01, 
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'initial_probs': initial_probs,
            'eigencam_heatmap': eigencam_heatmap,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all,
            'initial_dice_vs_gt': calculate_dice_score(initial_pred_mask_bool.astype(np.uint8), target_class_gt.astype(np.uint8))
        })

    logger.info(f"Scan complete. {len(qualified_images)} images fully qualified.")
    if len(qualified_images) == 0: sys.exit()

    # =========================================================================
    # PHASE 2: GLOBAL XAI TRAJECTORY RUN
    # =========================================================================
    logger.info(f"\n[PHASE 2] Executing IC-Restricted Prob-Rank I/D Trajectories over {len(baselines)} Baselines...")

    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images, desc="Processing Images", position=0):
        case_name = q_data['case_name']
        val_data = q_data['val_data']
        orig_image_01 = q_data['orig_image_01']
        target_class_gt = q_data['target_class_gt']
        segments = q_data['segments']
        initial_pred_mask_bool = q_data['initial_pred_mask_bool']
        initial_probs = q_data['initial_probs']
        eigencam_heatmap = q_data['eigencam_heatmap']
        ic_sps_all = q_data['ic_sps_all']
        oc_sps_all = q_data['oc_sps_all']

        current_dir = os.path.join(run_output_dir, case_name)
        os.makedirs(current_dir, exist_ok=True)

        # Generate Visualizations
        in_context_map, out_of_context_map = np.zeros_like(segments), np.zeros_like(segments)
        for sp_id in ic_sps_all: in_context_map[segments == sp_id] = sp_id
        for sp_id in oc_sps_all: out_of_context_map[segments == sp_id] = sp_id
        
        visualize_segmentation_comparison(orig_image_01, target_class_gt, initial_pred_mask_bool, segments, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_dir, case_name + "_overview.png"))

        # Clean ranking of the IC superpixels, computed once per image and used for every step
        clean_ranks, clean_means = ic_ranks(initial_probs, segments, ic_sps_all)

        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided',
            model_diffusion=diffusion,
            device=device,
            val_data=val_data,
            orig_image_01=orig_image_01,
            target_class_gt=target_class_gt,
            segments_slic=segments,
            ic_sps_all=ic_sps_all,               # Perturbing and tracking only the GT anatomy (in-context superpixels)
            clean_ranks=clean_ranks,
            initial_pred_mask_bool=initial_pred_mask_bool,
            initial_probs=initial_probs,
            baselines=baselines,                  # black / mean / random canvases
            img_name=case_name,
            save_dir=current_dir         # Per-step 256x256 visuals
        )

        result = {
            "image_info": {"name": case_name},
            "clean_ranking": clean_ranking_record(segments, ic_sps_all, clean_ranks, clean_means),
            "prob_guided_analysis": res_global,
        }
        with open(os.path.join(current_dir, f"{case_name}_analysis.json"), 'w') as f:
            json.dump(result, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
import os, sys
import json
import argparse
import logging
import random
import time
import itertools
import math
import gc
import zlib
from math import exp, ceil

import numpy as np
import pandas as pd
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
from skimage.color import gray2rgb, rgb2gray
import torch.utils.data as data

from lib.TransFuse import TransFuse_S 

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

def get_strict_ic_oc_superpixels(superpixels_segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0):
    """Absolute Anatomical Anchoring: Ground Truth explicitly defines domains."""
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

# --- RGB CHANNEL-WISE FIX: Noise Application ---
# Noise strength per unit noise level, tuned for ISIC. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
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
        target_std = NOISE_PARAMS['gaussian_sigma'] * noise_level          # Poisson std on mid-grey matches the Gaussian std
        scale_factor = 0.5 / max(target_std, 1e-6) ** 2
        noise_char_value = target_std
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

def generate_baseline_canvas(orig_image_01, baseline_type, img_name):
    """I/D canvas: 'black' (zeros), 'mean' (the image's per-channel mean colour) or 'random'
    (uniform values in [0, 1] per pixel and channel, seeded per image so all models get the same canvas)."""
    if baseline_type == 'black':
        return np.zeros_like(orig_image_01)
    elif baseline_type == 'mean':
        mean_color = np.mean(orig_image_01, axis=(0, 1), keepdims=True)
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

def predict_from_numpy(model_instance, img_np_01, device, gt_shape):
    """TransFuse-specific helper to maintain ImageNet normalization during I/D steps."""
    tensor = torch.from_numpy(img_np_01).permute(2, 0, 1).float()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = normalize(tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, res = model_instance(input_tensor) # TransFuse specific unpack
        res = F.upsample(res, size=gt_shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred_mask = np.where(probs >= 0.5, 1, 0)
    return pred_mask, probs

def calculate_eigencam(model, target_layer_name, image_tensor, image_size):
    feature_maps = []
    logger = logging.getLogger('base')

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
    import matplotlib.pyplot as plt
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
    axes[4].imshow(mark_boundaries(display_image_rgb, in_context_map)); axes[4].set_title("In-Context SPs"); axes[4].axis('off')
    axes[5].imshow(mark_boundaries(display_image_rgb, out_of_context_map)); axes[5].set_title("Out-of-Context SPs"); axes[5].axis('off')
    axes[6].imshow(display_image_rgb); im = axes[6].imshow(eigencam_heatmap, cmap='jet', alpha=1); axes[6].set_title("EigenCAM"); axes[6].axis('off')

    plt.colorbar(im, ax=axes[6])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# --------------------------------------------------------------------------
# --- 3. Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_instance, device, orig_image_np_01, gt_mask_np, segments_slic,
    ic_sps_all, clean_ranks, initial_pred_mask, initial_probs,
    baselines, img_name, save_dir
):
    """Insertion/Deletion over the IC superpixels, one superpixel per step, in clean probability-rank
    order (the allowed use of probability). Every step stores the raw record with fraction_perturbed."""
    analysis_results = {"trajectories": {}}
    gt_shape = gt_mask_np.shape

    if len(ic_sps_all) == 0: return analysis_results

    # --- Save Initial States and SP Map (same as Pascal VOC) ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)
    save_downsampled_image(mark_boundaries(orig_image_np_01, segments_slic), os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_np_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))

    # Deletion / insertion order: clean probability rank (ic_ranks); ties: larger area first, then sp_id
    sp_areas = {int(sp): int(np.sum(segments_slic == sp)) for sp in ic_sps_all}
    order = sorted(zip(clean_ranks, ic_sps_all), key=lambda t: (t[0], -sp_areas[int(t[1])], int(t[1])))
    ranking_methods = {'prob_ranked': [int(sp) for _, sp in order]}

    N_perturb = len(ic_sps_all)
    k_values = list(range(1, N_perturb + 1))  # one superpixel per step
    clean_bg_iou = bg_iou(initial_pred_mask, gt_mask_np)

    def record(step_idx, fraction, img_01, pred, probs):
        ssim_score = structural_similarity(orig_image_np_01, img_01, data_range=1.0, channel_axis=-1)
        return step_record(step_idx, "fraction_perturbed", fraction, ssim_score, pred, probs, gt_mask_np,
                           segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)

    # --- EXECUTE TRAJECTORIES OVER THE BASELINES ---
    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_np_01, baseline, img_name)
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        pred_n, prob_n = predict_from_numpy(model_instance, bg_canvas, device, gt_shape)

        for r_name, r_ids in ranking_methods.items():
            # Step 0: deletion starts from the clean image, insertion from the full canvas
            traj = {
                "deletion": [step_record(0, "fraction_perturbed", 0.0, 1.0, initial_pred_mask, initial_probs, gt_mask_np,
                                         segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)],
                "insertion": [record(0, 0.0, bg_canvas, pred_n, prob_n)],
            }

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
                traj["deletion"].append(record(k, fraction, img_del, pred_del, prob_del))

                # --- 2. Insertion Step ---
                img_ins = np.where(mask_3d, orig_image_np_01, bg_canvas)
                pred_ins, prob_ins = predict_from_numpy(model_instance, img_ins, device, gt_shape)
                traj["insertion"].append(record(k, fraction, img_ins, pred_ins, prob_ins))

                # --- Save the perturbed inputs and predicted masks (same as Pascal VOC) ---
                step_vis_dir = os.path.join(baseline_vis_dir, f"step_{k:03d}")
                os.makedirs(step_vis_dir, exist_ok=True)
                
                save_downsampled_image(img_del, os.path.join(step_vis_dir, "deletion_input.jpg"))
                save_downsampled_image(pred_del, os.path.join(step_vis_dir, "deletion_pred.png"), is_mask=True)
                
                save_downsampled_image(img_ins, os.path.join(step_vis_dir, "insertion_input.jpg"))
                save_downsampled_image(pred_ins, os.path.join(step_vis_dir, "insertion_pred.png"), is_mask=True)

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            analysis_results["trajectories"][baseline][r_name] = traj

    return analysis_results

# --------------------------------------------------------------------------
# --- 4. Main Two-Phase Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (TransFuse)")
    parser.add_argument('--pth_path', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/TransFuse/TransFuse-19_best.pth')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/cluster/home/bhupenk/bhupender/datasets/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--target_layer_name', type=str, default='up_c')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/results/transfuse/prism_out_insertion_deletion')
    parser.add_argument('--txtfile', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/ISIC/img_sel.txt')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--superpixel_size', type=int, default=150)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
    logger = logging.getLogger('base')
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

    # I/D canvases
    baselines = ['black', 'mean', 'random']

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS
    # =========================================================================
    logger.info(f"\n[PHASE 1] Strict Anatomical Qualifier Scan...")
    qualified_images = []
    
    if not os.path.exists(args.txtfile):
        logger.error(f"FATAL: Frozen cohort file '{args.txtfile}' not found.")
        sys.exit()
    
    with open(args.txtfile, 'r') as f:
        frozen_image_names = set([line.strip() for line in f.readlines()])
    
    for i, (image, gt, name_list) in enumerate(tqdm(test_loader, desc="Qualifying Slices")):
        if len(qualified_images) >= args.num_images: break
        
        name = name_list[0]
        if name not in frozen_image_names: continue
            
        image_tensor = image.to(device)
        _, _, h, w = image_tensor.shape

        gt_mask_np = np.asarray(gt, np.float32).squeeze(0).squeeze(0)
        pil_gt = Image.fromarray(gt_mask_np.astype(np.uint8))
        gt_mask_np = np.array(pil_gt.resize((w, h), Image.NEAREST)).astype(np.float32)
        gt_mask_np /= (gt_mask_np.max() + 1e-8)

        # Initial forward pass to extract continuous probabilities
        with torch.no_grad():
            _, _, res = model_instance(image_tensor)
            res = F.upsample(res, size=gt_mask_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            initial_probs = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_mask = np.where(initial_probs >= 0.5, 1, 0)
        
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        image_for_slic = inv_normalize(image_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        orig_image_np_01 = np.clip(image_for_slic, 0, 1)

        # Calculate EigenCAM for Dual-Ranking benchmark
        eigencam_heatmap = calculate_eigencam(model_instance, args.target_layer_name, image_tensor, (orig_image_np_01.shape[0], orig_image_np_01.shape[1]))

        segments_slic = slic(orig_image_np_01, n_segments=args.superpixel_size, compactness=10, sigma=1)
        
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask_np, ic_threshold=1.0, oc_threshold=0.0)
        if not ic_sps_all: continue

        qualified_images.append({
            'name': name, 'image_tensor': image_tensor, 'orig_image_np_01': orig_image_np_01, 
            'gt_mask_np': gt_mask_np, 'segments_slic': segments_slic, 
            'pred_mask': pred_mask, 'initial_probs': initial_probs, 'eigencam_heatmap': eigencam_heatmap,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all
        })

    # =========================================================================
    # PHASE 2: SINGLE-RUN CAUSAL TRAJECTORY EXTRACTION (I/D)
    # =========================================================================
    logger.info(f"\n[PHASE 2] Executing IC-Restricted Prob-Rank I/D Trajectories over {len(baselines)} Baselines...")

    run_output_dir = args.output_dir
    os.makedirs(run_output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images, desc="Processing Images", position=0):
        name, orig_image_np_01 = q_data['name'], q_data['orig_image_np_01']
        gt_mask_np, segments_slic = q_data['gt_mask_np'], q_data['segments_slic']
        pred_mask, initial_probs = q_data['pred_mask'], q_data['initial_probs']
        eigencam_heatmap = q_data['eigencam_heatmap']
        ic_sps_all, oc_sps_all = q_data['ic_sps_all'], q_data['oc_sps_all']

        current_dir = os.path.join(run_output_dir, name)
        os.makedirs(current_dir, exist_ok=True)

        # Generate Visualizations
        in_context_map, out_of_context_map = np.zeros_like(segments_slic), np.zeros_like(segments_slic)
        for sp_id in ic_sps_all: in_context_map[segments_slic == sp_id] = sp_id
        for sp_id in oc_sps_all: out_of_context_map[segments_slic == sp_id] = sp_id
        
        gt_mask_for_sp = (np.array(Image.fromarray((gt_mask_np * 255).astype(np.uint8)).resize(segments_slic.shape[::-1], Image.NEAREST)) > 127).astype(np.uint8) if gt_mask_np.shape != segments_slic.shape else gt_mask_np.astype(np.uint8)
        
        visualize_segmentation_comparison(orig_image_np_01, gt_mask_for_sp, pred_mask, segments_slic, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_dir, name + "_overview.png"))

        # Clean ranking of the IC superpixels, computed once per image and used for every step
        clean_ranks, clean_means = ic_ranks(initial_probs, segments_slic, ic_sps_all)

        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided',
            model_instance=model_instance,
            device=device,
            orig_image_np_01=orig_image_np_01,
            gt_mask_np=gt_mask_np,
            segments_slic=segments_slic,
            ic_sps_all=ic_sps_all,               # Perturbing and tracking only the GT anatomy (in-context superpixels)
            clean_ranks=clean_ranks,
            initial_pred_mask=pred_mask,
            initial_probs=initial_probs,
            baselines=baselines,                  # black / mean / random canvases
            img_name=name,
            save_dir=current_dir         # Per-step 256x256 visuals
        )

        result = {
            "image_info": {"name": name},
            "clean_ranking": clean_ranking_record(segments_slic, ic_sps_all, clean_ranks, clean_means),
            "prob_guided_analysis": res_global,
        }
        with open(os.path.join(current_dir, f"{name}_analysis.json"), 'w') as f:
            json.dump(result, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
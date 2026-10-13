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

def generate_dynamic_steps(N, max_steps=20):
    """Guarantees strict monotonic scaling mapping to the exact superpixel count."""
    if N == 0: return []
    if N <= max_steps: return list(range(1, N + 1))
    return sorted(list(set(np.linspace(1, N, max_steps).astype(int))))

def perturb_superpixels_with_level(original_image_scaled, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """Generates continuous noise mapping."""
    perturbed_image = np.copy(original_image_scaled)
    region_mask = np.isin(superpixels_segments, selected_superpixel_ids)
    if not np.any(region_mask): return perturbed_image
    
    H, W, C = perturbed_image.shape
    if noise_type == 'gaussian':
        for c in range(C): perturbed_image[region_mask, c] += np.random.normal(0, 0.1 * noise_level, size=(H, W))[region_mask]
    elif noise_type == 'salt_pepper':
        density = min(0.5, 0.05 * noise_level)
        rows, cols = np.where(region_mask)
        num_noise = int(density * len(rows))
        if num_noise > 0:
            indices = np.random.choice(len(rows), num_noise, replace=False)
            for c in range(C): perturbed_image[rows[indices], cols[indices], c] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        scale = max(0.05, 3.2 - noise_level) * 500.0
        for c in range(C): perturbed_image[region_mask, c] = np.random.poisson(np.clip(perturbed_image[region_mask, c], 0, 1) * scale) / scale
    elif noise_type == 'speckle':
        for c in range(C): perturbed_image[region_mask, c] *= (1.0 + np.random.normal(0, 0.1 * noise_level, size=(H, W))[region_mask])
    elif noise_type == 'gaussian_blur':
        for c in range(C): perturbed_image[region_mask, c] = gaussian_filter(original_image_scaled[:, :, c], sigma=50 * noise_level)[region_mask]
    elif noise_type == 'rician':
        for c in range(C):
            px = perturbed_image[region_mask, c]
            perturbed_image[region_mask, c] = np.sqrt(np.square(px + np.random.normal(0, 0.1 * noise_level, size=px.shape)) + np.square(np.random.normal(0, 0.1 * noise_level, size=px.shape)))
            
    return np.clip(perturbed_image, 0.0, 1.0)

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
        return perturb_superpixels_with_level(orig_image_01, dummy_all_mask, [1], baseline_type, noise_level=3.0)

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
# --- 3. Global EigenCAM Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_instance, device, orig_image_np_01, gt_mask_np, segments_slic,
    sps_to_perturb, sps_to_measure, initial_pred_mask, initial_probs, eigencam_heatmap,
    ic_sps_all, oc_sps_all, baselines
):
    """Executes Insertion/Deletion ranked by Probability AND EigenCAM, saving step-wise superpixel metrics on GT."""
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    gt_shape = gt_mask_np.shape
    
    if len(sps_to_perturb) == 0 or len(sps_to_measure) == 0: return analysis_results
    
    max_id = int(np.max(segments_slic)) + 1
    sp_areas = {int(sp_id): int(np.sum(segments_slic == sp_id)) for sp_id in np.unique(segments_slic)}
    
    # 1. Map Areas, Domains, Probabilities, and EigenCAM Intensities
    rank_data = []
    for sp_id in sps_to_perturb:
        mask = (segments_slic == sp_id)
        area = sp_areas[int(sp_id)]
        mean_p = float(np.mean(initial_probs[mask]))
        mean_eigen = float(np.mean(eigencam_heatmap[mask]))
        
        if sp_id in ic_sps_all: gt_domain = "IC"
        elif sp_id in oc_sps_all: gt_domain = "OC"
        else: gt_domain = "Mixed"

        rank_data.append({
            "sp_id": int(sp_id), 
            "area": area, 
            "mean_prob": mean_p, 
            "mean_eigen": mean_eigen,
            "gt_domain": gt_domain
        })
        
    # 2. Assign explicit Rank Integers
    rank_data.sort(key=lambda x: (x['mean_prob'], x['area']), reverse=True)
    for rank, item in enumerate(rank_data, 1):
        item['prob_rank'] = rank
    prob_ranked_ids = [item['sp_id'] for item in rank_data]

    rank_data.sort(key=lambda x: (x['mean_eigen'], x['area']), reverse=True)
    for rank, item in enumerate(rank_data, 1):
        item['eigen_rank'] = rank
    eigen_ranked_ids = [item['sp_id'] for item in rank_data]
        
    analysis_results["rank_distribution"] = rank_data
    ranking_methods = {'prob_ranked': prob_ranked_ids, 'eigen_ranked': eigen_ranked_ids}
    
    # 3. Dynamic Steps & Helpers
    N_perturb = len(sps_to_perturb)
    k_values = generate_dynamic_steps(N_perturb, max_steps=20)
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
        # 🚨 FIX: Renamed key to 'global_iou'
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
        # 🚨 FIX: Map to the renamed 'global_iou' key
        y = [e['global_iou'] / 100.0 for e in traj]
        return float(auc(x, y))

    # --- EXECUTE TRAJECTORIES OVER THE 8 BASELINES ---
    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_np_01, baseline)
        
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

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            traj["auc_del"] = calculate_auc(traj["deletion"])
            traj["auc_ins"] = calculate_auc(traj["insertion"])
            analysis_results["trajectories"][baseline][r_name] = traj

    return analysis_results

# --------------------------------------------------------------------------
# --- 4. Main Two-Phase Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (TransFuse)")
    parser.add_argument('--pth_path', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/TransFuse-19_best.pth')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dataroot_val', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--target_layer_name', type=str, default='up_c')
    parser.add_argument('--output_dir', type=str, default='./prism_out_insertion_deletion')
    parser.add_argument('--txtfile', type=str, default='/bhupender/2025/Mouse_other/Skin_Lesion/img_sel.txt')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--superpixel_size', type=int, default=500)
    
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

    # 6+2 Baselines
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

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
    logger.info(f"\n[PHASE 2] Executing Global Dual-Rank I/D Trajectories over 8 Baselines...")

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
        
        all_sps = np.unique(segments_slic)
        all_sps = all_sps[all_sps != 0]

        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='global_eigen_guided', 
            model_instance=model_instance, 
            device=device, 
            orig_image_np_01=orig_image_np_01, 
            gt_mask_np=gt_mask_np, 
            segments_slic=segments_slic, 
            sps_to_perturb=all_sps,              # Perturbing the entire image based on Eigen Rank
            sps_to_measure=ic_sps_all,           # Strictly tracking the GT anatomy ranks
            initial_pred_mask=pred_mask, 
            initial_probs=initial_probs, 
            eigencam_heatmap=eigencam_heatmap, 
            ic_sps_all=ic_sps_all, 
            oc_sps_all=oc_sps_all, 
            baselines=baselines                  # Loop over 6+2 canvases
        )

        with open(os.path.join(current_dir, f"{name}_analysis.json"), 'w') as f: 
            json.dump({"eigen_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
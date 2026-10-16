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
from sklearn.metrics import auc
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
    'poisson_photons': 300.0,      # photons = 300 * max(0.05, 3.2 - level)
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
        photons = NOISE_PARAMS['poisson_photons'] * max(0.05, 3.2 - noise_level)
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

def generate_baseline_canvas(orig_image_01, baseline_type):
    """Generates the Null-Information canvas (Black, Mean, or Max-Noise) for I/D."""
    if baseline_type == 'black':
        return np.zeros_like(orig_image_01)
    elif baseline_type == 'mean':
        mean_color = np.mean(orig_image_01) # Scalar for 2D images
        return np.ones_like(orig_image_01) * mean_color
    else:
        dummy_all_mask = np.ones_like(orig_image_01, dtype=int)
        return perturb_image_slice(orig_image_01, dummy_all_mask, [1], baseline_type, noise_level=30.0)

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
    sps_to_perturb, sps_to_measure, initial_pred_mask_bool, initial_probs,
    ic_sps_all, oc_sps_all, baselines
):
    """Executes Insertion/Deletion ranked by probability, one superpixel per step, saving step-wise superpixel metrics on GT."""
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    
    if len(sps_to_perturb) == 0 or len(sps_to_measure) == 0: return analysis_results
    
    max_id = int(np.max(segments_slic)) + 1
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
    bg_gt = (target_class_gt == 0)

    # Function to get individual GT superpixel tracking at each step
    def get_step_superpixels(current_pred_mask, current_probs):
        # 🚨 FIX: Micro IoU now strictly compares current_pred_mask to the Ground Truth (target_class_gt == 1)
        intersect_mask = (current_pred_mask == 1) & (target_class_gt == 1)
        union_mask = (current_pred_mask == 1) | (target_class_gt == 1)
        
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

    base_sps = get_step_superpixels(initial_pred_mask_bool, initial_probs)
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
        bg_canvas = generate_baseline_canvas(orig_image_01, baseline)
        
        # Base Deletion Metrics
        base_intersect = np.sum((initial_pred_mask_bool == 1) & (target_class_gt == 1))
        base_union = np.sum((initial_pred_mask_bool == 1) | (target_class_gt == 1))
        base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0
        base_f_prob = float(np.mean(initial_probs[initial_pred_mask_bool == 1])) if np.any(initial_pred_mask_bool == 1) else 0.0

        # Base Insertion Metrics (Fully destroyed image)
        pred_n_bool, prob_n = predict_slice_diffusion(model_diffusion, bg_canvas, val_data, device)
        
        ins_union0 = np.sum((pred_n_bool == 1) | (target_class_gt == 1))
        ins_base_iou = (np.sum((pred_n_bool == 1) & (target_class_gt == 1)) / ins_union0 * 100.0) if ins_union0 > 0 else 100.0
        ins_base_prob = float(np.mean(prob_n[pred_n_bool == 1])) if np.any(pred_n_bool == 1) else 0.0
        ins_base_sps = get_step_superpixels(pred_n_bool, prob_n)

        for r_name, r_ids in ranking_methods.items():
            traj = {"deletion": [], "insertion": []}
            traj["deletion"].append(create_entry(0.0, base_global_iou, 0.0, base_f_prob, base_sps))
            traj["insertion"].append(create_entry(0.0, ins_base_iou, 0.0, ins_base_prob, ins_base_sps))
            
            pbar = tqdm(k_values, desc=f"   -> [{domain_name[:2].upper()}] {baseline.upper()} | {r_name}", leave=False)
            for k in pbar:
                fraction = round(k / N_perturb, 4)
                mask_2d = np.isin(segments_slic, r_ids[:k])

                # --- 1. Deletion Step ---
                img_del = np.where(mask_2d, bg_canvas, orig_image_01)
                pred_del_bool, prob_del = predict_slice_diffusion(model_diffusion, img_del, val_data, device)
                
                iou_del = (np.sum((pred_del_bool == 1) & (target_class_gt == 1)) / max(1, np.sum((pred_del_bool == 1) | (target_class_gt == 1)))) * 100.0
                spill_del = np.sum((pred_del_bool == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_del = float(np.mean(prob_del[pred_del_bool == 1])) if np.any(pred_del_bool == 1) else 0.0
                del_sps = get_step_superpixels(pred_del_bool, prob_del)
                traj["deletion"].append(create_entry(fraction, iou_del, spill_del, prob_f_del, del_sps))

                # --- 2. Insertion Step ---
                img_ins = np.where(mask_2d, orig_image_01, bg_canvas)
                pred_ins_bool, prob_ins = predict_slice_diffusion(model_diffusion, img_ins, val_data, device)
                
                iou_ins = (np.sum((pred_ins_bool == 1) & (target_class_gt == 1)) / max(1, np.sum((pred_ins_bool == 1) | (target_class_gt == 1)))) * 100.0
                spill_ins = np.sum((pred_ins_bool == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_ins = float(np.mean(prob_ins[pred_ins_bool == 1])) if np.any(pred_ins_bool == 1) else 0.0
                ins_sps = get_step_superpixels(pred_ins_bool, prob_ins)
                traj["insertion"].append(create_entry(fraction, iou_ins, spill_ins, prob_f_ins, ins_sps))

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            traj["auc_del"] = calculate_auc(traj["deletion"])
            traj["auc_ins"] = calculate_auc(traj["insertion"])
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

    # 6+2 Baselines
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

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
    logger.info(f"\n[PHASE 2] Executing IC-Restricted Prob-Rank I/D Trajectories over 8 Baselines...")

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


        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided', 
            model_diffusion=diffusion, 
            device=device, 
            val_data=val_data, 
            orig_image_01=orig_image_01, 
            target_class_gt=target_class_gt, 
            segments_slic=segments, 
            sps_to_perturb=ic_sps_all,           # Perturbing only the GT anatomy (in-context superpixels)
            sps_to_measure=ic_sps_all,           # Strictly tracking the GT anatomy ranks
            initial_pred_mask_bool=initial_pred_mask_bool, 
            initial_probs=initial_probs, 
            ic_sps_all=ic_sps_all, 
            oc_sps_all=oc_sps_all, 
            baselines=baselines                  # Loop over 6+2 canvases
        )

        with open(os.path.join(current_dir, f"{case_name}_analysis.json"), 'w') as f: 
            json.dump({"prob_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
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
from sklearn.metrics import auc
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
# Noise strength per unit noise level, tuned for Synapse. The noise forms are shared by all
# PRISM datasets (RGB: Pascal VOC, ISIC; grayscale: Synapse, XCAD); only these values differ.
NOISE_PARAMS = {
    'gaussian_sigma': 0.1,         # std = 0.1 * level
    'speckle_sigma': 0.1,          # std = 0.1 * level
    'salt_pepper_density': 0.05,   # density = min(0.5, 0.05 * level)
    'poisson_photons': 100.0,      # photons = 100 * max(0.05, 3.2 - level)
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
        mean_color = np.mean(orig_image_01) # Scalar for 2D images
        return np.ones_like(orig_image_01) * mean_color
    else:
        # Generate the fully saturated noisy background for the 6 noise variants
        dummy_all_mask = np.ones_like(orig_image_01, dtype=int)
        return perturb_image_slice(orig_image_01, dummy_all_mask, [1], baseline_type, noise_level=30.0)

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
        except AttributeError:
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
# --- 3. Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_instance, device, orig_image_01, target_class_gt, segments_slic,
    sps_to_perturb, sps_to_measure, initial_pred_mask_bool, initial_probs,
    patch_size, target_class_index, ic_sps_all, oc_sps_all, baselines, save_dir
):
    """Executes Insertion/Deletion ranked by probability, one superpixel per step, saving step-wise superpixel metrics on GT."""
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    
    if len(sps_to_perturb) == 0 or len(sps_to_measure) == 0: return analysis_results
    
    max_id = int(np.max(segments_slic)) + 1

    # --- Save Initial States and SP Map (same as Pascal VOC) ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)
    save_downsampled_image(mark_boundaries(orig_image_01, segments_slic), os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))
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
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        
        # Base Deletion Metrics
        base_intersect = np.sum((initial_pred_mask_bool == 1) & (target_class_gt == 1))
        base_union = np.sum((initial_pred_mask_bool == 1) | (target_class_gt == 1))
        base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0
        base_f_prob = float(np.mean(initial_probs[initial_pred_mask_bool == 1])) if np.any(initial_pred_mask_bool == 1) else 0.0

        # Base Insertion Metrics (Fully destroyed image)
        pred_n_bool, prob_n, _ = predict_slice_maxvit(model_instance, bg_canvas, patch_size, target_class_index, device)
        
        ins_union0 = np.sum((pred_n_bool == 1) | (target_class_gt == 1))
        ins_base_iou = (np.sum((pred_n_bool == 1) & (target_class_gt == 1)) / ins_union0 * 100.0) if ins_union0 > 0 else 100.0
        ins_base_prob = float(np.mean(prob_n[pred_n_bool == 1])) if np.any(pred_n_bool == 1) else 0.0
        ins_base_sps = get_step_superpixels(pred_n_bool, prob_n)

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

                # --- 1. Deletion Step ---
                img_del = np.where(mask_2d, bg_canvas, orig_image_01)
                pred_del_bool, prob_del, _ = predict_slice_maxvit(model_instance, img_del, patch_size, target_class_index, device)
                
                iou_del = (np.sum((pred_del_bool == 1) & (target_class_gt == 1)) / max(1, np.sum((pred_del_bool == 1) | (target_class_gt == 1)))) * 100.0
                spill_del = np.sum((pred_del_bool == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_del = float(np.mean(prob_del[pred_del_bool == 1])) if np.any(pred_del_bool == 1) else 0.0
                del_sps = get_step_superpixels(pred_del_bool, prob_del)
                traj["deletion"].append(create_entry(fraction, iou_del, spill_del, prob_f_del, del_sps))

                # --- 2. Insertion Step ---
                img_ins = np.where(mask_2d, orig_image_01, bg_canvas)
                pred_ins_bool, prob_ins, _ = predict_slice_maxvit(model_instance, img_ins, patch_size, target_class_index, device)
                
                iou_ins = (np.sum((pred_ins_bool == 1) & (target_class_gt == 1)) / max(1, np.sum((pred_ins_bool == 1) | (target_class_gt == 1)))) * 100.0
                spill_ins = np.sum((pred_ins_bool == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_ins = float(np.mean(prob_ins[pred_ins_bool == 1])) if np.any(pred_ins_bool == 1) else 0.0
                ins_sps = get_step_superpixels(pred_ins_bool, prob_ins)
                traj["insertion"].append(create_entry(fraction, iou_ins, spill_ins, prob_f_ins, ins_sps))

                # --- Save the perturbed inputs and predicted masks (same as Pascal VOC) ---
                step_vis_dir = os.path.join(baseline_vis_dir, f"step_{k:03d}")
                os.makedirs(step_vis_dir, exist_ok=True)
                
                save_downsampled_image(img_del, os.path.join(step_vis_dir, "deletion_input.jpg"))
                save_downsampled_image(pred_del_bool, os.path.join(step_vis_dir, "deletion_pred.png"), is_mask=True)
                
                save_downsampled_image(img_ins, os.path.join(step_vis_dir, "insertion_input.jpg"))
                save_downsampled_image(pred_ins_bool, os.path.join(step_vis_dir, "insertion_pred.png"), is_mask=True)

                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            traj["auc_del"] = calculate_auc(traj["deletion"])
            traj["auc_ins"] = calculate_auc(traj["insertion"])
            analysis_results["trajectories"][baseline][r_name] = traj

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
# 4. MAIN EXECUTION
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

    # 6+2 Baselines
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

    # =========================================================================
    # PHASE 1: ABSOLUTE ANATOMICAL QUALIFIERS & HEATMAP GENERATION
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
        
        eigencam_heatmap = calculate_eigencam(net, input_tensor, args.target_layer_names, (image_slice_01.shape[0], image_slice_01.shape[1]))
        segments = slic(image_slice_01, n_segments=args.num_superpixels, compactness=0.1, sigma=1.0, channel_axis=None)

        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments, target_class_gt, ic_threshold=1.0, oc_threshold=0.0)
        if not ic_sps_all: continue

        qualified_images.append({
            'case_name': case_name, 'image_slice_01': image_slice_01, 'input_tensor': input_tensor,
            'target_class_gt': target_class_gt, 'segments': segments, 
            'initial_pred_mask_bool': initial_pred_mask_bool, 'initial_probs': initial_probs,
            'eigencam_heatmap': eigencam_heatmap,
            'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all
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
        case_name, image_slice_01 = q_data['case_name'], q_data['image_slice_01']
        target_class_gt, segments = q_data['target_class_gt'], q_data['segments']
        initial_pred_mask_bool, initial_probs = q_data['initial_pred_mask_bool'], q_data['initial_probs']
        eigencam_heatmap = q_data['eigencam_heatmap']
        ic_sps_all, oc_sps_all = q_data['ic_sps_all'], q_data['oc_sps_all']

        current_dir = os.path.join(run_output_dir, case_name)
        os.makedirs(current_dir, exist_ok=True)

        # Generate Visualizations
        in_context_map, out_of_context_map = np.zeros_like(segments), np.zeros_like(segments)
        for sp_id in ic_sps_all: in_context_map[segments == sp_id] = sp_id
        for sp_id in oc_sps_all: out_of_context_map[segments == sp_id] = sp_id
        
        visualize_segmentation_comparison(image_slice_01, target_class_gt, initial_pred_mask_bool, segments, in_context_map, out_of_context_map, eigencam_heatmap, save_path=os.path.join(current_dir, case_name + "_overview.png"))
        

        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided', 
            model_instance=net, 
            device=device, 
            orig_image_01=image_slice_01, 
            target_class_gt=target_class_gt, 
            segments_slic=segments, 
            sps_to_perturb=ic_sps_all,           # Perturbing only the GT anatomy (in-context superpixels)
            sps_to_measure=ic_sps_all,           # Strictly tracking the GT anatomy ranks
            initial_pred_mask_bool=initial_pred_mask_bool, 
            initial_probs=initial_probs, 
            patch_size=args.img_size, 
            target_class_index=args.target_class_index, 
            ic_sps_all=ic_sps_all, 
            oc_sps_all=oc_sps_all, 
            baselines=baselines,                  # Loop over 6+2 canvases
            save_dir=current_dir         # Per-step 256x256 visuals
        )

        with open(os.path.join(current_dir, f"{case_name}_analysis.json"), 'w') as f: 
            json.dump({"prob_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Speed Single-Pass I/D Inference (MaxViT)")
    parser.add_argument("--volume_path", type=str, default="/cluster/home/bhupenk/bhupender/datasets/datasets/project_TransUNet/data/Synapse/test_vol_h5")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse")
    parser.add_argument("--model_path", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/deformableLKA/2D/MaxViT_deform_LKA.pth")
    parser.add_argument("--output_dir", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/results/dlka/prism_out_insertion_deletion")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--target_class_index", type=int,default=8)
    parser.add_argument("--img_cohort", type=str, default="/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Synapse/img_sel.txt", help="pre compiled image list.")
    parser.add_argument("--num_images", type=int, default=50, help="Target number of qualified images to find.")
    parser.add_argument("--num_superpixels", type=int, default=500)
    parser.add_argument('--target_layer_names', nargs='+', default=['last_layer'])

    args = parser.parse_args()
    main(args)
import os
import argparse
import json
import logging
import random
import math
import sys
import gc
import zlib

import numpy as np
import pandas as pd
import torch
from PIL import Image

from abc import ABC, abstractmethod

from scipy.ndimage import distance_transform_edt, center_of_mass, label, gaussian_filter
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


import random 
random_seed = 42
random.seed(random_seed)
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
# --------------------------------------------------------------------------
# --- 1. Scoring & Base Helpers
# --------------------------------------------------------------------------
def calculate_iou_score(mask1, mask2, roi_mask, label=1):
    roi_mask = roi_mask.astype(bool)
    if mask1.shape != mask2.shape:
        pil_img = Image.fromarray(mask1.astype(np.uint8) * 255)
        mask1 = np.array(pil_img.resize(mask2.shape[::-1], Image.NEAREST)) / 255
    if mask1.shape != roi_mask.shape:
        pil_img_roi = Image.fromarray(roi_mask.astype(np.uint8) * 255)
        roi_mask = (np.array(pil_img_roi.resize(mask1.shape[::-1], Image.NEAREST)) / 255).astype(bool)
        
    mask1_roi, mask2_roi = mask1[roi_mask], mask2[roi_mask]
    m1_roi, m2_roi = (mask1_roi == label), (mask2_roi == label)
    intersect, union = np.sum(m1_roi & m2_roi), np.sum(m1_roi | m2_roi)
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
            for c in range(C):
                indices = np.random.choice(len(rows), num_noise, replace=False)
                perturbed_image[rows[indices], cols[indices], c] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        target_std = 0.1 * noise_level          # Poisson std on mid-grey matches the Gaussian std
        scale = 0.5 / max(target_std, 1e-6) ** 2
        for c in range(C): perturbed_image[region_mask, c] = np.random.poisson(np.clip(perturbed_image[region_mask, c], 0, 1) * scale) / scale
    elif noise_type == 'speckle':
        for c in range(C): perturbed_image[region_mask, c] *= (1.0 + np.random.normal(0, 0.1 * noise_level, size=(H, W))[region_mask])
    elif noise_type == 'gaussian_blur':
        for c in range(C): perturbed_image[region_mask, c] = gaussian_filter(original_image_scaled_01[:, :, c], sigma=10 * noise_level)[region_mask]
    elif noise_type == 'rician':
        for c in range(C):
            px = perturbed_image[region_mask, c]
            perturbed_image[region_mask, c] = np.sqrt(np.square(px + np.random.normal(0, 0.1 * noise_level, size=px.shape)) + np.square(np.random.normal(0, 0.1 * noise_level, size=px.shape)))
            
    return np.clip(perturbed_image, 0.0, 1.0)

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

# --- NEW: Image Downsampling and Saving Helper ---
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
# --- 2. Dataset & Model Wrapper
# --------------------------------------------------------------------------
class GenericSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))
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
            cc_mask = (labeled_mask == (np.argmax(sizes) + 1))

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

            cy = int(np.clip(min_y + local_cy, 0, mask_resized.shape[0] - 1))
            cx = int(np.clip(min_x + local_cx, 0, mask_resized.shape[1] - 1))
            prompt_data = {'points': [[cx, cy]], 'labels': [1]}
            
        return {
            'image_tensor': self.transform(image), 
            'mask_tensor': self.mask_transform(mask), 
            'prompt_points': torch.tensor(prompt_data['points'], dtype=torch.float32), 
            'prompt_labels': torch.tensor(prompt_data['labels'], dtype=torch.float32), 
            'img_name': img_name
        }

def create_generic_dataloader(data_dir, batch_size=1, image_size=(1024, 512)):
    return DataLoader(GenericSegmentationDataset(root_dir=data_dir, image_size=image_size), batch_size=1, shuffle=False)

class BaseModelWrapper(ABC):
    def __init__(self, model, device): self.model, self.device = model, device
    @abstractmethod
    def predict(self, batch: dict): pass

class SAM2ModelWrapper(BaseModelWrapper):
    def predict(self, batch: dict):
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = np.ascontiguousarray((image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        H, W = image_np_rgb.shape[:2]
        
        points, labels = batch['prompt_points'].to(self.device), batch['prompt_labels'].to(self.device)
        if points.dim() == 2: points = points.unsqueeze(0)
        if labels.dim() == 1: labels = labels.unsqueeze(0)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.model.set_image(image_np_rgb)
            masks, _, logits = self.model.predict(point_coords=points, point_labels=labels, multimask_output=True)
            
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            if logits_tensor.dim() == 3: logits_tensor = logits_tensor.unsqueeze(0)
                
            logits_high_res = torch.nn.functional.interpolate(logits_tensor, size=(H, W), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits_high_res).squeeze(0).cpu().numpy()

        return masks.astype(np.uint8), probs

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> BaseModelWrapper:
    logger = logging.getLogger('base')
    logger.info(f"Loading SAM2 model...")
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    model_cfg_path = os.path.abspath(args.model_config)
    if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.dirname(model_cfg_path))
    sam_model = build_sam2(os.path.basename(model_cfg_path), args.model_path, device=device)
    return SAM2ModelWrapper(SAM2ImagePredictor(sam_model), device)

# --------------------------------------------------------------------------
# --- 3. Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    ic_sps_all, clean_ranks, initial_pred_mask, initial_probs, best_channel_index,
    baselines, img_name, save_dir
):
    """Insertion/Deletion over the IC superpixels, one superpixel per step, in clean probability-rank
    order (the allowed use of probability). Every step stores the raw record with fraction_perturbed."""
    analysis_results = {"trajectories": {}}
    to_tensor = transforms.ToTensor()

    if len(ic_sps_all) == 0: return analysis_results

    # --- PROPOSITION 4: Save Initial States and SP Map ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)

    sp_map = mark_boundaries(orig_image_01, segments_slic)
    save_downsampled_image(sp_map, os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))

    # Deletion / insertion order: clean probability rank (ic_ranks); ties: larger area first, then sp_id
    sp_areas = {int(sp): int(np.sum(segments_slic == sp)) for sp in ic_sps_all}
    order = sorted(zip(clean_ranks, ic_sps_all), key=lambda t: (t[0], -sp_areas[int(t[1])], int(t[1])))
    ranking_methods = {'prob_ranked': [int(sp) for _, sp in order]}

    # --- PROPOSITION 1: Strict One-by-One Iteration ---
    N_perturb = len(ic_sps_all)
    k_values = list(range(1, N_perturb + 1))
    clean_bg_iou = bg_iou(initial_pred_mask, gt_mask)

    def predict(img_01):
        b = batch.copy()
        b['image_tensor'] = to_tensor(img_01).unsqueeze(0).float()
        masks, probs = model_wrapper.predict(b)
        return masks[best_channel_index], probs[best_channel_index]

    def record(step_idx, fraction, img_01, pred, probs):
        ssim_score = structural_similarity(orig_image_01, img_01, data_range=1.0, channel_axis=-1)
        return step_record(step_idx, "fraction_perturbed", fraction, ssim_score, pred, probs, gt_mask,
                           segments_slic, ic_sps_all, clean_ranks, clean_bg_iou)

    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_01, baseline, img_name)

        # Save Initial State for Insertion specific to this baseline
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        pred_n, prob_n = predict(bg_canvas)

        for r_name, r_ids in ranking_methods.items():
            # Step 0: deletion starts from the clean image, insertion from the full canvas
            traj = {
                "deletion": [step_record(0, "fraction_perturbed", 0.0, 1.0, initial_pred_mask, initial_probs, gt_mask,
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

                # Deletion Step
                img_del = np.where(mask_3d, bg_canvas, orig_image_01)
                pred_del, prob_del = predict(img_del)
                traj["deletion"].append(record(k, fraction, img_del, pred_del, prob_del))

                # Insertion Step
                img_ins = np.where(mask_3d, orig_image_01, bg_canvas)
                pred_ins, prob_ins = predict(img_ins)
                traj["insertion"].append(record(k, fraction, img_ins, pred_ins, prob_ins))

                # --- PROPOSITION 3: Save the perturbed inputs and predicted masks ---
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
# --- 4. Main Controller
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (SAMv2)")
    parser.add_argument('--data_dir', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_config', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--output_dir', type=str, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/results/sam2/prism_out_insertion_deletion')
    parser.add_argument('--txtfile', type=str, default='./img_sel.txt')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--target_class', type=str, default='person')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_wrapper = get_model_wrapper(args.model_name, args, device)
    val_loader = create_generic_dataloader(data_dir=os.path.join(args.data_dir, args.target_class), image_size=tuple(args.image_size))
   
    baselines = ['black', 'mean', 'random']

    logger.info(f"\n[PHASE 1] Strict Anatomical Qualifier Scan...")
    with open(args.txtfile, 'r') as f: frozen_image_names = set([line.strip() for line in f.readlines()])

    qualified_images = []
    for batch in tqdm(val_loader):
        if len(qualified_images) >= args.num_images: break
        img_name = batch['img_name'][0]
        if img_name not in frozen_image_names: continue
            
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()

        all_pred_masks, all_probs = model_wrapper.predict(batch)
        iou_scores = [calculate_iou_score(mask, gt_mask, np.ones_like(gt_mask, dtype=bool)) for mask in all_pred_masks]
        best_idx = np.argmax(iou_scores)
        
        segments_slic = slic(orig_image_01, n_segments=100, compactness=10, sigma=1, channel_axis=-1)
        
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask, ic_threshold=1.0, oc_threshold=0.00)
        if not ic_sps_all: continue

        qualified_images.append({
            'batch': batch, 'img_name': img_name, 'orig_image_01': orig_image_01, 'gt_mask': gt_mask,
            'segments_slic': segments_slic, 'pred_mask': all_pred_masks[best_idx], 'clean_probs': all_probs[best_idx], 
            'best_idx': best_idx, 'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all
        })

    logger.info(f"\n[PHASE 2] Executing IC-Restricted Prob-Rank I/D Trajectories over {len(baselines)} Baselines...")

    os.makedirs(args.output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images):
        stem = q_data['img_name'].rsplit('.', 1)[0]
        current_dir = os.path.join(args.output_dir, stem)
        os.makedirs(current_dir, exist_ok=True)

        # Clean ranking of the IC superpixels, computed once per image and used for every step
        clean_ranks, clean_means = ic_ranks(q_data['clean_probs'], q_data['segments_slic'], q_data['ic_sps_all'])

        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided',
            model_wrapper=model_wrapper,
            batch=q_data['batch'],
            orig_image_01=q_data['orig_image_01'],
            gt_mask=q_data['gt_mask'],
            segments_slic=q_data['segments_slic'],
            ic_sps_all=q_data['ic_sps_all'],
            clean_ranks=clean_ranks,
            initial_pred_mask=q_data['pred_mask'],
            initial_probs=q_data['clean_probs'],
            best_channel_index=q_data['best_idx'],
            baselines=baselines,
            img_name=stem,
            save_dir=current_dir  # Passing the directory down for image saving
        )

        result = {
            "image_info": {"name": stem},
            "clean_ranking": clean_ranking_record(q_data['segments_slic'], q_data['ic_sps_all'], clean_ranks, clean_means),
            "prob_guided_analysis": res_global,
        }
        with open(os.path.join(current_dir, f"{stem}_analysis.json"), 'w') as f:
            json.dump(result, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
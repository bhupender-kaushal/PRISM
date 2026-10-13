import os
import argparse
import json
import logging
import random
import math
import sys
import gc

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

from sklearn.metrics import auc

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
            indices = np.random.choice(len(rows), num_noise, replace=False)
            for c in range(C): perturbed_image[rows[indices], cols[indices], c] = np.random.choice([0.0, 1.0], num_noise)
    elif noise_type == 'poisson':
        scale = max(0.05, 3.2 * noise_level) * 300.0
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

def generate_baseline_canvas(orig_image_01, baseline_type):
    if baseline_type == 'black':
        return np.zeros_like(orig_image_01)
    elif baseline_type == 'mean':
        mean_color = np.mean(orig_image_01, axis=(0, 1), keepdims=True)
        return np.ones_like(orig_image_01) * mean_color
    else:
        dummy_all_mask = np.ones(orig_image_01.shape[:2], dtype=int)
        return perturb_superpixels_with_level(orig_image_01, dummy_all_mask, [1], baseline_type, noise_level=30.0)

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
            min_y, min_x = np.argwhere(cc_mask).min(axis=0)
            cy, cx = center_of_mass(cc_mask)
            
            cy = int(np.clip(cy, 0, mask_resized.shape[0] - 1))
            cx = int(np.clip(cx, 0, mask_resized.shape[1] - 1))
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
    sps_to_perturb, sps_to_measure, initial_pred_mask, initial_probs, best_channel_index,
    ic_sps_all, oc_sps_all, baselines, save_dir
):
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    to_tensor = transforms.ToTensor()
    
    if len(sps_to_perturb) == 0 or len(sps_to_measure) == 0: return analysis_results
    
    max_id = int(np.max(segments_slic)) + 1
    valid_sp_to_measure = [int(sp) for sp in sps_to_measure if int(sp) < max_id]
    sp_areas = {int(sp_id): int(np.sum(segments_slic == sp_id)) for sp_id in np.unique(segments_slic)}
    
    # --- PROPOSITION 4: Save Initial States and SP Map ---
    vis_dir_root = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir_root, exist_ok=True)
    
    sp_map = mark_boundaries(orig_image_01, segments_slic)
    save_downsampled_image(sp_map, os.path.join(vis_dir_root, "superpixel_map.jpg"))
    save_downsampled_image(orig_image_01, os.path.join(vis_dir_root, "initial_deletion_start.jpg"))
    
    # 1. Base Ranking (Probabilities)
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
        
    rank_data.sort(key=lambda x: (x['mean_prob'], x['area']), reverse=True)
    for rank, item in enumerate(rank_data, 1):
        item['prob_rank'] = rank
    prob_ranked_ids = [item['sp_id'] for item in rank_data]
        
    # --- PROPOSITION 2: Save SP Ranks (Stored seamlessly in the JSON output) ---
    analysis_results["rank_distribution"] = rank_data
    ranking_methods = {'prob_ranked': prob_ranked_ids}
    
    # --- PROPOSITION 1: Strict One-by-One Iteration ---
    N_perturb = len(sps_to_perturb)
    k_values = list(range(1, N_perturb + 1))  # Replaced linspace with 1-by-1 stepping
    
    bg_gt = (gt_mask == 0)

    def get_step_superpixels(current_pred_mask, current_probs):
        intersect_mask = (current_pred_mask == 1) & (gt_mask > 0)
        union_mask = (current_pred_mask == 1) | (gt_mask > 0)
        
        intersect_counts = np.bincount(segments_slic[intersect_mask].ravel(), minlength=max_id)
        union_counts = np.bincount(segments_slic[union_mask].ravel(), minlength=max_id)
        
        step_sps = []
        for sp_id in valid_sp_to_measure:
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
        return step_sps

    def create_entry(frac, global_iou, spill, mean_f_prob, step_sps): 
        return {
            "fraction_perturbed": frac, 
            "global_iou": round(float(global_iou), 2), 
            "spillover_ratio": round(float(spill), 4), 
            "mean_foreground_prob": round(float(mean_f_prob), 4),
            "superpixels": step_sps
        }

    def calculate_auc(traj):
        if len(traj) < 2: return 0.0
        x = [e['fraction_perturbed'] for e in traj]
        y = [e['global_iou'] / 100.0 for e in traj]
        return float(auc(x, y))

    base_sps = get_step_superpixels(initial_pred_mask, initial_probs)

    for baseline in baselines:
        analysis_results["trajectories"][baseline] = {}
        bg_canvas = generate_baseline_canvas(orig_image_01, baseline)
        
        # Save Initial State for Insertion specific to this baseline
        save_downsampled_image(bg_canvas, os.path.join(vis_dir_root, f"initial_insertion_start_{baseline}.jpg"))
        
        base_intersect = np.sum((initial_pred_mask == 1) & (gt_mask > 0))
        base_union = np.sum((initial_pred_mask == 1) | (gt_mask > 0))
        base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0
        base_f_prob = float(np.mean(initial_probs[initial_pred_mask == 1])) if np.any(initial_pred_mask == 1) else 0.0

        b_noise = batch.copy()
        b_noise['image_tensor'] = to_tensor(bg_canvas).unsqueeze(0).float()
        masks_n, probs_n = model_wrapper.predict(b_noise)
        pred_n, prob_n = masks_n[best_channel_index], probs_n[best_channel_index]
        
        ins_intersect0 = np.sum((pred_n == 1) & (gt_mask > 0))
        ins_union0 = np.sum((pred_n == 1) | (gt_mask > 0))
        ins_base_iou = (ins_intersect0 / ins_union0 * 100.0) if ins_union0 > 0 else 100.0
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

                # Deletion Step
                img_del = np.where(mask_3d, bg_canvas, orig_image_01)
                b_del = batch.copy()
                b_del['image_tensor'] = to_tensor(img_del).unsqueeze(0).float()
                m_del, p_del = model_wrapper.predict(b_del)
                pred_del, prob_del = m_del[best_channel_index], p_del[best_channel_index]
                
                iou_del = (np.sum((pred_del == 1) & (gt_mask > 0)) / max(1, np.sum((pred_del == 1) | (gt_mask > 0)))) * 100.0
                spill_del = np.sum((pred_del == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_del = float(np.mean(prob_del[pred_del == 1])) if np.any(pred_del == 1) else 0.0
                del_sps = get_step_superpixels(pred_del, prob_del)
                traj["deletion"].append(create_entry(fraction, iou_del, spill_del, prob_f_del, del_sps))

                # Insertion Step
                img_ins = np.where(mask_3d, orig_image_01, bg_canvas)
                b_ins = batch.copy()
                b_ins['image_tensor'] = to_tensor(img_ins).unsqueeze(0).float()
                m_ins, p_ins = model_wrapper.predict(b_ins)
                pred_ins, prob_ins = m_ins[best_channel_index], p_ins[best_channel_index]
                
                iou_ins = (np.sum((pred_ins == 1) & (gt_mask > 0)) / max(1, np.sum((pred_ins == 1) | (gt_mask > 0)))) * 100.0
                spill_ins = np.sum((pred_ins == 1) & bg_gt) / max(1, np.sum(bg_gt))
                prob_f_ins = float(np.mean(prob_ins[pred_ins == 1])) if np.any(pred_ins == 1) else 0.0
                ins_sps = get_step_superpixels(pred_ins, prob_ins)
                traj["insertion"].append(create_entry(fraction, iou_ins, spill_ins, prob_f_ins, ins_sps))

                # --- PROPOSITION 3: Save the perturbed inputs and predicted masks ---
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
# --- 4. Main Controller
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PRISM Global XAI Robustness Engine (SAMv2)")
    parser.add_argument('--data_dir', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam2_hiera_large')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_config', type=str, required=False, default='/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/SAM2/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--output_dir', type=str, default='./prism_out_insertion_deletion')
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
   
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

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

    logger.info(f"\n[PHASE 2] Executing IC-Restricted Dual-Rank I/D Trajectories over 8 Baselines...")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images):
        current_dir = os.path.join(args.output_dir, q_data['img_name'].rsplit('.', 1)[0])
        os.makedirs(current_dir, exist_ok=True)
        
        res_global = run_dual_trajectory_analysis(
            domain_name='prob_guided', 
            model_wrapper=model_wrapper, 
            batch=q_data['batch'], 
            orig_image_01=q_data['orig_image_01'], 
            gt_mask=q_data['gt_mask'], 
            segments_slic=q_data['segments_slic'], 
            sps_to_perturb=q_data['ic_sps_all'], 
            sps_to_measure=q_data['ic_sps_all'], 
            initial_pred_mask=q_data['pred_mask'], 
            initial_probs=q_data['clean_probs'], 
            best_channel_index=q_data['best_idx'], 
            ic_sps_all=q_data['ic_sps_all'], 
            oc_sps_all=q_data['oc_sps_all'], 
            baselines=baselines,
            save_dir=current_dir  # Passing the directory down for image saving
        )

        with open(os.path.join(current_dir, f"{q_data['img_name'].rsplit('.', 1)[0]}_analysis.json"), 'w') as f: 
            json.dump({"prob_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
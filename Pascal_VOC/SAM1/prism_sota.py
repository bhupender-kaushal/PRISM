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

from scipy.ndimage import distance_transform_edt, center_of_mass, label
from skimage.metrics import structural_similarity
from skimage.segmentation import slic, mark_boundaries
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'segment_anything' library.")
    sys.exit()

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

def generate_dynamic_steps(N, max_steps=20):
    if N == 0: return []
    if N <= max_steps: return list(range(1, N + 1))
    return sorted(list(set(np.linspace(1, N, max_steps).astype(int))))

def perturb_superpixels_with_level(original_image_scaled_01, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """Applies strict RGB channel-wise noise."""
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
        scale = max(0.05, 3.2 - noise_level) * 300.0
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

# --------------------------------------------------------------------------
# --- 2. Dataset, Model Wrapper & EigenCAM
# --------------------------------------------------------------------------
class GenericSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(image_size, interpolation=Image.NEAREST), transforms.ToTensor()])
    
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

class SAM1ModelWrapper(BaseModelWrapper):
    def predict(self, batch: dict):
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        H, W = image_np_rgb.shape[:2]
        
        points_np = batch['prompt_points'].squeeze(0).cpu().numpy()
        labels_np = batch['prompt_labels'].squeeze(0).cpu().numpy()

        with torch.no_grad(), torch.cuda.amp.autocast():
            try: self.model.reset_image() 
            except AttributeError: pass 
                
            self.model.set_image(image_np_rgb)
            masks, _, logits = self.model.predict(point_coords=points_np, point_labels=labels_np, multimask_output=True)
            
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            if logits_tensor.dim() == 3: logits_tensor = logits_tensor.unsqueeze(0)
                
            logits_high_res = torch.nn.functional.interpolate(logits_tensor, size=(H, W), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits_high_res).squeeze(0).cpu().numpy()
            
        return masks.astype(np.uint8), probs

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> BaseModelWrapper:
    logger = logging.getLogger('base')
    logger.info(f"Loading SAMv1 model: type='{args.model_type}', checkpoint='{args.model_path}'")
    
    if model_name.startswith('sam_'):
        sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
        sam.to(device=device)
        return SAM1ModelWrapper(SamPredictor(sam), device)
    else:
        raise ValueError("Model type not supported.")

def calculate_eigencam_sam1(model_wrapper, batch, image_size):
    """Safely extracts EigenCAM from SAM1's image encoder."""
    feature_maps = []
    
    try: target_layer = model_wrapper.model.model.image_encoder.blocks[-1]
    except AttributeError: return np.zeros(image_size, dtype=np.float32)

    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    _ = model_wrapper.predict(batch) 
    hook.remove()

    if not feature_maps: return np.zeros(image_size, dtype=np.float32)

    features = feature_maps[0].squeeze(0).to(torch.float32) 
    if features.dim() == 3: 
        H, W, C = features.shape
        features = features.permute(2, 0, 1) 
    
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
    return torch.nn.functional.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()


# --------------------------------------------------------------------------
# --- 3. Global EigenCAM Causal Trajectory Extraction Engine
# --------------------------------------------------------------------------
def run_dual_trajectory_analysis(
    domain_name, model_wrapper, batch, orig_image_01, gt_mask, segments_slic,
    sps_to_perturb, sps_to_measure, initial_pred_mask, initial_probs, eigencam_heatmap, best_channel_index,
    ic_sps_all, oc_sps_all, baselines
):
    """Executes Insertion/Deletion ranked by Probability AND EigenCAM, saving step-wise superpixel metrics on GT."""
    analysis_results = {
        "rank_distribution": [],
        "trajectories": {}
    }
    to_tensor = transforms.ToTensor()
    
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
    bg_gt = (gt_mask == 0)

    # Function to get individual GT superpixel tracking at each step
    def get_step_superpixels(current_pred_mask, current_probs):
        intersect_mask = (current_pred_mask == 1) & (gt_mask > 0)
        union_mask = (current_pred_mask == 1) | (gt_mask > 0)
        
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
        bg_canvas = generate_baseline_canvas(orig_image_01, baseline)
        
        # Base Deletion Metrics
        base_intersect = np.sum((initial_pred_mask == 1) & (gt_mask > 0))
        base_union = np.sum((initial_pred_mask == 1) | (gt_mask > 0))
        base_global_iou = (base_intersect / base_union * 100.0) if base_union > 0 else 100.0
        base_f_prob = float(np.mean(initial_probs[initial_pred_mask == 1])) if np.any(initial_pred_mask == 1) else 0.0

        # Base Insertion Metrics (Fully destroyed image)
        b_noise = batch.copy()
        b_noise['image_tensor'] = to_tensor(bg_canvas).unsqueeze(0).float()
        masks_n, probs_n = model_wrapper.predict(b_noise)
        pred_n, prob_n = masks_n[best_channel_index], probs_n[best_channel_index]
        
        ins_union0 = np.sum((pred_n == 1) | (gt_mask > 0))
        ins_base_iou = (np.sum((pred_n == 1) & (gt_mask > 0)) / ins_union0 * 100.0) if ins_union0 > 0 else 100.0
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

                # --- 2. Insertion Step ---
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
    parser = argparse.ArgumentParser(description="PRISM Dual-Rank I/D Robustness Engine (SAMv1)")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final')
    parser.add_argument('--model_name', type=str, default='sam_vit_h')
    parser.add_argument('--model_type', type=str, default='vit_h')
    parser.add_argument('--model_path', type=str, required=False, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--output_dir', type=str, default='./prism_out_insertion_deletion')
    parser.add_argument('--txtfile', type=str, default='/bhupender/2025/Mouse_other/img_sel.txt')
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
   
    # 6+2 Baselines
    baselines = ['black', 'mean', 'poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']

    # --- PHASE 1: Absolute Qualifiers ---
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
        
        eigencam_heatmap = calculate_eigencam_sam1(model_wrapper, batch, (orig_image_01.shape[0], orig_image_01.shape[1]))
        segments_slic = slic(orig_image_01, n_segments=500, compactness=10, sigma=1, channel_axis=-1)
        
        ic_sps_all, oc_sps_all = get_strict_ic_oc_superpixels(segments_slic, gt_mask, ic_threshold=1.0, oc_threshold=0.00)
        if not ic_sps_all: continue

        qualified_images.append({
            'batch': batch, 'img_name': img_name, 'orig_image_01': orig_image_01, 'gt_mask': gt_mask,
            'segments_slic': segments_slic, 'pred_mask': all_pred_masks[best_idx], 'clean_probs': all_probs[best_idx], 
            'eigencam_heatmap': eigencam_heatmap, 'best_idx': best_idx, 'ic_sps_all': ic_sps_all, 'oc_sps_all': oc_sps_all
        })

    # --- PHASE 2: Global XAI Trajectory Run ---
    logger.info(f"\n[PHASE 2] Executing Global Dual-Rank I/D Trajectories over 8 Baselines...")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for q_data in tqdm(qualified_images):
        current_dir = os.path.join(args.output_dir, q_data['img_name'].rsplit('.', 1)[0])
        os.makedirs(current_dir, exist_ok=True)
        
        all_sps = np.unique(q_data['segments_slic'])
        all_sps = all_sps[all_sps != 0]
        
        # Unified Global Execution
        res_global = run_dual_trajectory_analysis(
            domain_name='global_eigen_guided', 
            model_wrapper=model_wrapper, 
            batch=q_data['batch'], 
            orig_image_01=q_data['orig_image_01'], 
            gt_mask=q_data['gt_mask'], 
            segments_slic=q_data['segments_slic'], 
            sps_to_perturb=all_sps,              # Perturbing the entire image based on Eigen Rank
            sps_to_measure=q_data['ic_sps_all'], # Strictly tracking the GT anatomy ranks
            initial_pred_mask=q_data['pred_mask'], 
            initial_probs=q_data['clean_probs'], 
            eigencam_heatmap=q_data['eigencam_heatmap'], 
            best_channel_index=q_data['best_idx'], 
            ic_sps_all=q_data['ic_sps_all'], 
            oc_sps_all=q_data['oc_sps_all'], 
            baselines=baselines                  # Loop over 6+2 canvases
        )

        with open(os.path.join(current_dir, f"{q_data['img_name'].rsplit('.', 1)[0]}_analysis.json"), 'w') as f: 
            json.dump({"eigen_guided_analysis": res_global}, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
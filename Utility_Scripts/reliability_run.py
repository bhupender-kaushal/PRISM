# run_sam2_reliability.py

# --------------------------------------------------------------------------
# --- 1. Imports and Dependencies
# --------------------------------------------------------------------------
# This script requires the following libraries:
# pip install torch torchvision numpy scipy scikit-image scikit-learn pandas seaborn matplotlib tqdm
#
# You must also have the SAM-2 library accessible.
# 1. Clone the repository: git clone https://github.com/facebookresearch/sam2.git
# 2. Add it to your Python path if necessary, e.g., by running this script from the same parent directory.
# --------------------------------------------------------------------------

import os
import argparse
import json
import logging
import random
from abc import ABC, abstractmethod
from math import exp
import sys



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from traitlets import default
from scipy.ndimage import gaussian_filter
from skimage.color import gray2rgb
from skimage.metrics import structural_similarity
from skimage.segmentation import mark_boundaries, slic
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import center_of_mass

# Import the specific, up-to-date functions for SAM-2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("FATAL ERROR: Could not import from the 'sam2' library.")
    print("Please ensure you have cloned the 'sam2' repository from GitHub")
    print("and that its location is accessible to your Python environment.")
    exit()

# --------------------------------------------------------------------------
# --- 2. Scoring and Helper Functions
# --------------------------------------------------------------------------

def calculate_iou_score(mask1, mask2, roi_mask, label=1):
    """Calculates the Intersection over Union (IoU) within a given Region of Interest (ROI)."""
    roi_mask = roi_mask.astype(bool)
    if mask1.shape != mask2.shape:
        pil_img = Image.fromarray(mask1.astype(np.uint8) * 255)
        resized_img = pil_img.resize(mask2.shape[::-1], Image.NEAREST)
        mask1 = np.array(resized_img) / 255
    if mask1.shape != roi_mask.shape:
        pil_img_roi = Image.fromarray(roi_mask.astype(np.uint8) * 255)
        resized_img_roi = pil_img_roi.resize(mask1.shape[::-1], Image.NEAREST)
        roi_mask = (np.array(resized_img_roi) / 255).astype(bool)
    mask1_roi, mask2_roi = mask1[roi_mask], mask2[roi_mask]
    m1_roi, m2_roi = (mask1_roi == label), (mask2_roi == label)
    intersect = np.sum(m1_roi & m2_roi)
    union = np.sum(m1_roi | m2_roi)
    if union == 0: return 100.0
    return (float(intersect) / float(union)) * 100

def select_superpixels_by_gt_mask(superpixels_segments, gt_mask):
    """Selects all superpixels that have any overlap with the ground truth mask."""
    gt_mask_bool = gt_mask > 0
    overlapping_labels = np.unique(superpixels_segments[gt_mask_bool])
    return [label for label in overlapping_labels if label != 0]

def perturb_superpixels_with_level(original_image_scaled_01, superpixels_segments, selected_superpixel_ids, noise_type, noise_level):
    """Perturbs image regions (in [0,1] range) and returns the perturbed image."""
    perturbed_image = np.copy(original_image_scaled_01)
    for sp_id in selected_superpixel_ids:
        superpixel_mask = (superpixels_segments == sp_id)
        # Ensure mask is broadcastable for 3-channel images
        superpixel_mask_3c = np.stack([superpixel_mask]*3, axis=-1)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.1 * noise_level, perturbed_image.shape)
            perturbed_image[superpixel_mask_3c] += noise[superpixel_mask_3c]
        elif noise_type == 'salt_pepper':
            for c in range(3):
                channel_mask = superpixel_mask
                num_pixels = np.sum(channel_mask)
                num_noise = int(min(0.5, 0.05 * noise_level) * num_pixels)
                if num_noise > 0:
                    rows, cols = np.where(channel_mask)
                    indices = random.sample(range(len(rows)), num_noise)
                    for i in indices:
                        perturbed_image[rows[i], cols[i], c] = random.choice([0.0, 1.0])
                        
    return np.clip(perturbed_image, 0.0, 1.0)



def visualize_segmentation_comparison(orig_image, gt_mask, pred_mask, eigencam_heatmap, prompt_points=None, save_path=None):
    """Visualizes the original image, GT, prediction, EigenCAM, and overlays prompt points."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    # Original Image with Prompt Points
    axes[0].imshow(orig_image)
    if prompt_points is not None:
        for point in prompt_points:
            y, x = point  # Assuming (y, x) format
            axes[0].scatter(x, y, c='red', s=120, marker='*', edgecolors='black', linewidths=1.5, label='Prompt Point')
    axes[0].set_title("Original Image with Prompts")
    axes[0].axis('off')
    
    # Ground Truth Mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    # Predicted Mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    # EigenCAM Heatmap
    axes[3].imshow(orig_image)
    im = axes[3].imshow(eigencam_heatmap, cmap='jet', alpha=0.5)
    axes[3].set_title("EigenCAM Saliency")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
# --------------------------------------------------------------------------
# --- 3. Generic Data Loading and Model Wrappers
# --------------------------------------------------------------------------

class GenericSegmentationDataset(Dataset):
    """A generic dataset that loads images, masks, and point prompts."""
    def __init__(self, root_dir, image_size=(1024, 512)):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        
        with open(os.path.join(root_dir, 'prompts.json')) as f:
            self.prompts = json.load(f)
            
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)
    
    
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        prompt_data = self.prompts[img_name]
        # Resize mask to image_size and compute center of mass for prompt
        mask_resized = self.mask_transform(mask).squeeze().numpy()
        if np.sum(mask_resized) > 0:
            cy, cx = center_of_mass(mask_resized)
            prompt_data = {
            'points': [[cy, cx]],
            'labels': [1]
            }
        else:
            # fallback: use original prompt if mask is empty
            prompt_data = self.prompts[img_name]

        return {
            'image_tensor': self.transform(image),
            'mask_tensor': self.mask_transform(mask),
            'path': img_path,
            'prompt_points': torch.tensor(prompt_data['points'], dtype=torch.float32),
            'prompt_labels': torch.tensor(prompt_data['labels'], dtype=torch.float32),
            'img_name': img_name
        }

def create_generic_dataloader(data_dir, batch_size=1, image_size=(1024, 512)):
    dataset = GenericSegmentationDataset(root_dir=data_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

class BaseModelWrapper(ABC):
    """Abstract Base Class defines the contract for any model to be used in the pipeline."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def predict(self, batch: dict) -> np.ndarray:
        """Takes a batch dict and returns a single binary numpy mask."""
        pass

    @abstractmethod
    def get_target_layer(self, name: str) -> nn.Module:
        """Takes a string name and returns the corresponding model layer."""
        pass

class SAM2ModelWrapper(BaseModelWrapper):
    """The adapter for the SAM-2 model, implementing the contract."""


    def predict(self, batch: dict) -> np.ndarray:
        image_tensor = batch['image_tensor'].squeeze(0)
        image_np_rgb = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # reversed channels produce a negative-stride view; make a contiguous copy to avoid the torchvision error
        image_np_bgr = np.ascontiguousarray(image_np_rgb[:, :, ::-1])

        # handle collated prompts from DataLoader (batch dim may be present)
        points = batch['prompt_points']
        labels = batch['prompt_labels']

        # remove the extra leading batch dim if dataloader added it (shape: 1, N, 2) -> (N, 2)
        if isinstance(points, torch.Tensor) and points.dim() == 3 and points.shape[0] == 1:
            points = points.squeeze(0)
        if isinstance(labels, torch.Tensor) and labels.dim() == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)

        points = points.to(self.device)
        labels = labels.to(self.device)

        # ensure predictor receives (B, N, 2) and (B, N)
        points_in = points.unsqueeze(0) if points.dim() == 2 else points
        labels_in = labels.unsqueeze(0) if labels.dim() == 1 else labels

        # Use autocast for performance, as learned from the official notebook
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.set_image(image_np_bgr)
            masks, scores, _ = self.model.predict(
                point_coords=points_in,
                point_labels=labels_in,
                multimask_output=True,
            )
        print(f'printing mask shape from SAM2MODELWRAPPER{masks.shape}')
        # Select the mask with the maximum number of pixels (largest area)
        mask_areas = [np.sum(mask) for mask in masks]
        best_idx = int(np.argmax(mask_areas))
        print(f"Mask areas: {mask_areas}")
        print(f"Selected mask index: {best_idx} with area: {mask_areas[best_idx]}")
        masks = masks[best_idx]
        return masks.squeeze().astype(np.uint8)

    # def get_target_layer(self, name: str) -> nn.Module:
    #     # The actual nn.Module is inside the predictor's .model attribute
    #     curr_obj = self.model.model.image_encoder
    #     for part in name.split('.'):
    #         curr_obj = getattr(curr_obj, part)
    #     return curr_obj



    def get_target_layer(self, name: str) -> nn.Module:
        # The actual nn.Module is inside the predictor's .model attribute
        base = self.model.model.image_encoder
        # 1) Try dotted attribute path (e.g. "blocks.31.norm1")
        parts = name.split('.')
        try:
            curr_obj = base
            for part in parts:
                curr_obj = getattr(curr_obj, part)
            return curr_obj
        except Exception:
            # 1b) If user asked for the last conv layer, find last nn.Conv2d in the model
            if name.lower() in ('last_conv', 'last_conv2d', 'last_conv_layer', 'conv_last'):
                last_conv = None
                last_name = None
                for full_name, module in self.model.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                        last_name = full_name
                if last_conv is not None:
                    print(f"[get_target_layer] Returning last Conv2d module found: {last_name}")
                    return last_conv

            # 2) Fallback: search all named modules for a matching suffix or substring
            for full_name, module in self.model.model.named_modules():
                if full_name.endswith(name) or name in full_name:
                    print(f"[get_target_layer] Returning module by named_modules search: {full_name} -> {module}")
                    return module
            # 3) Nothing found -> raise with helpful info
            top_level = [n for n, _ in self.model.model.named_children()]
            raise AttributeError(f"Could not find target layer '{name}'. Top-level children of predictor.model: {top_level}. "
                                f"Use 'last_conv' to automatically select the final Conv2d, or list available modules:\n"
                                f"  python - <<'PY'\n"
                                f"from sam2.build_sam import build_sam2\n"
                                f"from sam2.sam2_image_predictor import SAM2ImagePredictor\n"
                                f"m = build_sam2('{os.path.basename(args.model_config)}', '{args.model_path}', device='cpu')\n"
                                f"p = SAM2ImagePredictor(m)\n"
                                f"print('\\n'.join([n for n,_ in p.model.named_modules()]))\n"
                                f"PY")

def get_model_wrapper(model_name: str, args: argparse.Namespace, device: torch.device) -> BaseModelWrapper:
    """Factory function to build and wrap the specified model."""
    logger = logging.getLogger('base')
    logger.info(f"Loading model: {model_name} from config: {args.model_config}")
    if model_name.startswith('sam2_'):
        model_cfg_path = os.path.abspath(args.model_config)
        cfg_dir = os.path.dirname(model_cfg_path)
        cfg_name = os.path.basename(model_cfg_path)

        # initialize Hydra search path to include the sam2 configs directory so sam2 can compose the config
        try:

            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            from hydra import initialize_config_dir, compose
            # initialize_config_dir accepts an absolute path (unlike initialize(config_path=...))

            initialize_config_dir(config_dir=cfg_dir)
            # Compose once with the basename to validate the config is visible in the search path.
            _ = compose(config_name=cfg_name)
            # Pass the basename to build_sam2 so it composes using the initialized search path
            sam_model = build_sam2(cfg_name, args.model_path, device=device)
        except Exception as e:
            logger.warning(f"Hydra init/compose warning: {e} -- falling back to passing absolute path to build_sam2")
            # fallback: pass absolute path (may still fail if build_sam2 uses compose internally)
            sam_model = build_sam2(model_cfg_path, args.model_path, device=device)

        predictor = SAM2ImagePredictor(sam_model)
        return SAM2ModelWrapper(predictor, device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
# ...existing code...
# ...existing code...

# --------------------------------------------------------------------------
# --- 4. Core Analysis and Visualization
# --------------------------------------------------------------------------
# def calculate_eigencam(model_wrapper: BaseModelWrapper, target_layer_name: str, batch: dict, image_size: tuple) -> np.ndarray:
#     feature_maps = []
#     logger = logging.getLogger('base')
#     try:
#         target_layer = model_wrapper.get_target_layer(target_layer_name)
#     except Exception as e:
#         logger.error(f"Could not find target layer '{target_layer_name}'. Error: {e}")
#         return np.zeros(image_size, dtype=np.float32)

#     def hook_fn(module, input, output): feature_maps.append(output)
#     hook = target_layer.register_forward_hook(hook_fn)
#     _ = model_wrapper.predict(batch) # This forward pass triggers the hook
#     hook.remove()

#     if not feature_maps:
#         logger.error("EigenCAM hook did not capture any feature maps."); return np.zeros(image_size, dtype=np.float32)

#     features = feature_maps[0].squeeze(0)
#     features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
#     pca = PCA(n_components=1).fit(features_flat)
#     saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
#     saliency_map = np.maximum(saliency_map, 0)
#     s_min, s_max = saliency_map.min(), saliency_map.max()
#     saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)
#     saliency_tensor = torch.from_numpy(saliency_map)[None, None, ...].float()
#     return F.interpolate(saliency_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze().numpy()

def calculate_eigencam(model_wrapper: BaseModelWrapper, target_layer_name: str, batch: dict, image_size: tuple) -> np.ndarray:
    feature_maps = []
    logger = logging.getLogger('base')
    try:
        target_layer = model_wrapper.get_target_layer(target_layer_name)
    except Exception as e:
        logger.error(f"Could not find target layer '{target_layer_name}'. Error: {e}")
        return np.zeros(image_size[:2] if len(image_size) >= 2 else image_size, dtype=np.float32)

    def hook_fn(module, input, output): feature_maps.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    _ = model_wrapper.predict(batch) # This forward pass triggers the hook
    hook.remove()

    if not feature_maps:
        logger.error("EigenCAM hook did not capture any feature maps."); return np.zeros(image_size[:2] if len(image_size) >= 2 else image_size, dtype=np.float32)

    features = feature_maps[0].squeeze(0).to(torch.float32)  # (C, Hf, Wf)
    features_flat = features.view(features.shape[0], -1).permute(1, 0).cpu().numpy()
    pca = PCA(n_components=3).fit(features_flat)
    saliency_map = (features.cpu().numpy() * pca.components_[0][:, None, None]).sum(axis=0)
    saliency_map = np.maximum(saliency_map, 0)
    s_min, s_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(saliency_map)

    # ensure image_size is (H, W) (user may have passed (H, W, C))
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        out_size = (int(image_size[0]), int(image_size[1]))
    else:
        out_size = tuple(image_size)

    saliency_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float()  # (1,1,Hf,Wf)
    # interpolate expects output size as (H, W)
    saliency_resized = F.interpolate(saliency_tensor, size=out_size, mode='bilinear', align_corners=False)
    return saliency_resized.squeeze().cpu().numpy()


def run_reliability_analysis(model_wrapper, batch, orig_image_01, gt_mask, segments_slic, superpixels_to_measure, noise_types):
    """Runs the full ranking analysis by perturbing the object and measuring IoU changes."""
    logger = logging.getLogger('base')
    analysis_results = {}
    to_tensor_transform = transforms.ToTensor()

    for noise_type in noise_types:
        logger.info(f"--- Analyzing Noise Type: {noise_type.upper()} ---")
        sp_iou_history = {sp_id: [] for sp_id in superpixels_to_measure}
        sp_rank_history = {sp_id: [] for sp_id in superpixels_to_measure}
        
        for noise_level in np.arange(0.1, 1.1, 0.2): # Reduced range for speed
            perturbed_image_01 = perturb_superpixels_with_level(orig_image_01, segments_slic, superpixels_to_measure, noise_type, noise_level)
            
            perturbed_batch = batch.copy()
            perturbed_batch['image_tensor'] = to_tensor_transform(perturbed_image_01).unsqueeze(0).float()
            perturbed_pred_mask = model_wrapper.predict(perturbed_batch)

            current_level_sp_data = [{"sp_id": int(sp_id), "iou": round(calculate_iou_score(perturbed_pred_mask, gt_mask, segments_slic == sp_id), 2)} for sp_id in superpixels_to_measure]
            sorted_superpixels = sorted(current_level_sp_data, key=lambda x: x['iou'], reverse=True)
            current_ranks_map = {data['sp_id']: rank for rank, data in enumerate(sorted_superpixels, 1)}
            for sp_id in superpixels_to_measure:
                sp_iou_history[sp_id].append(next((item['iou'] for item in current_level_sp_data if item['sp_id'] == sp_id), 0))
                sp_rank_history[sp_id].append(current_ranks_map.get(sp_id))

        reliability_results = []
        for sp_id in superpixels_to_measure:
            avg_iou = np.mean(sp_iou_history[sp_id]) if sp_iou_history[sp_id] else 0
            rank_list = [r for r in sp_rank_history[sp_id] if r is not None]
            avg_rank_jump = (sum(abs(rank_list[i] - rank_list[i-1]) for i in range(1, len(rank_list))) / (len(rank_list) - 1)) if len(rank_list) > 1 else 0
            score = (avg_iou / exp(avg_rank_jump / 50.0)) if avg_iou > 0 else 0
            reliability_results.append({"sp_id": int(sp_id), "reliability_score": round(score, 4), "avg_iou": round(avg_iou, 2)})
        
        analysis_results[noise_type] = sorted(reliability_results, key=lambda x: x['reliability_score'], reverse=True)
    return analysis_results

def list_model_layers(model_wrapper: BaseModelWrapper):
    """Lists all layers of the model."""
    logger = logging.getLogger('base')
    logger.info("Listing all layers of the model:")
    for name, module in model_wrapper.model.model.named_modules():
        logger.info(f"{name}: {module}")
# --------------------------------------------------------------------------
# --- 5. Main Execution Block
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automated Segmentation Model Reliability Pipeline")
    parser.add_argument('--data_dir', type=str, required=False, default='/bhupender/2025/Mouse_other/pascal_sam_dataset_final', help='Path to the prepared dataset directory.')
    parser.add_argument('--model_name', type=str, required=False, default='sam2_hiera_large', help='Name of the model (e.g., sam2_hiera_large).')
    parser.add_argument('--model_path', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/checkpoints/sam2.1_hiera_large.pt', help='Path to the model checkpoint file (.pt).')
    parser.add_argument('--model_config', type=str, required=False, default='/bhupender/2025/Mouse_other/SAMv2/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml', help='Path to the model config file (.yaml).')
    parser.add_argument('--target_layer_name', type=str, required=False, default='sam_mask_decoder.conv_s1', help='Dot-separated path to the model layer for EigenCAM.')
    parser.add_argument('--output_dir', type=str, default='./reliability_results')
    parser.add_argument('--num_iterations', type=int, default=2, help="Number of images to process from the dataset.")
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help="Image size as height width.")
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(args.output_dir, "analysis.log")), logging.StreamHandler()])
    logger = logging.getLogger('base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Model and Data Loading ---
    model_wrapper = get_model_wrapper(args.model_name, args, device)
       # List all layers of the model
    list_model_layers(model_wrapper)
    val_loader = create_generic_dataloader(data_dir=args.data_dir, image_size=tuple(args.image_size))
    # sys.exit(0)

    val_loader = create_generic_dataloader(data_dir=args.data_dir, image_size=tuple(args.image_size))
   
    noise_types = ['gaussian', 'salt_pepper']
    all_images_reliability = {nt: [] for nt in noise_types}

    # --- Main Loop ---
    for idata, batch in enumerate(val_loader):
        if idata >= len(val_loader) or idata >= args.num_iterations:
            break
        
        img_name = batch['img_name'][0]
        logger.info(f"--- Processing image {idata + 1}/{args.num_iterations}: {img_name} ---")
        current_image_output_dir = os.path.join(args.output_dir, img_name.rsplit('.', 1)[0])
        os.makedirs(current_image_output_dir, exist_ok=True)
        
        gt_mask = batch['mask_tensor'].squeeze().cpu().numpy().astype(np.uint8)
        orig_image_01 = batch['image_tensor'].squeeze().permute(1, 2, 0).cpu().numpy()
        
        orig_image_rgb_01 = np.array(Image.fromarray((orig_image_01 * 255).astype(np.uint8)).convert('RGB')) / 255.0
        segments_slic = slic(orig_image_rgb_01, n_segments=200, compactness=10, sigma=1, channel_axis=-1)
        
        in_context_sp_ids = select_superpixels_by_gt_mask(segments_slic, gt_mask)
        if not in_context_sp_ids:
            logger.warning(f"No superpixels found overlapping with the GT mask for {img_name}. Skipping.")
            continue
            
        pred_mask = model_wrapper.predict(batch)
        print(f'prediction mask shape{pred_mask.shape}')
        eigencam_heatmap = calculate_eigencam(model_wrapper, args.target_layer_name, batch, orig_image_rgb_01.shape)
        # Extract prompt points

        prompt_points = batch['prompt_points'].squeeze(0).cpu().numpy()  # Shape: (N, 2)

        # Visualize and save the overview image
        overview_path = os.path.join(current_image_output_dir, "overview.png")
        visualize_segmentation_comparison(
            orig_image_01, 
            gt_mask, 
            pred_mask, 
            eigencam_heatmap, 
            prompt_points=prompt_points, 
            save_path=overview_path
        )

        final_analysis = run_reliability_analysis(model_wrapper, batch, orig_image_01, gt_mask, segments_slic, in_context_sp_ids, noise_types)
        
        json_save_path = os.path.join(current_image_output_dir, "reliability_scores.json")
        with open(json_save_path, 'w') as f: json.dump(final_analysis, f, indent=4)
        
        for nt in noise_types:
            scores = [item['reliability_score'] for item in final_analysis.get(nt, [])]
            if scores: all_images_reliability[nt].append(np.mean(scores))
                
    # --- Final Summary Plotting ---
    logger.info("--- Generating Final Summary Plot ---")
    plot_data = []
    for noise_type, scores in all_images_reliability.items():
        if scores: plot_data.append({'Noise Type': noise_type, 'Average Reliability Score': np.mean(scores)})
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=(10, 6)); bars = sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', palette='viridis'); plt.title('Average Model Reliability Across Different Noise Types', fontsize=16); plt.ylabel('Reliability Score (Higher is Better)')
        plt.ylim(0, max(100, df_plot['Average Reliability Score'].max() * 1.1) if not df_plot.empty else 100)
        for bar in bars.patches: bars.annotate(f'{bar.get_height():.2f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
        plt.tight_layout(); save_path = os.path.join(args.output_dir, "final_reliability_summary.png"); plt.savefig(save_path, dpi=300); plt.close()
        logger.info(f"Saved final summary plot to {save_path}")

    logger.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
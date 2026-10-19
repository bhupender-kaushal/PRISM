import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms.functional as TF

from skimage.metrics import structural_similarity
from skimage.segmentation import slic

# --------------------------------------------------------------------------
# 1. Dataset Loaders 
# --------------------------------------------------------------------------
def load_pascal_or_isic(base_dir, target_size=(352, 352)):
    images = []
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir): return images
    
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(valid_extensions):
            img = Image.open(os.path.join(img_dir, f)).convert('RGB')
            img = img.resize(target_size, Image.BILINEAR)
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Base replacement (PASCAL format)
            mask_file = f.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Fallback replacement (ISIC format)
            if not os.path.exists(mask_path):
                mask_file = f.replace('.jpg', '_Segmentation.png')
                mask_path = os.path.join(mask_dir, mask_file)
                
            if not os.path.exists(mask_path): continue
                
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(target_size, Image.NEAREST)
            mask_np = (np.array(mask) > 127).astype(np.uint8)
            
            images.append({'name': f, 'image': img_np, 'gt_mask': mask_np})
    return images

def load_synapse(base_dir, list_dir, target_class=8, split="test_vol", target_slice=50):
    images = []
    if not os.path.exists(list_dir) or not os.path.exists(base_dir): return images
    
    sample_list = open(os.path.join(list_dir, f'{split}.txt')).readlines()
    for vol_name in sample_list:
        vol_name = vol_name.strip('\n')
        filepath = os.path.join(base_dir, f"{vol_name}.npy.h5")
        if not os.path.exists(filepath): continue
        
        with h5py.File(filepath, 'r') as data:
            vol_img, vol_label = data['image'][:], data['label'][:]
            
            valid_slices = np.where(np.any(vol_label == target_class, axis=(1, 2)))[0]
            slice_idx = valid_slices[len(valid_slices)//2] if len(valid_slices) > 0 else min(target_slice, vol_img.shape[0]-1)
            
            img_slice = vol_img[slice_idx, :, :]
            label_slice = vol_label[slice_idx, :, :]
            
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            mask_np = (label_slice == target_class).astype(np.uint8)
            
            images.append({'name': f"{vol_name}_slice_{slice_idx}", 'image': img_slice, 'gt_mask': mask_np})
    return images

def load_xcad(dataroot):
    images = []
    test_img_dir = os.path.join(dataroot, 'test', 'images')
    test_mask_dir = os.path.join(dataroot, 'test', 'masks')
    if not os.path.exists(test_img_dir) or not os.path.exists(test_mask_dir): return images
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for f in sorted(os.listdir(test_img_dir)):
        if f.lower().endswith(valid_extensions):
            img = Image.open(os.path.join(test_img_dir, f)).convert('L')
            img_np = np.array(img).astype(np.float32) / 255.0
            
            mask_path = os.path.join(test_mask_dir, f)
            if not os.path.exists(mask_path): continue
            
            mask = Image.open(mask_path).convert('L')
            mask_np = (np.array(mask) > 127).astype(np.uint8)
            
            images.append({'name': f, 'image': img_np, 'gt_mask': mask_np})
    return images

# --------------------------------------------------------------------------
# 2. Geometric Isolations
# --------------------------------------------------------------------------
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

def generate_superpixels(image_01):
    c_axis = -1 if image_01.ndim == 3 else None
    segments = slic(image_01, n_segments=500, compactness=10.0, sigma=1.0, channel_axis=c_axis, start_label=1)
    return segments

# --------------------------------------------------------------------------
# 3. PyTorch Hardware Acceleration Engine (GPU)
# --------------------------------------------------------------------------
def perturb_tensor_gpu(img_tensor, mask_tensor, noise_type, noise_level):
    """Executes high-speed matrix modifications entirely within VRAM."""
    perturbed = img_tensor.clone()

    if noise_type == 'gaussian':
        noise = torch.randn_like(img_tensor) * (0.1 * noise_level)
        perturbed += noise * mask_tensor
    elif noise_type in ['s&p', 'salt_pepper']:
        noise_char_value = min(0.5, 0.05 * noise_level)
        rand_tensor = torch.rand_like(img_tensor)
        salt = (rand_tensor < (noise_char_value / 2.0)).float()
        pepper = (rand_tensor > (1.0 - noise_char_value / 2.0)).float()
        perturbed = perturbed * (1.0 - salt * mask_tensor) + salt * mask_tensor
        perturbed = perturbed * (1.0 - pepper * mask_tensor)
    elif noise_type == 'poisson':
        sigma = max(0.05, 3.2 - noise_level)
        noisy_region = torch.poisson(perturbed * 25.0 * sigma) / (25.0 * sigma)
        perturbed = perturbed * (1.0 - mask_tensor) + noisy_region * mask_tensor
    elif noise_type == 'speckle':
        noise = torch.randn_like(img_tensor) * (0.1 * noise_level)
        perturbed *= (1.0 + noise * mask_tensor)
    elif noise_type == 'gaussian_blur':
        kernel_size = int(noise_level * 2) * 2 + 1 
        kernel_size = max(3, kernel_size)
        sigma = 0.5 * noise_level
        blurred = TF.gaussian_blur(img_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        perturbed = perturbed * (1.0 - mask_tensor) + blurred * mask_tensor
    elif noise_type == 'rician':
        n1 = torch.randn_like(img_tensor) * (0.1 * noise_level)
        n2 = torch.randn_like(img_tensor) * (0.1 * noise_level)
        noisy_region = torch.sqrt(torch.square(perturbed + n1) + torch.square(n2))
        perturbed = perturbed * (1.0 - mask_tensor) + noisy_region * mask_tensor

    return torch.clamp(perturbed, 0.0, 1.0)

# --------------------------------------------------------------------------
# 4. Graphing Engine
# --------------------------------------------------------------------------
def plot_dataset_thresholds(df, output_dir):
    sns.set_theme(style="whitegrid")
    
    # Mathematical Fix: Average across noise types and trials to find the Mean V(rho) per image.
    img_avg_df = df.groupby(['Dataset', 'Image', 'Rho'])['Survival_Steps'].mean().reset_index()
    img_avg_df.rename(columns={'Survival_Steps': 'Mean_Assessment_Points'}, inplace=True)
    
    datasets = img_avg_df['Dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 6), sharey=True)
    
    # Handle single-dataset edge case
    if len(datasets) == 1: 
        axes = [axes]
    
    # --- NEW: Common Title and Common X-Axis Label ---
    fig.suptitle("Assessment Points vs Superpixel Percentage Across Datasets", fontsize=26, weight='bold', y=1.05)
    fig.supxlabel(r"OC Superpixel Perturbation Percentage ($\rho_{oc}$)", fontsize=22, y=-0.05)
    
    for ax, ds_name in zip(axes, datasets):
        ds_data = img_avg_df[img_avg_df['Dataset'] == ds_name]
        
        # Seaborn automatically averages the `Mean_Assessment_Points` across all images for the dataset
        sns.lineplot(
            data=ds_data, x='Rho', y='Mean_Assessment_Points', 
            marker='o', errorbar='sd', ax=ax, color='b', linewidth=2
        )
        
        ax.axhspan(30, 45, color='green', alpha=0.15, label='Target Assessment Region (30-45)')
        ax.axhline(35, color='r', linestyle='--', label=r'Optimal Threshold $\tau=35$')
        
        # Set subplot title to just the dataset name to keep it clean
        ax.set_title(ds_name, fontsize=22)
        
        # Clear individual X-labels so they don't clash with the common sup_xlabel
        ax.set_xlabel("")
        
        if ax == axes[0]: 
            ax.set_ylabel(r"Average Assessment Points $N$", fontsize=20)
            
        ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        ax.legend(loc='upper right')
        
    # Ensure layout engine doesn't crush the overarching titles
    plt.tight_layout()
    
    # Added bbox_inches='tight' to prevent the suptitle and supxlabel from being cropped
    plt.savefig(os.path.join(output_dir, "Balanced_Threshold_Optimization.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------------------------
# 5. Main Execution Engine
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="High-Throughput GPU Superpixel Threshold Analyzer")
    parser.add_argument('--pascal_dir', type=str, default='/bhupender/2025/Mouse_other/dataset/pascal_sam_dataset_final/person')
    parser.add_argument('--isic_dir', type=str, default='/bhupender/datasets/Skin_Cancer/segmentation/ISIC2016/test')
    parser.add_argument('--synapse_dir', type=str, default='/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5')
    parser.add_argument('--synapse_list_dir', type=str, default="./lists/lists_Synapse")
    parser.add_argument('--xcad_dir', type=str, default='/bhupender/datasets/Angiogram/XCAD/')
    parser.add_argument('--output_dir', type=str, default='./oc_threshold_results')
    
    parser.add_argument('--max_images_per_dataset', type=int, default=20)
    parser.add_argument('--noise_step', type=float, default=0.05)
    parser.add_argument('--noise_types', nargs='+', default=['gaussian', 'salt_pepper', 'poisson', 'speckle', 'gaussian_blur', 'rician'])
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('base')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Hardware Acceleration Engine Active. Target Device: {device.type.upper()}")

    rho_values = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    raw_datasets = {}
    
    if args.pascal_dir: raw_datasets['PASCAL'] = load_pascal_or_isic(args.pascal_dir)
    if args.isic_dir: raw_datasets['ISIC'] = load_pascal_or_isic(args.isic_dir)
    if args.synapse_dir: raw_datasets['Synapse'] = load_synapse(args.synapse_dir, args.synapse_list_dir)
    if args.xcad_dir: raw_datasets['XCAD'] = load_xcad(args.xcad_dir)

    results_data = []

    for ds_name, images in raw_datasets.items():
        if not images: continue
            
        random.shuffle(images)
        selected_images = images[:args.max_images_per_dataset]
        logger.info(f"\n--- Processing Dataset: {ds_name} ({len(selected_images)} images) ---")
        
        for img_data in tqdm(selected_images, desc=f"Scanning {ds_name}"):
            orig_img = img_data['image']
            gt_mask = img_data['gt_mask']
            segments = generate_superpixels(orig_img)
            
            _, oc_sps_all = get_strict_ic_oc_superpixels(segments, gt_mask, ic_threshold=1.0, oc_threshold=0.0)
            if len(oc_sps_all) == 0: continue

            # Pre-load Original Tensor to GPU to prevent memory bottleneck
            if orig_img.ndim == 2:
                orig_tensor = torch.from_numpy(orig_img).unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                orig_tensor = torch.from_numpy(orig_img).permute(2,0,1).unsqueeze(0).float().to(device)
            
            channel = orig_tensor.size(1)

            for rho in rho_values:
                num_to_perturb = int(rho * len(oc_sps_all))
                
                for noise_type in args.noise_types:
                    for trial in range(3):
                        sp_to_perturb = random.sample(oc_sps_all, num_to_perturb)
                        
                        # Pre-compute Boolean Matrix Mask on GPU
                        region_mask_np = np.isin(segments, sp_to_perturb)
                        mask_tensor = torch.from_numpy(region_mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
                        if channel == 3: mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
                        
                        survival_steps = 0
                        noise_levels = np.arange(args.noise_step, 3.1, args.noise_step)
                        
                        for noise_level in noise_levels:
                            pert_tensor = perturb_tensor_gpu(orig_tensor, mask_tensor, noise_type, noise_level)
                            
                            # Shift Tensor to CPU strictly for skimage assessment parity
                            pert_np = pert_tensor.squeeze(0).cpu().numpy()
                            if channel == 3:
                                pert_np = pert_np.transpose(1, 2, 0)
                                ssim_score = structural_similarity(orig_img, pert_np, data_range=1.0, channel_axis=-1)
                            else:
                                pert_np = pert_np.squeeze(0)
                                ssim_score = structural_similarity(orig_img, pert_np, data_range=1.0)
                            
                            if ssim_score < 0.70: break
                            survival_steps += 1
                        
                        results_data.append({
                            'Dataset': ds_name, 'Image': img_data['name'], 'Rho': rho,
                            'Noise_Type': noise_type, 'Trial': trial, 'Survival_Steps': survival_steps,
                            'Superpixels_Perturbed': num_to_perturb
                        })

    df = pd.DataFrame(results_data)
    csv_path = os.path.join(args.output_dir, "oc_threshold_sweep_data.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\nSaved hardware-accelerated sweep data to {csv_path}")
    plot_dataset_thresholds(df, args.output_dir)

if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directly import the model from your project structure
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer

# =================================================================================
# DATASET LOADER
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
        data = h5py.File(filepath)
        image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label, 'case_name': vol_name}
        return sample

# =================================================================================
# INFERENCE FUNCTION
# =================================================================================
def run_inference_on_slice(model, image_slice, patch_size, device='cuda'):
    from scipy.ndimage import zoom
    from torchvision import transforms

    h, w = image_slice.shape
    
    if h != patch_size or w != patch_size:
        slice_rescaled = zoom(image_slice, (patch_size / h, patch_size / w), order=3)
    else:
        slice_rescaled = image_slice

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = x_transforms(slice_rescaled).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    
    if h != patch_size or w != patch_size:
        prediction = zoom(prediction, (h / patch_size, w / patch_size), order=0)
        
    return prediction

# =================================================================================
# MAIN LOGIC
# =================================================================================
def find_samples_with_class(dataset, target_class):
    """Scans the dataset to find all volume/slice pairs containing the target class."""
    valid_samples = []
    print(f"Scanning dataset for class {target_class}...")
    for vol_idx in tqdm(range(len(dataset))):
        sample = dataset[vol_idx]
        if sample is None:
            continue
        label_volume = sample['label']
        for slice_idx in range(label_volume.shape[0]):
            if np.any(label_volume[slice_idx] == target_class):
                valid_samples.append({'vol_idx': vol_idx, 'slice_idx': slice_idx})
    return valid_samples

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Model
    net = MaxViT_deformableLKAFormer(num_classes=args.num_classes).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    # 2. Initialize Dataset
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)

    # 3. Find all valid samples
    valid_samples = find_samples_with_class(db_test, args.target_class_index)
    if not valid_samples:
        print(f"No samples found with target class {args.target_class_index}. Exiting.")
        return
    
    # 4. Select N random samples
    num_to_select = min(args.num_samples, len(valid_samples))
    selected_samples = random.sample(valid_samples, k=num_to_select)
    print(f"Found {len(valid_samples)} slices. Processing {num_to_select} random samples.")

    # 5. Run inference and visualize
    for i, sample_info in enumerate(selected_samples):
        vol_idx, slice_idx = sample_info['vol_idx'], sample_info['slice_idx']
        
        volume_data = db_test[vol_idx]
        image_slice = volume_data['image'][slice_idx, :, :]
        label_slice = volume_data['label'][slice_idx, :, :]
        case_name = f"{volume_data['case_name']}_slice_{slice_idx}"

        full_prediction_mask = run_inference_on_slice(net, image_slice, args.img_size, device)
        class_specific_pred = (full_prediction_mask == args.target_class_index).astype(np.uint8)
        
        # --- Visualization ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Analysis for {case_name} - Target Class: {args.target_class_index}", fontsize=16)

        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title("Original Image Slice")
        axes[0].axis('off')

        axes[1].imshow(label_slice, cmap='nipy_spectral')
        axes[1].set_title("Ground Truth (All Classes)")
        axes[1].axis('off')

        axes[2].imshow(class_specific_pred, cmap='gray')
        axes[2].set_title(f"Predicted Mask for Class {args.target_class_index}")
        axes[2].axis('off')
        
        axes[3].imshow(full_prediction_mask, cmap='nipy_spectral')
        axes[3].set_title(f"Overall Predicted Mask (Classes: {np.unique(full_prediction_mask).size})")
        axes[3].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, f"{case_name}_class_{args.target_class_index}.png")
        plt.savefig(output_path)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_path", type=str, default="/bhupender/datasets/project_TransUNet/data/Synapse/test_vol_h5", help="Root dir for test volume data")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="Directory for lists_Synapse/test_vol.txt")
    parser.add_argument("--model_path", type=str, default="./MaxViT_deform_LKA.pth", help="Path to the trained model .pth file")
    parser.add_argument("--num_classes", type=int, default=9, help="Output channel of network")
    parser.add_argument("--img_size", type=int, default=224, help="Input patch size of network")
    parser.add_argument("--output_dir", type=str, default="./results_class_inference", help="Directory to save output visualizations")

    # --- Custom arguments for this script ---
    parser.add_argument("--target_class_index", type=int, default=3, required=False, help="The specific class index to analyze (e.g., 1 for Spleen).")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of random images to select and process.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
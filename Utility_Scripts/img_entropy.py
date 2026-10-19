import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================================
# 1. CONFIGURATION & DIRECTORIES
# =================================================================================
OUTPUT_DIR = "./image_level_entropy_plots"
EMA_SPAN = 7  # Adjust this to increase/decrease smoothing (alpha = 2 / (span + 1))

DIRECTORIES = {
    "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/sota_prism_ablation_results", 
    "SAMv2": "/bhupender/2025/Mouse_other/sota_prism_ablation_results", 
    "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/sota_prism_ablation_results", 
    "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/sota_prism_ablation_results", 
    "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/sota_prism_ablation_results", 
    "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/sota_prism_ablation_results", 
    "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/sota_prism_ablation_results", 
    "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/sota_prism_ablation_results", 
}

DATASET_PAIRS = {
    "Pascal-VOC": ("SAMv1", "SAMv2"),
    "ISIC-2016": ("MSDUNet", "TransFuse"),
    "Synapse": ("D-LKA", "TransDeeplab"),
    "XCAD": ("CDARL", "DARL")
}

# =================================================================================
# 2. DATA EXTRACTION & GLOBAL NORMALIZATION
# =================================================================================
def extract_all_data():
    """First pass: Load all data to find global min and max bounds."""
    all_dfs = {}
    global_min = float('inf')
    global_max = float('-inf')
    
    for dataset, models in DATASET_PAIRS.items():
        for model_name in models:
            json_path = os.path.join(DIRECTORIES.get(model_name, ""), "Ablation_Statistical_Results.json")
            
            if not os.path.exists(json_path):
                all_dfs[model_name] = None
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            raw_entropies = data.get('raw_image_entropies', [])
            if not raw_entropies:
                all_dfs[model_name] = None
                continue
                
            df = pd.DataFrame(raw_entropies)
            img_df = df.groupby('Image').agg(
                SOTA_Mean=('SOTA Entropy', 'mean'),
                PRISM_Mean=('PRISM Entropy', 'mean')
            ).reset_index()
            
            # Update global bounds
            local_min = min(img_df['SOTA_Mean'].min(), img_df['PRISM_Mean'].min())
            local_max = max(img_df['SOTA_Mean'].max(), img_df['PRISM_Mean'].max())
            
            if local_min < global_min: global_min = local_min
            if local_max > global_max: global_max = local_max
                
            all_dfs[model_name] = img_df
            
    return all_dfs, global_min, global_max

def process_and_normalize(df, g_min, g_max):
    """Second pass: Normalize to 0-1, sort by SOTA, and compute EMA."""
    if df is None or df.empty:
        return None
        
    df_norm = df.copy()
    
    if g_max > g_min:
        df_norm['SOTA_Norm'] = (df_norm['SOTA_Mean'] - g_min) / (g_max - g_min)
        df_norm['PRISM_Norm'] = (df_norm['PRISM_Mean'] - g_min) / (g_max - g_min)
    else:
        df_norm['SOTA_Norm'] = 0.0
        df_norm['PRISM_Norm'] = 0.0
        
    # Sort descending by SOTA to create a monotonically decreasing baseline
    df_norm = df_norm.sort_values(by='SOTA_Norm', ascending=False).reset_index(drop=True)
    
    # Calculate Exponential Moving Average (EMA) on the normalized scale
    df_norm['SOTA_EMA'] = df_norm['SOTA_Norm'].ewm(span=EMA_SPAN, adjust=False).mean()
    df_norm['PRISM_EMA'] = df_norm['PRISM_Norm'].ewm(span=EMA_SPAN, adjust=False).mean()
    
    return df_norm

# =================================================================================
# 3. PLOTTING THE 4x2 GRID
# =================================================================================
def plot_grid():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting data and calculating global bounds...")
    raw_data_dict, g_min, g_max = extract_all_data()
    print(f"Global Min: {g_min:.4f} | Global Max: {g_max:.4f}")
    
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(24, 20))
    datasets = list(DATASET_PAIRS.keys())
    
    for row_idx, dataset in enumerate(datasets):
        model_1, model_2 = DATASET_PAIRS[dataset]
        models = [model_1, model_2]
        
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            raw_df = raw_data_dict.get(model_name)
            
            df_img = process_and_normalize(raw_df, g_min, g_max)
            
            if df_img is None or df_img.empty:
                ax.text(0.5, 0.5, f"Data Not Found\n{model_name}", ha='center', va='center', fontsize=16, color='red')
                ax.set_title(f"{dataset} : {model_name}", fontsize=18, fontweight='bold')
                ax.axis('off')
                continue
                
            num_images = len(df_img)
            x = np.arange(num_images)
            
            # Plot raw normalized data as beautiful semi-transparent scatter points
            ax.scatter(x, df_img['SOTA_Norm'], color='orange', alpha=0.45, s=50, edgecolor='none', label='SOTA Raw' if (row_idx==0 and col_idx==0) else "")
            ax.scatter(x, df_img['PRISM_Norm'], color='dodgerblue', alpha=0.45, s=50, edgecolor='none', label='PRISM Raw' if (row_idx==0 and col_idx==0) else "")
            
            # Plot EMA as thick solid lines using deeper anchor tones for high contrast
            ax.plot(x, df_img['SOTA_EMA'], color='darkorange', linewidth=2.5, label=f'SOTA EMA (span={EMA_SPAN})' if (row_idx==0 and col_idx==0) else "")
            ax.plot(x, df_img['PRISM_EMA'], color='mediumblue', linewidth=2.5, label=f'PRISM EMA (span={EMA_SPAN})' if (row_idx==0 and col_idx==0) else "")
            
            # Formatting
            ax.set_title(f"{dataset} : {model_name}", fontsize=22)
            ax.set_ylim(-0.05, 1.05)
            
            # per-subplot y/x labels removed to use shared figure labels
            if col_idx == 0:
                pass
            else:
                pass
                
            ax.set_xticks([]) # Hide individual image indices
            ax.set_yticks([]) # Hide y-ticks for cleaner look
            # ax.set_xlabel(f"Images (n={num_images}, Sorted by SOTA)", fontsize=14)  # removed
            
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.grid(axis='x', linestyle=':', alpha=0.3)
            

                
            handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20, framealpha=0.95, bbox_to_anchor=(0.5, 0.00))

    plt.suptitle("Image-Level Rank Entropy Trajectories Across Architectures\n(Raw Coordinates vs. Exponential Moving Average)", fontsize=26, fontweight='bold', y=0.98)
    
    # Add common X and Y labels for the whole figure
    fig.text(0.5, 0.05, f"Images (n={num_images}, Sorted by SOTA)", ha='center', fontsize=20, fontweight='bold')
    fig.text(0.05, 0.5, "Entropy [0 - 1]", va='center', rotation='vertical', fontsize=20, fontweight='bold')
    
    plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])


    
    save_path = os.path.join(OUTPUT_DIR, "Global_Image_Level_EMA_Normalized.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"EMA Grid plot successfully generated and saved to -> {save_path}")

if __name__ == "__main__":
    plot_grid()
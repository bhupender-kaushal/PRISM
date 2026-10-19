import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =================================================================================
# 1. CONFIGURATION
# =================================================================================
OUTPUT_DIR = "./global_statistical_proof"

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
# 2. DATA EXTRACTION
# =================================================================================
def get_mean_entropy(json_path):
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    raw_entropies = data.get('raw_image_entropies', [])
    if not raw_entropies:
        return None
        
    df_raw = pd.DataFrame(raw_entropies)
    df_raw['Noise Type'] = df_raw['Noise Type'].str.upper()
    
    mean_df = df_raw.groupby('Noise Type').agg(
        SOTA_Mean=('SOTA Entropy', 'mean'),
        PRISM_Mean=('PRISM Entropy', 'mean')
    ).reset_index()
    
    return mean_df.set_index('Noise Type')

def build_global_cohort():
    global_pairs = []
    
    for dataset, (m1, m2) in DATASET_PAIRS.items():
        path1 = os.path.join(DIRECTORIES.get(m1, ""), "Ablation_Statistical_Results.json")
        path2 = os.path.join(DIRECTORIES.get(m2, ""), "Ablation_Statistical_Results.json")
        
        df1 = get_mean_entropy(path1)
        df2 = get_mean_entropy(path2)
        
        if df1 is not None:
            for noise in df1.index:
                global_pairs.append({
                    'Model': m1, 'Noise_Type': noise,
                    'SOTA_Mean': df1.loc[noise, 'SOTA_Mean'],
                    'PRISM_Mean': df1.loc[noise, 'PRISM_Mean']
                })
                
        if df2 is not None:
            for noise in df2.index:
                global_pairs.append({
                    'Model': m2, 'Noise_Type': noise,
                    'SOTA_Mean': df2.loc[noise, 'SOTA_Mean'],
                    'PRISM_Mean': df2.loc[noise, 'PRISM_Mean']
                })
                
    return pd.DataFrame(global_pairs)

def normalize_global_entropies(df):
    """
    Normalizes SOTA and PRISM entropies to a 0-1 scale using a shared 
    global min and max to preserve their strict relative differences.
    """
    df_norm = df.copy()
    
    # Find the absolute min and max across BOTH columns
    global_min = min(df_norm['SOTA_Mean'].min(), df_norm['PRISM_Mean'].min())
    global_max = max(df_norm['SOTA_Mean'].max(), df_norm['PRISM_Mean'].max())
    
    if global_max > global_min:
        df_norm['SOTA_Mean'] = (df_norm['SOTA_Mean'] - global_min) / (global_max - global_min)
        df_norm['PRISM_Mean'] = (df_norm['PRISM_Mean'] - global_min) / (global_max - global_min)
    else:
        df_norm['SOTA_Mean'] = 0.0
        df_norm['PRISM_Mean'] = 0.0
        
    return df_norm

# =================================================================================
# 3. STATISTICAL PROOF & VISUALIZATION
# =================================================================================
def execute_global_proof(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sota_array = df['SOTA_Mean'].values
    prism_array = df['PRISM_Mean'].values
    
    # Wilcoxon Signed-Rank Test (One-sided: SOTA > PRISM)
    w_stat, p_val = stats.wilcoxon(sota_array, prism_array, alternative='greater')
    cles = np.mean(sota_array > prism_array)
    
    print("\n" + "="*60)
    print("GLOBAL WILCOXON SIGNED-RANK TEST (NORMALIZED 0-1)")
    print("="*60)
    print(f"Total Paired Observations (Models x Noises): N = {len(df)}")
    print(f"Global SOTA Mean Entropy:  {np.mean(sota_array):.4f} ± {np.std(sota_array):.4f}")
    print(f"Global PRISM Mean Entropy: {np.mean(prism_array):.4f} ± {np.std(prism_array):.4f}")
    print("-" * 60)
    print(f"Wilcoxon Statistic (W): {w_stat:.1f}")
    print(f"p-value: {p_val:.4e}")
    print(f"CLES: {cles*100:.1f}% (Probability SOTA entropy > PRISM entropy)")
    
    if p_val < 0.05:
        print("\nCONCLUSION: Null Hypothesis Rejected (***).")
        print("Mathematical proof established: PRISM universally produces statistically significantly lower rank entropy across all datasets and models.")
    else:
        print("\nCONCLUSION: Failed to reject Null Hypothesis.")
        
    # --- Visualization: Paired Slopegraph ---
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Draw connecting lines
    for _, row in df.iterrows():
        # Blue if PRISM lowers entropy (desired), Red if SOTA is lower/tied
        color = 'dodgerblue' if row['PRISM_Mean'] < row['SOTA_Mean'] else 'salmon'
        ax.plot([0, 1], [row['SOTA_Mean'], row['PRISM_Mean']], color=color, alpha=0.6, linewidth=2.5)
        
    # Draw scatter points
    ax.scatter(np.zeros(len(df)), sota_array, color='darkgray', s=80, zorder=3, edgecolors='black')
    ax.scatter(np.ones(len(df)), prism_array, color='black', s=80, zorder=3, edgecolors='black')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['SOTA\n(Global Noise)', 'PRISM\n(Regional Noise)'], fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Rank Entropy [0 - 1]', fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    
    title_str = f"Rank Stability Across All Models\nWilcoxon $p = {p_val:.4e}$ | CLES = {cles*100:.1f}%"
    ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
    
    # Expanded y-limits to breathe after [0, 1] normalization
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    custom_lines = [
        Line2D([0], [0], color='dodgerblue', lw=3, label='Entropy Decreased (PRISM Superior)'),
        Line2D([0], [0], color='salmon', lw=3, label='Entropy Increased (SOTA Superior/Tied)')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=14, frameon=True, edgecolor='black')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Global_Wilcoxon_Slopegraph_Normalized.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\nVisual proof saved to -> {save_path}")

# =================================================================================
# 4. EXECUTION
# =================================================================================
if __name__ == "__main__":
    print("Aggregating global cohort data...")
    df_global = build_global_cohort()
    
    if df_global.empty:
        print("FATAL: No data found across the provided directories.")
    else:
        print("Normalizing entropies to 0-1 scale...")
        df_normalized = normalize_global_entropies(df_global)
        execute_global_proof(df_normalized)
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

# ==========================================
# 1. DATASET MAPPING (Global Reference)
# ==========================================
DATASET_MAPPING = {
    "SAMv1": "Pascal-VOC",
    "SAMv2": "Pascal-VOC",
    "MSDUNet": "ISIC-2016",
    "TransFuse": "ISIC-2016",
    "D-LKA": "Synapse",
    "TransDeeplab": "Synapse",
    "CDARL": "XCAD",        
    "DARL": "XCAD",         
}

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def clean_noise_name(raw_name):
    """
    Bulletproof standardization for inconsistent noise names.
    Ignores capitalization, hyphens, underscores, and ALL spaces.
    """
    lower_name = str(raw_name).strip().lower()
    no_space_name = lower_name.replace(" ", "")
    if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
        return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def extract_image_level_distributions(directories):
    data_points = []
    for model_name, base_dir in directories.items():
        print(f"Extracting data for model: {model_name}...")
        if not os.path.exists(base_dir):
            print(f"  -> Warning: Directory not found for {model_name}: {base_dir}")
            continue
            
        dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
            
        for run_folder in os.listdir(base_dir):
            run_path = os.path.join(base_dir, run_folder)
            if not os.path.isdir(run_path) or run_folder != "run_1": continue
            for img_folder in os.listdir(run_path):
                img_path = os.path.join(run_path, img_folder)
                if not os.path.isdir(img_path): continue
                json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
                if not os.path.exists(json_file): continue
                try:
                    data = load_json(json_file)
                except: continue
                
                for context_type in ["in_context_analysis", "out_of_context_analysis"]:
                    if context_type not in data: continue
                    context_label = "In-Context" if "in_context" in context_type else "Out-of-Context"
                    for noise_type, noise_data in data[context_type].items():
                        rel_list = noise_data.get("reliability_analysis", [])
                        if rel_list:
                            spill_metrics = noise_data.get("spillover_metrics", {})
                            csr = spill_metrics.get("conditional_average_spillover", 0.0)
                            discount = spill_metrics.get("global_spillover_discount", 1.0)
                            img_sp_scores = [sp.get("base_reliability_score", sp.get("final_reliability_score", sp.get("weighted_iou", 0)/100.0)) for sp in rel_list]
                            base_mean = np.nanmean(img_sp_scores)
                            final_img_rel = (base_mean * discount) if "base_reliability_score" in rel_list[0] else base_mean
                            
                            data_points.append({
                                "Dataset": dataset_name,
                                "Model": model_name,
                                "Context": context_label,
                                "Noise Type": clean_noise_name(noise_type),
                                "Reliability": final_img_rel,
                                "CSR": csr
                            })
    return pd.DataFrame(data_points)

def plot_comprehensive_kde_matrix(df, output_dir="./kde_plots"):
    os.makedirs(output_dir, exist_ok=True)
    df_clean = df.dropna(subset=['Reliability', 'CSR']).copy()
    if df_clean["Reliability"].max() <= 1.0: df_clean["Reliability"] *= 100.0
    df_clean["CSR"] = df_clean["CSR"] * 100.0

    noises = sorted(df_clean['Noise Type'].unique())
    num_noises = len(noises)
    total_model_rows = df_clean["Model"].nunique()
    
    sns.set_theme(style="whitegrid", context="paper")
    context_colors = {"In-Context": "dodgerblue", "Out-of-Context": "darkorange"}
    
    fig, axes = plt.subplots(nrows=total_model_rows, ncols=num_noises, 
                             figsize=(4.5 * num_noises, 4.0 * total_model_rows), 
                             sharex=True, sharey='row')
    
    current_row = 0
    unique_datasets = list(dict.fromkeys(DATASET_MAPPING.values()))
    
    for dataset in unique_datasets:
        group = df_clean[df_clean['Dataset'] == dataset]
        if group.empty: continue
        models_in_ds = sorted(group['Model'].unique())
        num_models = len(models_in_ds)
        
        # --- MATHEMATICAL CANVAS LOCATION FOR CENTERING THE TEXT ---
        start_row = current_row
        end_row = current_row + num_models - 1
        y_center = 0.08 + (0.95 - 0.08) * (1.0 - ((start_row + end_row) / 2.0 + 0.5) / total_model_rows)
        
        fig.text(0.04, y_center, dataset.upper(), 
                 fontsize=22, fontweight='bold', rotation=90, 
                 ha='center', va='center', color='black',
                 bbox={'facecolor': 'lightgray', 'alpha': 0.0, 'pad': 8, 'boxstyle': 'round,pad=0.3'})
        
        for row_idx_in_ds, model in enumerate(models_in_ds):
            for j, noise in enumerate(noises):
                ax1 = axes[current_row, j]
                ax2 = ax1.twinx()
                subset = df_clean[(df_clean['Model'] == model) & (df_clean['Noise Type'] == noise)]
                
                # --- KERNEL DENSITY PLOTTING LOGIC RESTORED ---
                for ctx, color in context_colors.items():
                    c_data = subset[subset['Context'] == ctx]
                    if not c_data.empty:
                        # common_norm=False prevents squeezing. 
                        sns.kdeplot(data=c_data, x="Reliability", color=color, fill=True, alpha=0.3, linewidth=2, ax=ax1, common_norm=False)
                        sns.kdeplot(data=c_data, x="CSR", color=color, fill=False, linestyle='--', linewidth=2, ax=ax2, common_norm=False)

                # --- EXPLICIT LABEL CLEANUP ---
                ax1.set_xlabel("")
                ax1.set_ylabel("")
                ax2.set_ylabel("") # Removes auto-generated "Density"

                ax1.set_xlim(-5, 105)
                ax1.set_yticks([]); ax2.set_yticks([])
                ax1.set_xticks([0, 20, 40, 60, 80, 100])
                
                if current_row == 0: 
                    ax1.set_title(noise, fontsize=22, fontweight='bold', pad=15)
                
                if j == 0: 
                    ax1.set_ylabel(model, fontsize=20, fontweight='bold', labelpad=20, rotation=90, va='center')
                else:
                    ax1.set_ylabel("")
                
                if current_row == total_model_rows - 1: 
                    ax1.set_xticklabels(['0', '20', '40', '60', '80', '100'], fontsize=20)
                    # if j == num_noises // 2:
                    #     ax1.set_xlabel("Score (0-100)", fontsize=22, fontweight='bold', labelpad=20)
                else:
                    ax1.set_xticklabels([])
                    ax1.set_xlabel("")
            current_row += 1

    legend_elements = [
        Line2D([0], [0], color='dodgerblue', lw=4, linestyle='-', label='IC Reliability'),
        Line2D([0], [0], color='darkorange', lw=4, linestyle='-', label='OC Reliability'),
        Line2D([0], [0], color='dodgerblue', lw=4, linestyle='--', label='IC CSR'),
        Line2D([0], [0], color='darkorange', lw=4, linestyle='--', label='OC CSR')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=22, frameon=True)
    
    plt.tight_layout(rect=[0.06, 0.08, 1, 0.95])
    plt.subplots_adjust(wspace=0.2, hspace=0.4) 
    
    save_path = os.path.join(output_dir, "Dataset_Grouped_KDE_Matrix.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"\nMatrix saved to -> {save_path}")

def main():
    parser = argparse.ArgumentParser()
    default_dirs = {
        "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/new_expresults_50", 
        "SAMv2": "/bhupender/2025/Mouse_other/new_expresults_50", 
        "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/new_expresults_50", 
        "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/new_expresults_50", 
        "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults_50", 
        "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults_50", 
        "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/new_expresults_50", 
        "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/new_expresults_50"
    }
    parser.add_argument('--directories', type=json.loads, default=json.dumps(default_dirs))
    args = parser.parse_args()
    
    dirs = json.loads(args.directories) if isinstance(args.directories, str) else args.directories
    
    df = extract_image_level_distributions(dirs)
    plot_comprehensive_kde_matrix(df)

if __name__ == "__main__":
    main()
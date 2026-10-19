# import os
# import json
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import argparse
# import math

# # ==========================================
# # 1. DATASET MAPPING
# # ==========================================
# DATASET_MAPPING = {
#     "SAMv1": "Pascal-VOC",
#     "SAMv2": "Pascal-VOC",
#     "MSDUNet": "ISIC-2016",
#     "TransFuse": "ISIC-2016",
#     "D-LKA": "Synapse",
#     "TransDeeplab": "Synapse",
#     "CDARL": "XCAD",        
#     "DARL": "XCAD",         
# }

# GAMMA = 1.0

# def load_json(file_path):
#     with open(file_path, "r") as file:
#         return json.load(file)

# def clean_noise_name(raw_name):
#     lower_name = str(raw_name).strip().lower()
#     no_space_name = lower_name.replace(" ", "")
#     if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
#         return 'Salt & Pepper'
#     return str(raw_name).strip().replace('_', ' ').title()

# # ==========================================
# # NEW MATHEMATICAL ENGINE: Independent Trajectory Integration
# # ==========================================
# def extract_comprehensive_data(directories):
#     """
#     Crawls through model directories and extracts the integrated Trajectory Utility 
#     (A_ic and A_oc) aggregated strictly at the image level.
#     """
#     data_points = []
    
#     for model_name, base_dir in directories.items():
#         print(f"Extracting data for model: {model_name}...")
#         if not os.path.exists(base_dir):
#             print(f"  -> Warning: Directory not found for {model_name}: {base_dir}")
#             continue
            
#         dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
            
#         for root, dirs, files in os.walk(base_dir):
#             for file in files:
#                 if file.endswith("_analysis.json"):
#                     parts = root.replace('\\', '/').split('/')
#                     run_name = "run_1"
#                     for p in parts:
#                         if p.startswith("run_"):
#                             run_name = p
#                             break
                    
#                     if run_name != "run_1":
#                         continue
                        
#                     json_path = os.path.join(root, file)
#                     try: data = load_json(json_path)
#                     except Exception: continue
                    
#                     ic_data = data.get("in_context_analysis", {})
#                     oc_data = data.get("out_of_context_analysis", {})
#                     shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))
                    
#                     for raw_noise in shared_noises:
#                         noise_type = clean_noise_name(raw_noise)
#                         ic_levels = ic_data[raw_noise].get("level_data", [])
#                         oc_levels = oc_data[raw_noise].get("level_data", [])
                        
#                         if not ic_levels or not oc_levels: continue
                        
#                         # Extract IC trajectory
#                         IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels]
#                         OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels]

#                         N_ic, N_oc = len(IC_vals), len(OC_vals)
#                         if N_ic <= 1 or N_oc <= 1: continue

#                         IC_0 = max(1e-6, IC_vals[0])
#                         OC_0 = max(1e-6, OC_vals[0])
                        
#                         # 1. Independent IC Integration
#                         W_ic, A_ic = 0.0, 0.0
#                         for k in range(1, N_ic):
#                             eta_k = k / (N_ic - 1)
#                             w_k = math.exp(-GAMMA * eta_k)
#                             A_ic += w_k * (IC_vals[k] / IC_0)
#                             W_ic += w_k
#                         norm_A_ic = A_ic / W_ic if W_ic > 0 else 0.0

#                         # 2. Independent OC Integration
#                         W_oc, A_oc = 0.0, 0.0
#                         for k in range(1, N_oc):
#                             eta_k = k / (N_oc - 1)
#                             w_k = math.exp(-GAMMA * eta_k)
#                             A_oc += w_k * (OC_vals[k] / OC_0)
#                             W_oc += w_k
#                         norm_A_oc = A_oc / W_oc if W_oc > 0 else 0.0
                        
#                         # Bound to prevent hallucination artifacts skewing the density
#                         A_ic_final = min(1.0, norm_A_ic) * 100.0
#                         A_oc_final = min(1.0, norm_A_oc) * 100.0
                        
#                         data_points.append({
#                             "Dataset": dataset_name,
#                             "Model": model_name,
#                             "Context": "In-Context",
#                             "Noise_Type": noise_type,
#                             "Utility": A_ic_final
#                         })
#                         data_points.append({
#                             "Dataset": dataset_name,
#                             "Model": model_name,
#                             "Context": "Out-of-Context",
#                             "Noise_Type": noise_type,
#                             "Utility": A_oc_final
#                         })
                            
#     return pd.DataFrame(data_points)

# # ==========================================
# # KDE PLOTTERS
# # ==========================================
# def plot_comprehensive_kde_grid(df, output_dir="./kde_plots"):
#     """
#     Creates a 2x4 KDE grid. 
#     Plots Regional Trajectory Utility on the X-axis.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     df_clean = df.dropna(subset=['Utility']).copy()

#     unique_datasets = [ds for ds in ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD'] if ds in df_clean['Dataset'].unique()]
    
#     plot_pairs = []
#     for dataset in unique_datasets:
#         dataset_models = sorted(df_clean[df_clean['Dataset'] == dataset]['Model'].unique())
#         for model in dataset_models:
#             plot_pairs.append((dataset, model))
            
#     plot_pairs = plot_pairs[:8]
    
#     fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
#     sns.set_theme(style="whitegrid", context="paper")
    
#     ic_color = "dodgerblue"
#     oc_color = "darkorange"
    
#     for i, (dataset, model) in enumerate(plot_pairs):
#         row_idx = i // 4
#         col_idx = i % 4
            
#         print(f"Plotting Comprehensive KDE distributions for: {dataset} | {model}")
        
#         ax1 = axes[row_idx, col_idx]
        
#         subset = df_clean[(df_clean['Model'] == model) & (df_clean['Dataset'] == dataset)]
#         ic_data = subset[subset['Context'] == 'In-Context']
#         oc_data = subset[subset['Context'] == 'Out-of-Context']
        
#         if not ic_data.empty:
#             sns.kdeplot(data=ic_data, x='Utility', color=ic_color, fill=True, alpha=0.3, linewidth=3, ax=ax1)
#         if not oc_data.empty:
#             sns.kdeplot(data=oc_data, x='Utility', color=oc_color, fill=True, alpha=0.3, linewidth=3, ax=ax1)
        
#         ax1.set_title(f"{dataset} | {model}", fontsize=20, fontweight='bold', pad=15)
        
#         # Remove local axis labels to use global supxlabel/supylabel
#         ax1.set_xlabel("")
#         ax1.set_ylabel("")
        
#         ax1.tick_params(axis='both', which='major', labelsize=16)
#         ax1.set_xlim(-5, 105)

#     for i in range(len(plot_pairs), 8):
#         row_idx = i // 4
#         col_idx = i % 4
#         axes[row_idx, col_idx].set_visible(False)
#         for ax in axes[row_idx, col_idx].figure.axes:
#             if ax is not axes[row_idx, col_idx] and ax.bbox.bounds == axes[row_idx, col_idx].bbox.bounds:
#                 ax.set_visible(False)
            
#     # Global Axis Labels
#     fig.supxlabel("Regional Scores ($E_{ic}$ and $E_{oc}$) %", fontsize=24, fontweight='bold', y=0.04)
#     # fig.supylabel("Distribution Density", fontsize=24, fontweight='bold', x=0.03)
            
#     legend_elements = [
#         Line2D([0], [0], color=ic_color, lw=4, label='In-Context (IC) Score'),
#         Line2D([0], [0], color=oc_color, lw=4, label='Out-of-Context (OC) Score')
#     ]
    
#     fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
#                ncol=2, fontsize=20, frameon=True, edgecolor='black', handlelength=2.5)
               
#     plt.suptitle("Dataset | Model-Wise Density", fontsize=28, fontweight='bold', y=0.98)
    
#     plt.tight_layout(pad=3.0)
#     plt.subplots_adjust(wspace=0.30, hspace=0.35, top=0.90, bottom=0.12, left=0.12) 
    
#     save_pdf = os.path.join(output_dir, "Comprehensive_Dataset_KDE_Grid_2x4.pdf")
#     plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
#     print(f"\nComprehensive 2x4 KDE grid successfully saved to:\n -> {save_pdf}")
#     plt.close()


# def plot_model_perturbation_kdes(df, output_dir="./kde_plots"):
#     """
#     Creates an 8x6 Monolithic KDE Grid:
#     - 8 Rows (1 for each Model, Grouped by Dataset).
#     - 6 Columns (1 for each Noise Type).
#     - Hides all ticks/labels except Top Titles (Noise) & Left Labels (Dataset/Model).
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     df_clean = df.dropna(subset=['Utility']).copy()

#     ic_color = "dodgerblue"
#     oc_color = "darkorange"
#     sns.set_theme(style="whitegrid", context="paper")
    
#     # Establish strict row ordering grouped by Dataset
#     ordered_datasets = ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD']
#     ordered_models = []
#     dataset_rows = {}
    
#     r_idx = 0
#     for ds in ordered_datasets:
#         if ds in df_clean['Dataset'].unique():
#             ds_models = sorted(df_clean[df_clean['Dataset'] == ds]['Model'].unique())
#             dataset_rows[ds] = []
#             for m in ds_models:
#                 ordered_models.append((ds, m))
#                 dataset_rows[ds].append(r_idx)
#                 r_idx += 1
                
#     # Extract unique sorted noises (expecting 6 columns)
#     noises = sorted(df_clean['Noise_Type'].unique())[:6]
#     num_rows = len(ordered_models)
#     num_cols = len(noises)
    
#     if num_rows == 0 or num_cols == 0:
#         return
        
#     print(f"\nPlotting {num_rows}x{num_cols} Monolithic Model-Perturbation KDE Grid...")
    
#     # Dynamic Figure Sizing (Roughly 24x16 for an 8x6 matrix)
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 4, num_rows * 2.5))
#     if num_rows == 1: axes = np.expand_dims(axes, axis=0)
#     if num_cols == 1: axes = np.expand_dims(axes, axis=1)
    
#     for r, (dataset, model) in enumerate(ordered_models):
#         for c, noise in enumerate(noises):
#             ax1 = axes[r, c]
            
#             subset = df_clean[(df_clean['Model'] == model) & (df_clean['Noise_Type'] == noise)]
#             ic_data = subset[subset['Context'] == 'In-Context']
#             oc_data = subset[subset['Context'] == 'Out-of-Context']
            
#             # Plot KDE Distributions
#             if not ic_data.empty:
#                 sns.kdeplot(data=ic_data, x='Utility', color=ic_color, fill=True, alpha=0.3, linewidth=2, ax=ax1)
#             if not oc_data.empty:
#                 sns.kdeplot(data=oc_data, x='Utility', color=oc_color, fill=True, alpha=0.3, linewidth=2, ax=ax1)
            
#             # Strict constraints & clean visuals
#             ax1.set_xlim(-5, 105)
#             ax1.set_xlabel("")
#             ax1.set_ylabel("")
            
#             # Remove ALL ticks & labels as requested for the dense grid
#             ax1.set_xticklabels([])
#             ax1.set_yticklabels([])
#             ax1.tick_params(bottom=False, left=False)
            
#             # Column Header (Noise Names) - only on top row
#             if r == 0:
#                 ax1.set_title(noise, fontsize=20, fontweight='bold', pad=15)
                
#             # Row Header (Model ONLY) - only on leftmost column
#             if c == 0:
#                 ax1.set_ylabel(f"{model}", fontsize=18, fontweight='bold', 
#                                rotation=0, ha='right', va='center', labelpad=15)

#     # Global Axes
#     fig.supxlabel("Regional Scores ($E_{ic}$ and $E_{oc}$) %", fontsize=28, fontweight='bold', y=0.04)
#     # fig.supylabel("Distribution Density", fontsize=28, fontweight='bold', x=0.01)

#     # Force draw to get geometric coordinates for the vertical dataset names
#     fig.canvas.draw()
    
#     for dataset, row_indices in dataset_rows.items():
#         if not row_indices: continue
        
#         # Geometrically find center Y between the associated model rows
#         y_centers = [axes[r, 0].get_position().y0 + (axes[r, 0].get_position().height / 2.0) for r in row_indices]
#         avg_y = sum(y_centers) / len(y_centers)
        
#         # Position X to the left of the subplot boundary and model names
#         x_pos = axes[0, 0].get_position().x0 - 0.08
        
#         # Vertical dataset name, no bbox frame
#         fig.text(x_pos, avg_y, dataset, rotation=90, va='center', ha='center', 
#                  fontsize=22, fontweight='bold')

#     # Build Unified Legend
#     legend_elements = [
#         Line2D([0], [0], color=ic_color, lw=4, label='In-Context (IC) Score'),
#         Line2D([0], [0], color=oc_color, lw=4, label='Out-of-Context (OC) Score')
#     ]
    
#     fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01), 
#                ncol=2, fontsize=22, frameon=True, edgecolor='black', handlelength=2.5)
               
#     plt.suptitle("Model Perturbation-Wise KDE Plots", fontsize=32, fontweight='bold', y=0.98)
    
#     # Pack layout tightly, leave space on left for vertical dataset names and bottom for Legend
#     plt.tight_layout(pad=1.0)
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.92, bottom=0.09, left=0.16)
    
#     save_pdf = os.path.join(output_dir, "Monolithic_Model_Perturbation_KDE_Grid.pdf")
#     plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
#     print(f"-> Monolithic 8x6 KDE grid successfully saved to: {save_pdf}")
#     plt.close()

# def main():
#     parser = argparse.ArgumentParser(description="Plot Comprehensive PRISM KDEs.")
#     parser.add_argument('--directories', type=json.loads, 
#                         default={
#                             "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out", 
#                             "SAMv2": "/bhupender/2025/Mouse_other/prism_out", 
#                             "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out", 
#                             "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out", 
#                             "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out", 
#                             "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out", 
#                             "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out", 
#                             "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out", 
#                         },
#                         help="JSON string of directories")
#     args = parser.parse_args()

#     print("Extracting Comprehensive Image-Level Data (New Math Engine)...")
#     df = extract_comprehensive_data(args.directories)
    
#     if df.empty:
#         print("No valid data found. Check your directory paths.")
#         return
        
#     print("\nGenerating Comprehensive 4x2 KDE Grid...")
#     plot_comprehensive_kde_grid(df, output_dir="./kde_plots")
    
#     plot_model_perturbation_kdes(df, output_dir="./kde_plots")

# if __name__ == "__main__":
#     main()



import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import math
import warnings

# Suppress KDE singular warnings for perfect 0 distributions (e.g., perfect RP=0)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. DATASET MAPPING
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

GAMMA = 1.0

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def clean_noise_name(raw_name):
    lower_name = str(raw_name).strip().lower()
    no_space_name = lower_name.replace(" ", "")
    if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
        return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

# ==========================================
# MATHEMATICAL ENGINE: Utilities + Penalties
# ==========================================
def calculate_step_rp_and_iou(level_0, level_k):
    """Calculates step-wise Rank Perplexity (RP)."""
    sp_0 = {sp['sp_id']: sp.get('rank', 0) for sp in level_0.get('superpixels', [])}
    sp_k = {sp['sp_id']: sp.get('rank', 0) for sp in level_k.get('superpixels', [])}
    
    N = len(sp_0)
    if N <= 1: return 0.0, 0.0
    
    max_jumps = math.floor((N**2) / 2.0)
    total_jumps = sum(abs(sp_k.get(sp_id, sp_0[sp_id]) - sp_0[sp_id]) for sp_id in sp_0)
    rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
    raw_iou = np.mean([sp['iou'] for sp in level_0.get('superpixels', [])]) if level_0.get('superpixels') else 0.0
    return raw_iou, rp

def extract_comprehensive_data(directories):
    """
    Crawls through model directories and extracts the integrated Trajectory Utility 
    (A_ic and A_oc) AND Penalties (CSR, RP) aggregated strictly at the image level.
    """
    data_points = []
    
    for model_name, base_dir in directories.items():
        print(f"Extracting data for model: {model_name}...")
        if not os.path.exists(base_dir):
            print(f"  -> Warning: Directory not found for {model_name}: {base_dir}")
            continue
            
        dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
            
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith("_analysis.json"):
                    parts = root.replace('\\', '/').split('/')
                    run_name = "run_1"
                    for p in parts:
                        if p.startswith("run_"):
                            run_name = p
                            break
                    
                    if run_name != "run_1":
                        continue
                        
                    json_path = os.path.join(root, file)
                    try: data = load_json(json_path)
                    except Exception: continue
                    
                    ic_data = data.get("in_context_analysis", {})
                    oc_data = data.get("out_of_context_analysis", {})
                    shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))
                    
                    for raw_noise in shared_noises:
                        noise_type = clean_noise_name(raw_noise)
                        ic_levels = ic_data[raw_noise].get("level_data", [])
                        oc_levels = oc_data[raw_noise].get("level_data", [])
                        
                        if not ic_levels or not oc_levels: continue
                        
                        # 1. Extract Trajectory Utility
                        IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels]
                        OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels]

                        N_ic, N_oc = len(IC_vals), len(OC_vals)
                        if N_ic <= 1 or N_oc <= 1: continue

                        IC_0 = max(1e-6, IC_vals[0])
                        OC_0 = max(1e-6, OC_vals[0])
                        
                        W_ic, A_ic = 0.0, 0.0
                        for k in range(1, N_ic):
                            eta_k = k / (N_ic - 1)
                            w_k = math.exp(-GAMMA * eta_k)
                            A_ic += w_k * (IC_vals[k] / IC_0)
                            W_ic += w_k
                        norm_A_ic = A_ic / W_ic if W_ic > 0 else 0.0

                        W_oc, A_oc = 0.0, 0.0
                        for k in range(1, N_oc):
                            eta_k = k / (N_oc - 1)
                            w_k = math.exp(-GAMMA * eta_k)
                            A_oc += w_k * (OC_vals[k] / OC_0)
                            W_oc += w_k
                        norm_A_oc = A_oc / W_oc if W_oc > 0 else 0.0
                        
                        A_ic_final = min(1.0, norm_A_ic) * 100.0
                        A_oc_final = min(1.0, norm_A_oc) * 100.0

                        # 2. Extract CSR (Scaled 0-100)
                        ic_csr = ic_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0) * 100.0
                        oc_csr = oc_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0) * 100.0
                        
                        # 3. Extract Trajectory RP (Scaled 0-100)
                        ic_rps = [calculate_step_rp_and_iou(ic_levels[0], lvl)[1] for lvl in ic_levels[1:]]
                        oc_rps = [calculate_step_rp_and_iou(oc_levels[0], lvl)[1] for lvl in oc_levels[1:]]
                        avg_ic_rp = np.mean(ic_rps) * 100.0 if ic_rps else 0.0
                        avg_oc_rp = np.mean(oc_rps) * 100.0 if oc_rps else 0.0
                        
                        data_points.append({
                            "Dataset": dataset_name, "Model": model_name, "Context": "In-Context",
                            "Noise_Type": noise_type, "Utility": A_ic_final, "CSR": ic_csr, "RP": avg_ic_rp
                        })
                        data_points.append({
                            "Dataset": dataset_name, "Model": model_name, "Context": "Out-of-Context",
                            "Noise_Type": noise_type, "Utility": A_oc_final, "CSR": oc_csr, "RP": avg_oc_rp
                        })
                            
    return pd.DataFrame(data_points)

# ==========================================
# KDE PLOTTERS
# ==========================================
def plot_comprehensive_kde_grid(df, output_dir="./kde_plots"):
    """
    Creates a 2x4 KDE grid. 
    Plots Regional Trajectory Utility and Penalties (CSR/RP).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_clean = df.dropna(subset=['Utility', 'CSR', 'RP']).copy()

    unique_datasets = [ds for ds in ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD'] if ds in df_clean['Dataset'].unique()]
    
    plot_pairs = []
    for dataset in unique_datasets:
        dataset_models = sorted(df_clean[df_clean['Dataset'] == dataset]['Model'].unique())
        for model in dataset_models:
            plot_pairs.append((dataset, model))
            
    plot_pairs = plot_pairs[:8]
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
    sns.set_theme(style="whitegrid", context="paper")
    
    ic_color = "dodgerblue"
    oc_color = "darkorange"
    
    for i, (dataset, model) in enumerate(plot_pairs):
        row_idx = i // 4
        col_idx = i % 4
            
        print(f"Plotting Comprehensive KDE distributions for: {dataset} | {model}")
        
        ax1 = axes[row_idx, col_idx]
        ax2 = ax1.twinx()  # Secondary axis for Penalties
        
        subset = df_clean[(df_clean['Model'] == model) & (df_clean['Dataset'] == dataset)]
        ic_data = subset[subset['Context'] == 'In-Context']
        oc_data = subset[subset['Context'] == 'Out-of-Context']
        
        # Plot 1: Utility (Filled)
        if not ic_data.empty: sns.kdeplot(data=ic_data, x='Utility', color=ic_color, fill=True, alpha=0.3, linewidth=3, ax=ax1, warn_singular=False)
        if not oc_data.empty: sns.kdeplot(data=oc_data, x='Utility', color=oc_color, fill=True, alpha=0.3, linewidth=3, ax=ax1, warn_singular=False)
        
        # Plot 2 & 3: Penalties (Dashed = CSR, Dotted = RP)
        if not ic_data.empty:
            sns.kdeplot(data=ic_data, x='CSR', color=ic_color, fill=False, linestyle='--', linewidth=2, ax=ax2, warn_singular=False)
            sns.kdeplot(data=ic_data, x='RP', color=ic_color, fill=False, linestyle=':', linewidth=2, ax=ax2, warn_singular=False)
        if not oc_data.empty:
            sns.kdeplot(data=oc_data, x='CSR', color=oc_color, fill=False, linestyle='--', linewidth=2, ax=ax2, warn_singular=False)
            sns.kdeplot(data=oc_data, x='RP', color=oc_color, fill=False, linestyle=':', linewidth=2, ax=ax2, warn_singular=False)
        
        ax1.set_title(f"{dataset} | {model}", fontsize=20, fontweight='bold', pad=15)
        
        # Remove local axis labels to use global supxlabel/supylabel
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='y', which='major', labelsize=16)
        ax1.set_xlim(-5, 105)
        ax2.grid(False)

    for i in range(len(plot_pairs), 8):
        row_idx = i // 4
        col_idx = i % 4
        axes[row_idx, col_idx].set_visible(False)
        for ax in axes[row_idx, col_idx].figure.axes:
            if ax is not axes[row_idx, col_idx] and ax.bbox.bounds == axes[row_idx, col_idx].bbox.bounds:
                ax.set_visible(False)
            
    # Global Axis Labels
    fig.supxlabel("Metric Scores ($E_{ic}$, $E_{oc}$) %,  and CSR, RP ", fontsize=24, fontweight='bold', y=0.04)
    # fig.supylabel("Distribution Density", fontsize=24, fontweight='bold', x=0.03)
            
    legend_elements = [
        Line2D([0], [0], color=ic_color, lw=4, label='IC Score (Filled)'),
        Line2D([0], [0], color=ic_color, lw=3, linestyle='--', label='IC CSR (Dashed)'),
        Line2D([0], [0], color=ic_color, lw=3, linestyle=':', label='IC RP (Dotted)'),
        Line2D([0], [0], color=oc_color, lw=4, label='OC Score (Filled)'),
        Line2D([0], [0], color=oc_color, lw=3, linestyle='--', label='OC CSR (Dashed)'),
        Line2D([0], [0], color=oc_color, lw=3, linestyle=':', label='OC RP (Dotted)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), 
               ncol=3, fontsize=18, frameon=True, edgecolor='black', handlelength=3.0)
               
    plt.suptitle("Dataset, Model-Wise Density (left y-axis) and Penalty Distributions (right y-axis)", fontsize=28, fontweight='bold', y=1.0)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.30, hspace=0.35, top=0.90, bottom=0.10, left=0.10, right=0.90) 
    
    save_pdf = os.path.join(output_dir, "Comprehensive_Dataset_KDE_Grid_2x4.pdf")
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
    print(f"\nComprehensive 2x4 KDE grid successfully saved to:\n -> {save_pdf}")
    plt.close()


def plot_model_perturbation_kdes(df, output_dir="./kde_plots"):
    """
    Creates an 8x6 Monolithic KDE Grid:
    - 8 Rows (1 for each Model, Grouped by Dataset).
    - 6 Columns (1 for each Noise Type).
    - Hides all ticks/labels except Top Titles (Noise) & Left Labels (Dataset/Model).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_clean = df.dropna(subset=['Utility', 'CSR', 'RP']).copy()

    ic_color = "dodgerblue"
    oc_color = "darkorange"
    sns.set_theme(style="whitegrid", context="paper")
    
    # Establish strict row ordering grouped by Dataset
    ordered_datasets = ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD']
    ordered_models = []
    dataset_rows = {}
    
    r_idx = 0
    for ds in ordered_datasets:
        if ds in df_clean['Dataset'].unique():
            ds_models = sorted(df_clean[df_clean['Dataset'] == ds]['Model'].unique())
            dataset_rows[ds] = []
            for m in ds_models:
                ordered_models.append((ds, m))
                dataset_rows[ds].append(r_idx)
                r_idx += 1
                
    # Extract unique sorted noises (expecting 6 columns)
    noises = sorted(df_clean['Noise_Type'].unique())[:6]
    num_rows = len(ordered_models)
    num_cols = len(noises)
    
    if num_rows == 0 or num_cols == 0:
        return
        
    print(f"\nPlotting {num_rows}x{num_cols} Monolithic Model-Perturbation KDE Grid...")
    
    # Dynamic Figure Sizing (Roughly 24x16 for an 8x6 matrix)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 4, num_rows * 2.5))
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)
    if num_cols == 1: axes = np.expand_dims(axes, axis=1)
    
    for r, (dataset, model) in enumerate(ordered_models):
        for c, noise in enumerate(noises):
            ax1 = axes[r, c]
            ax2 = ax1.twinx() # Secondary axis for Penalties
            
            subset = df_clean[(df_clean['Model'] == model) & (df_clean['Noise_Type'] == noise)]
            ic_data = subset[subset['Context'] == 'In-Context']
            oc_data = subset[subset['Context'] == 'Out-of-Context']
            
            # Plot Utility
            if not ic_data.empty: sns.kdeplot(data=ic_data, x='Utility', color=ic_color, fill=True, alpha=0.3, linewidth=2, ax=ax1, warn_singular=False)
            if not oc_data.empty: sns.kdeplot(data=oc_data, x='Utility', color=oc_color, fill=True, alpha=0.3, linewidth=2, ax=ax1, warn_singular=False)
            
            # Plot Penalties
            if not ic_data.empty:
                sns.kdeplot(data=ic_data, x='CSR', color=ic_color, fill=False, linestyle='--', linewidth=1.5, ax=ax2, warn_singular=False)
                sns.kdeplot(data=ic_data, x='RP', color=ic_color, fill=False, linestyle=':', linewidth=1.5, ax=ax2, warn_singular=False)
            if not oc_data.empty:
                sns.kdeplot(data=oc_data, x='CSR', color=oc_color, fill=False, linestyle='--', linewidth=1.5, ax=ax2, warn_singular=False)
                sns.kdeplot(data=oc_data, x='RP', color=oc_color, fill=False, linestyle=':', linewidth=1.5, ax=ax2, warn_singular=False)
            
            # Strict constraints & clean visuals
            ax1.set_xlim(-5, 105)
            ax1.set_xlabel(""); ax1.set_ylabel(""); ax2.set_ylabel("")
            
            # Remove ALL ticks & labels
            ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax2.set_yticklabels([])
            ax1.tick_params(bottom=False, left=False); ax2.tick_params(right=False)
            ax2.grid(False)
            
            # Column Header (Noise Names) - only on top row
            if r == 0:
                ax1.set_title(noise, fontsize=20, fontweight='bold', pad=15)
                
            # Row Header (Model ONLY) - only on leftmost column
            if c == 0:
                ax1.set_ylabel(f"{model}", fontsize=18, fontweight='bold', 
                               rotation=0, ha='right', va='center', labelpad=15)

    # Global Axes
    fig.supxlabel("Metric Scores (%)", fontsize=28, fontweight='bold', y=0.04)
    # fig.supylabel("Distribution Density", fontsize=28, fontweight='bold', x=0.01)

    # Force draw to get geometric coordinates for the vertical dataset names
    fig.canvas.draw()
    
    for dataset, row_indices in dataset_rows.items():
        if not row_indices: continue
        
        # Geometrically find center Y between the associated model rows
        y_centers = [axes[r, 0].get_position().y0 + (axes[r, 0].get_position().height / 2.0) for r in row_indices]
        avg_y = sum(y_centers) / len(y_centers)
        
        # Position X to the left of the subplot boundary and model names
        x_pos = axes[0, 0].get_position().x0 - 0.08
        
        # Vertical dataset name, no bbox frame
        fig.text(x_pos, avg_y, dataset, rotation=90, va='center', ha='center', 
                 fontsize=22, fontweight='bold')

    # Build Unified Legend
    legend_elements = [
        Line2D([0], [0], color=ic_color, lw=4, label='IC Score (Filled)'),
        Line2D([0], [0], color=ic_color, lw=3, linestyle='--', label='IC CSR (Dashed)'),
        Line2D([0], [0], color=ic_color, lw=3, linestyle=':', label='IC RP (Dotted)'),
        Line2D([0], [0], color=oc_color, lw=4, label='OC Score (Filled)'),
        Line2D([0], [0], color=oc_color, lw=3, linestyle='--', label='OC CSR (Dashed)'),
        Line2D([0], [0], color=oc_color, lw=3, linestyle=':', label='OC RP (Dotted)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=3, fontsize=22, frameon=True, edgecolor='black', handlelength=3.0)
               
    plt.suptitle("Model Perturbation-Wise KDE Plots", fontsize=32, fontweight='bold', y=0.98)
    
    # Pack layout tightly, leave space on left for vertical dataset names and bottom for Legend
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.92, bottom=0.09, left=0.16)
    
    save_pdf = os.path.join(output_dir, "Monolithic_Model_Perturbation_KDE_Grid.pdf")
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
    print(f"-> Monolithic 8x6 KDE grid successfully saved to: {save_pdf}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Comprehensive PRISM KDEs.")
    parser.add_argument('--directories', type=json.loads, 
                        default={
                            "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out", 
                            "SAMv2": "/bhupender/2025/Mouse_other/prism_out", 
                            "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out", 
                            "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out", 
                            "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out", 
                            "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out", 
                            "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out", 
                            "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out", 
                        },
                        help="JSON string of directories")
    args = parser.parse_args()

    print("Extracting Comprehensive Image-Level Data (New Math Engine)...")
    df = extract_comprehensive_data(args.directories)
    
    if df.empty:
        print("No valid data found. Check your directory paths.")
        return
        
    print("\nGenerating Comprehensive 4x2 KDE Grid...")
    plot_comprehensive_kde_grid(df, output_dir="./kde_plots")
    
    plot_model_perturbation_kdes(df, output_dir="./kde_plots")

if __name__ == "__main__":
    main()
# import os
# import json
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtransforms
# from matplotlib.lines import Line2D
# import argparse
# import math
# import logging

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
# # 6-STEP MATHEMATICAL ENGINE
# # ==========================================
# def calculate_step_rp_and_iou(level_0, level_k):
#     """Calculates step-wise Raw IoU and Absolute Rank Perplexity (RP)."""
#     sp_0 = {sp['sp_id']: sp.get('rank', 0) for sp in level_0.get('superpixels', [])}
#     sp_k = {sp['sp_id']: sp.get('rank', 0) for sp in level_k.get('superpixels', [])}
    
#     N = len(sp_0)
#     if N <= 1:
#         rp = 0.0
#     else:
#         max_jumps = math.floor((N**2) / 2.0)
#         total_jumps = sum(abs(sp_k.get(sp_id, sp_0[sp_id]) - sp_0[sp_id]) for sp_id in sp_0)
#         rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
        
#     raw_iou = np.mean([sp['iou'] for sp in level_k.get('superpixels', [])]) if level_k.get('superpixels') else 0.0
#     return raw_iou, rp

# def extract_run_level_data(directories):
#     """
#     Extracts run-level data exactly mimicking the Table Aggregator's os.walk behavior
#     and strict 6-Step Hierarchical Math.
#     """
#     data_points = []
    
#     for model_name, base_dir in directories.items():
#         if not os.path.exists(base_dir):
#             continue
            
#         dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
        
#         # Exact os.walk crawl matching your table script to guarantee 1:1 data parity
#         for root, dirs, files in os.walk(base_dir):
#             for file in files:
#                 if file.endswith("_analysis.json"):
#                     parts = root.replace('\\', '/').split('/')
#                     run_name = "run_1"
#                     for p in parts:
#                         if p.startswith("run_"):
#                             run_name = p
#                             break
                    
#                     try: run_num = int(run_name.split("_")[-1])
#                     except ValueError: run_num = 1
                    
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
#                         max_steps = max(len(ic_levels), len(oc_levels))
#                         if max_steps <= 1: continue
                        
#                         # Extract Central Image-Wide CSR
#                         ic_img_csr = ic_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0)
#                         oc_img_csr = oc_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0)
                        
#                         ic_lvl_0, oc_lvl_0 = ic_levels[0], oc_levels[0]
#                         ic_step_discounted, oc_step_discounted = [], []
                        
#                         # STEP 1: Local Step-Wise Discounting (RP Only)
#                         for k in range(1, max_steps):
#                             ic_lvl_k = ic_levels[k] if k < len(ic_levels) else ic_levels[-1]
#                             oc_lvl_k = oc_levels[k] if k < len(oc_levels) else oc_levels[-1]
                            
#                             ic_raw, ic_rp = calculate_step_rp_and_iou(ic_lvl_0, ic_lvl_k)
#                             oc_raw, oc_rp = calculate_step_rp_and_iou(oc_lvl_0, oc_lvl_k)
                            
#                             ic_step_discounted.append(ic_raw * math.exp(-ic_rp))
#                             oc_step_discounted.append(oc_raw * math.exp(-oc_rp))
                        
#                         # STEP 2: Trajectory Averaging
#                         avg_rp_discounted_ic = np.mean(ic_step_discounted)
#                         avg_rp_discounted_oc = np.mean(oc_step_discounted)
                        
#                         # STEP 3: Global Spillover Discount
#                         E_ic_noise = avg_rp_discounted_ic * math.exp(-ic_img_csr)
#                         E_oc_noise = avg_rp_discounted_oc * math.exp(-oc_img_csr)
                        
#                         # Fix applied: Removed the `* 100.0` here that caused values in the 1000s
#                         data_points.append({"Dataset": dataset_name, "Model": model_name, "Run": run_num, "Context": "In-Context", "Noise Type": noise_type, "Reliability": E_ic_noise})
#                         data_points.append({"Dataset": dataset_name, "Model": model_name, "Run": run_num, "Context": "Out-of-Context", "Noise Type": noise_type, "Reliability": E_oc_noise})
                            
#     return pd.DataFrame(data_points)

# def extract_intensity_level_data(directories):
#     """Extracts step-wise Raw IoU degradation using identical os.walk parsing."""
#     level_points = []
    
#     for model_name, base_dir in directories.items():
#         if not os.path.exists(base_dir): continue
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
                            
#                     json_path = os.path.join(root, file)
#                     try: data = load_json(json_path)
#                     except Exception: continue
                    
#                     if "in_context_analysis" not in data: continue
                            
#                     for raw_noise, noise_data in data["in_context_analysis"].items():
#                         noise_type = clean_noise_name(raw_noise)
#                         levels = noise_data.get("level_data", [])
#                         for level_entry in levels:
#                             n_level = level_entry.get("noise_level", 0.0)
#                             sp_data = level_entry.get("superpixels", [])
                            
#                             if sp_data:
#                                 mean_iou = np.nanmean([sp.get("iou", 0.0) for sp in sp_data])
#                                 level_points.append({
#                                     "Dataset": dataset_name,
#                                     "Model": model_name,
#                                     "Run": run_name,
#                                     "Noise Type": noise_type,
#                                     "Noise Level": n_level,
#                                     "IoU": mean_iou
#                                 })
                            
#     return pd.DataFrame(level_points)


# # ==========================================
# # PLOTTING FUNCTIONS
# # ==========================================
# def plot_comprehensive_grid(df, output_dir="./variance_plots"):
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Safe Auto-Scaler
#     if df['Reliability'].max() <= 1.05:
#         df['Reliability'] = df['Reliability'] * 100.0

#     # Numeric Index map for Seaborn continuous plotting bug fix
#     noise_order = sorted(df['Noise Type'].unique())
#     noise_mapping = {n: i for i, n in enumerate(noise_order)}
#     df['Noise_Idx'] = df['Noise Type'].map(noise_mapping)
    
#     context_colors = {"In-Context": "dodgerblue", "Out-of-Context": "darkorange"}
#     sns.set_theme(style="whitegrid", context="paper")
    
#     unique_datasets = [ds for ds in ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD'] if ds in df['Dataset'].unique()]
    
#     plot_pairs = []
#     for dataset in unique_datasets:
#         dataset_models = sorted(df[df['Dataset'] == dataset]['Model'].unique())
#         for model in dataset_models:
#             plot_pairs.append((dataset, model))
            
#     plot_pairs = plot_pairs[:8]
    
#     fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), sharey=True, sharex=True)
    
#     for i, (dataset, model) in enumerate(plot_pairs):
#         row_idx = i // 4
#         col_idx = i % 4
        
#         ax = axes[row_idx, col_idx]
#         subset = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
        
#         if subset.empty:
#             ax.set_visible(False)
#             continue
            
#         sns.lineplot(
#             data=subset, x="Noise_Idx", y="Reliability", hue="Context", 
#             style="Context", palette=context_colors, 
#             dashes={"In-Context": (1, 0), "Out-of-Context": (3, 3)}, 
#             linewidth=2.5, markers=True, markersize=8, 
#             errorbar='sd', ax=ax, legend=False,
#             err_kws={'alpha': 0.3} 
#         )
        
#         ax.set_title(f"{dataset} | {model}", fontsize=20, fontweight='bold', pad=10)
#         ax.set_ylim(-5, 105)
#         ax.tick_params(axis='y', which='major', labelsize=18)
        
#         if row_idx == 1 or i >= len(plot_pairs) - 4:
#             ax.set_xticks(range(len(noise_order)))
#             labels = [n.replace(" & ", " &\n") for n in noise_order]
#             ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=20)
#             ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
#         else:
#             ax.set_xticks(range(len(noise_order)))
#             ax.set_xticklabels([])
#             ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
            
#         ax.set_ylabel("Robustness Score (%)", fontsize=20) if col_idx == 0 else ax.set_ylabel("")
#         ax.set_xlabel("")

#     for i in range(len(plot_pairs), 8):
#         row_idx = i // 4
#         col_idx = i % 4
#         axes[row_idx, col_idx].set_visible(False)

#     context_handles = [
#         Line2D([0], [0], color=context_colors['In-Context'], lw=3, label='In-Context'),
#         Line2D([0], [0], color=context_colors['Out-of-Context'], lw=3, linestyle='--', label='Out-of-Context')
#     ]
    
#     fig.legend(handles=context_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
#                ncol=2, fontsize=20, frameon=True, edgecolor='black')
    
#     plt.suptitle("Dataset Variance Across Runs: Robustness vs Noise Type (6-Step Math)", fontsize=24, fontweight='bold', y=0.98)
#     plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
#     save_pdf = os.path.join(output_dir, "Unified_Dataset_Variance_Matrix_2x4.pdf")
#     plt.savefig(save_pdf)
#     plt.close()
#     print(f"Comprehensive 2x4 matrix grid successfully compiled and written to -> {save_pdf}")


# def plot_ic_oc_heatmap(df, output_dir="./variance_plots"):
#     os.makedirs(output_dir, exist_ok=True)
#     sns.set_theme(style="whitegrid")

#     # Safe Auto-Scaler
#     if df['Reliability'].max() <= 1.05:
#         df['Reliability'] = df['Reliability'] * 100.0

#     df_mean = df.groupby(["Model", "Noise Type", "Context"])["Reliability"].mean().reset_index()

#     df_mean["Dataset"] = df_mean["Model"].map(DATASET_MAPPING)
    
#     ordered_datasets = ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD']
#     ordered_models = []
#     for ds in ordered_datasets:
#         if ds in df_mean["Dataset"].unique():
#             m_list = sorted(df_mean[df_mean['Dataset'] == ds]['Model'].unique())
#             ordered_models.extend(m_list)

#     ic_pivot = df_mean[df_mean["Context"] == "In-Context"].pivot(
#         index="Model", columns="Noise Type", values="Reliability"
#     ).reindex(ordered_models)
    
#     oc_pivot = df_mean[df_mean["Context"] == "Out-of-Context"].pivot(
#         index="Model", columns="Noise Type", values="Reliability"
#     ).reindex(ordered_models)

#     fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
#     cmap = sns.color_palette("YlGnBu", as_cmap=True)

#     sns.heatmap(
#         ic_pivot, annot=True, fmt=".1f", cmap=cmap, ax=axes[0],
#         vmin=0, vmax=100, annot_kws={"size": 20},
#         cbar_kws={"label": "Regional Score ($E_{IC}$ %)"},
#     )
#     axes[0].set_title("In-Context (IC) Perturbation", fontsize=24, fontweight="bold", pad=15)
#     axes[0].set_ylabel("Model (Grouped by Dataset)", fontsize=24)
#     axes[0].set_xlabel("Noise Type", fontsize=24)
#     axes[0].collections[0].colorbar.set_label("Regional Score ($E_{IC}$ %)", size=24)  
#     axes[0].collections[0].colorbar.ax.tick_params(labelsize=20)  

#     sns.heatmap(
#         oc_pivot, annot=True, fmt=".1f", cmap=cmap, ax=axes[1],
#         vmin=0, vmax=100, annot_kws={"size": 20},  
#         cbar_kws={"label": "Regional Score ($E_{OC}$ %)"},
#     )
#     axes[1].set_title("Out-of-Context (OC) Perturbation", fontsize=24, fontweight="bold", pad=15)
#     axes[1].set_ylabel("")
#     axes[1].set_xlabel("Noise Type", fontsize=24)
#     axes[1].collections[0].colorbar.set_label("Regional Score ($E_{OC}$ %)", size=22)  
#     axes[1].collections[0].colorbar.ax.tick_params(labelsize=18)  

#     for ax in axes:
#         ax.tick_params(axis="x", rotation=60, labelsize=24)
#         ax.tick_params(axis="y", labelsize=24)

#     plt.tight_layout()
#     save_path = os.path.join(output_dir, "Model_Robustness_Heatmap.pdf")
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()
#     print(f"Heatmap successfully written to -> {save_path}")


# def plot_iou_intensity_curves(df_levels, output_dir="./variance_plots"):
#     if df_levels.empty: return
    
#     os.makedirs(output_dir, exist_ok=True)

#     # Safe Auto-Scaler
#     if df_levels['IoU'].max() <= 1.05:
#         df_levels['IoU'] = df_levels['IoU'] * 100.0

#     sns.set_theme(style="whitegrid", context="paper")
    
#     g = sns.FacetGrid(df_levels, col="Noise Type", col_wrap=3, height=4, aspect=1.25, sharey=True)

#     g.map_dataframe(sns.lineplot, x="Noise Level", y="IoU", hue="Model", 
#                     style="Model", markers=False, dashes=False, linewidth=2.5, errorbar='sd')
    
#     g.set_axis_labels("", "")
#     g.set_titles(col_template="{col_name}", size=24)
    
#     for ax in g.axes.flat:
#         ax.set_ylim(0, 105)
#         ax.tick_params(labelsize=22)
#         ax.grid(True, linestyle=':', alpha=0.7)
    
#     num_models = df_levels['Model'].nunique()
#     legend_cols = min(num_models, 4)
    
#     g.fig.supxlabel("Perturbation Intensity ", fontsize=26, fontweight='bold', y=0.15)
#     g.fig.supylabel("Structural Integrity (Raw IoU %)", fontsize=26, fontweight='bold', x=0.03)
    
#     g.add_legend(fontsize=26, title_fontsize=22, 
#                  bbox_to_anchor=(0.5, 0.13), loc='upper center', ncol=legend_cols, frameon=True)
    
#     plt.suptitle("Raw Performance Drop with Increasing Perturbation Intensity", y=1.02, fontsize=28, fontweight='bold')
    
#     plt.tight_layout()
#     g.fig.subplots_adjust(top=0.92, bottom=0.25, left=0.08)
    
#     save_path = os.path.join(output_dir, "IoU_vs_Intensity_Degradation.pdf")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"Intensity Curves successfully written to -> {save_path}")
    

# def main():
#     parser = argparse.ArgumentParser(description="Plot Dataset-Oriented Variance Grid and Heatmaps.")
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
#     parser.add_argument('--output_dir', type=str,default='./variance_plots_new' )
#     args = parser.parse_args()

#     print("Extracting penalized run-level data mapped by dataset (6-Step Math)...")
#     df = extract_run_level_data(args.directories)
    
#     if df.empty:
#         print("Extraction failed. Verify paths inside default configuration dictionary.")
#         return
        
#     print("Compiling grid visualizations...")
#     plot_comprehensive_grid(df, output_dir=args.output_dir)

#     print("\nCompiling IC vs OC Matrix Heatmap (Sorted by Dataset Order)...")
#     plot_ic_oc_heatmap(df, output_dir=args.output_dir)
    
#     print("\nExtracting step-by-step perturbation level data...")
#     df_levels = extract_intensity_level_data(args.directories)
    
#     if not df_levels.empty:
#         print("Compiling Intensity Degradation Curves...")
#         plot_iou_intensity_curves(df_levels, output_dir=args.output_dir)
    
#     print("\nAll supplementary plots generated successfully.")

# if __name__ == "__main__":
#     main()



import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import argparse
import math
import logging

# ==========================================
# 1. DATASET MAPPING & CONSTANTS
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
# NEW MATHEMATICAL ENGINE: Trajectory Integration
# ==========================================
def extract_run_level_data(directories):
    """
    Extracts run-level data exactly mimicking the Table Aggregator's os.walk behavior,
    but applying the new Independent Trajectory Integration math.
    """
    data_points = []
    
    for model_name, base_dir in directories.items():
        if not os.path.exists(base_dir):
            continue
            
        dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
        
        # Exact os.walk crawl to guarantee multiple run extraction
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith("_analysis.json"):
                    parts = root.replace('\\', '/').split('/')
                    run_name = "run_1"
                    for p in parts:
                        if p.startswith("run_"):
                            run_name = p
                            break
                    
                    try: run_num = int(run_name.split("_")[-1])
                    except ValueError: run_num = 1
                    
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
                        
                        IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels]
                        OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels]

                        N_ic, N_oc = len(IC_vals), len(OC_vals)
                        if N_ic <= 1 or N_oc <= 1: continue

                        IC_0 = max(1e-6, IC_vals[0])
                        OC_0 = max(1e-6, OC_vals[0])
                        
                        # 1. Independent IC Integration
                        W_ic, A_ic = 0.0, 0.0
                        for k in range(1, N_ic):
                            eta_k = k / (N_ic - 1)
                            w_k = math.exp(-GAMMA * eta_k)
                            A_ic += w_k * (IC_vals[k] / IC_0)
                            W_ic += w_k
                        norm_A_ic = A_ic / W_ic if W_ic > 0 else 0.0

                        # 2. Independent OC Integration
                        W_oc, A_oc = 0.0, 0.0
                        for k in range(1, N_oc):
                            eta_k = k / (N_oc - 1)
                            w_k = math.exp(-GAMMA * eta_k)
                            A_oc += w_k * (OC_vals[k] / OC_0)
                            W_oc += w_k
                        norm_A_oc = A_oc / W_oc if W_oc > 0 else 0.0
                        
                        # Strictly Bounded to prevent hallucination artifacts > 1.0 from skewing
                        A_ic_final = min(1.0, norm_A_ic) * 100.0
                        A_oc_final = min(1.0, norm_A_oc) * 100.0
                        
                        data_points.append({"Dataset": dataset_name, "Model": model_name, "Run": run_num, "Context": "In-Context", "Noise Type": noise_type, "Reliability": A_ic_final})
                        data_points.append({"Dataset": dataset_name, "Model": model_name, "Run": run_num, "Context": "Out-of-Context", "Noise Type": noise_type, "Reliability": A_oc_final})
                            
    return pd.DataFrame(data_points)

def extract_intensity_level_data(directories):
    """Extracts step-wise Raw IoU degradation using identical os.walk parsing."""
    level_points = []
    
    for model_name, base_dir in directories.items():
        if not os.path.exists(base_dir): continue
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
                            
                    json_path = os.path.join(root, file)
                    try: data = load_json(json_path)
                    except Exception: continue
                    
                    if "in_context_analysis" not in data: continue
                            
                    for raw_noise, noise_data in data["in_context_analysis"].items():
                        noise_type = clean_noise_name(raw_noise)
                        levels = noise_data.get("level_data", [])
                        for level_entry in levels:
                            n_level = level_entry.get("noise_level", 0.0)
                            sp_data = level_entry.get("superpixels", [])
                            
                            if sp_data:
                                mean_iou = np.nanmean([sp.get("iou", 0.0) for sp in sp_data])
                                level_points.append({
                                    "Dataset": dataset_name,
                                    "Model": model_name,
                                    "Run": run_name,
                                    "Noise Type": noise_type,
                                    "Noise Level": n_level,
                                    "IoU": mean_iou
                                })
                            
    return pd.DataFrame(level_points)


# ==========================================
# PLOTTING FUNCTIONS (Format Strictly Preserved)
# ==========================================
def plot_comprehensive_grid(df, output_dir="./variance_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Safe Auto-Scaler
    if df['Reliability'].max() <= 1.05:
        df['Reliability'] = df['Reliability'] * 100.0

    # Numeric Index map for Seaborn continuous plotting bug fix
    noise_order = sorted(df['Noise Type'].unique())
    noise_mapping = {n: i for i, n in enumerate(noise_order)}
    df['Noise_Idx'] = df['Noise Type'].map(noise_mapping)
    
    context_colors = {"In-Context": "dodgerblue", "Out-of-Context": "darkorange"}
    sns.set_theme(style="whitegrid", context="paper")
    
    unique_datasets = [ds for ds in ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD'] if ds in df['Dataset'].unique()]
    
    plot_pairs = []
    for dataset in unique_datasets:
        dataset_models = sorted(df[df['Dataset'] == dataset]['Model'].unique())
        for model in dataset_models:
            plot_pairs.append((dataset, model))
            
    plot_pairs = plot_pairs[:8]
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), sharey=True, sharex=True)
    
    for i, (dataset, model) in enumerate(plot_pairs):
        row_idx = i // 4
        col_idx = i % 4
        
        ax = axes[row_idx, col_idx]
        subset = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
        
        if subset.empty:
            ax.set_visible(False)
            continue
            
        sns.lineplot(
            data=subset, x="Noise_Idx", y="Reliability", hue="Context", 
            style="Context", palette=context_colors, 
            dashes={"In-Context": (1, 0), "Out-of-Context": (3, 3)}, 
            linewidth=2.5, markers=True, markersize=8, 
            errorbar='sd', ax=ax, legend=False,
            err_kws={'alpha': 0.3} 
        )
        
        ax.set_title(f"{dataset} | {model}", fontsize=20, fontweight='bold', pad=10)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='y', which='major', labelsize=18)
        
        if row_idx == 1 or i >= len(plot_pairs) - 4:
            ax.set_xticks(range(len(noise_order)))
            labels = [n.replace(" & ", " &\n") for n in noise_order]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=20)
            ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
        else:
            ax.set_xticks(range(len(noise_order)))
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
            
        ax.set_ylabel("Regional Utility ($A$) %", fontsize=20) if col_idx == 0 else ax.set_ylabel("")
        ax.set_xlabel("")

    for i in range(len(plot_pairs), 8):
        row_idx = i // 4
        col_idx = i % 4
        axes[row_idx, col_idx].set_visible(False)

    context_handles = [
        Line2D([0], [0], color=context_colors['In-Context'], lw=3, label='In-Context ($E_{ic}$)'),
        Line2D([0], [0], color=context_colors['Out-of-Context'], lw=3, linestyle='--', label='Out-of-Context ($E_{oc}$)')
    ]
    
    fig.legend(handles=context_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, fontsize=20, frameon=True, edgecolor='black')
    
    plt.suptitle("Dataset Variance Across Runs: Regional Scores vs Perturbation Type ", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    save_pdf = os.path.join(output_dir, "Unified_Dataset_Variance_Matrix_2x4.pdf")
    plt.savefig(save_pdf)
    plt.close()
    print(f"Comprehensive 2x4 matrix grid successfully compiled and written to -> {save_pdf}")


def plot_ic_oc_heatmap(df, output_dir="./variance_plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Safe Auto-Scaler
    if df['Reliability'].max() <= 1.05:
        df['Reliability'] = df['Reliability'] * 100.0

    df_mean = df.groupby(["Model", "Noise Type", "Context"])["Reliability"].mean().reset_index()

    df_mean["Dataset"] = df_mean["Model"].map(DATASET_MAPPING)
    
    ordered_datasets = ['Pascal-VOC', 'ISIC-2016', 'Synapse', 'XCAD']
    ordered_models = []
    for ds in ordered_datasets:
        if ds in df_mean["Dataset"].unique():
            m_list = sorted(df_mean[df_mean['Dataset'] == ds]['Model'].unique())
            ordered_models.extend(m_list)

    ic_pivot = df_mean[df_mean["Context"] == "In-Context"].pivot(
        index="Model", columns="Noise Type", values="Reliability"
    ).reindex(ordered_models)
    
    oc_pivot = df_mean[df_mean["Context"] == "Out-of-Context"].pivot(
        index="Model", columns="Noise Type", values="Reliability"
    ).reindex(ordered_models)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    sns.heatmap(
        ic_pivot, annot=True, fmt=".1f", cmap=cmap, ax=axes[0],
        vmin=0, vmax=100, annot_kws={"size": 20},
        cbar_kws={"label": "Regional Score ($E_{ic}$ %)"},
    )
    axes[0].set_title("In-Context (IC) Perturbation", fontsize=24, fontweight="bold", pad=15)
    axes[0].set_ylabel("Model (Grouped by Dataset)", fontsize=24)
    axes[0].set_xlabel("Noise Type", fontsize=24)
    axes[0].collections[0].colorbar.set_label("Average $E_{ic}$ %", size=24)  
    axes[0].collections[0].colorbar.ax.tick_params(labelsize=20)  

    sns.heatmap(
        oc_pivot, annot=True, fmt=".1f", cmap=cmap, ax=axes[1],
        vmin=0, vmax=100, annot_kws={"size": 20},  
        cbar_kws={"label": "Regional Score ($E_{oc}$ %)"},
    )
    axes[1].set_title("Out-of-Context (OC) Perturbation", fontsize=24, fontweight="bold", pad=15)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Noise Type", fontsize=24)
    axes[1].collections[0].colorbar.set_label("Average $E_{oc}$ %", size=24)  
    axes[1].collections[0].colorbar.ax.tick_params(labelsize=18)  

    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=24)
        ax.tick_params(axis="y", labelsize=24)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Model_Robustness_Heatmap.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Heatmap successfully written to -> {save_path}")


def plot_iou_intensity_curves(df_levels, output_dir="./variance_plots"):
    if df_levels.empty: return
    
    os.makedirs(output_dir, exist_ok=True)

    # Safe Auto-Scaler
    if df_levels['IoU'].max() <= 1.05:
        df_levels['IoU'] = df_levels['IoU'] * 100.0

    sns.set_theme(style="whitegrid", context="paper")
    
    g = sns.FacetGrid(df_levels, col="Noise Type", col_wrap=3, height=4, aspect=1.25, sharey=True)

    g.map_dataframe(sns.lineplot, x="Noise Level", y="IoU", hue="Model", 
                    style="Model", markers=False, dashes=False, linewidth=2.5, errorbar='sd')
    
    g.set_axis_labels("", "")
    g.set_titles(col_template="{col_name}", size=24)
    
    for ax in g.axes.flat:
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=22)
        ax.grid(True, linestyle=':', alpha=0.7)
    
    num_models = df_levels['Model'].nunique()
    legend_cols = min(num_models, 4)
    
    g.fig.supxlabel("Perturbation Intensity ", fontsize=26, fontweight='bold', y=0.15)
    g.fig.supylabel("IoU %", fontsize=26, fontweight='bold', x=0.03)
    
    g.add_legend(fontsize=26, title_fontsize=22, 
                 bbox_to_anchor=(0.5, 0.13), loc='upper center', ncol=legend_cols, frameon=True)
    
    plt.suptitle("Performance (IoU) Drop with Increasing Perturbation Intensity", y=1.02, fontsize=28, fontweight='bold')
    
    plt.tight_layout()
    g.fig.subplots_adjust(top=0.92, bottom=0.25, left=0.08)
    
    save_path = os.path.join(output_dir, "IoU_vs_Intensity_Degradation.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Intensity Curves successfully written to -> {save_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Plot Dataset-Oriented Variance Grid and Heatmaps.")
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
    parser.add_argument('--output_dir', type=str,default='./variance_plots_new' )
    args = parser.parse_args()

    print("Extracting penalized run-level data mapped by dataset (Trajectory Integration)...")
    df = extract_run_level_data(args.directories)
    
    if df.empty:
        print("Extraction failed. Verify paths inside default configuration dictionary.")
        return
        
    print("Compiling grid visualizations...")
    plot_comprehensive_grid(df, output_dir=args.output_dir)

    print("\nCompiling IC vs OC Matrix Heatmap (Sorted by Dataset Order)...")
    plot_ic_oc_heatmap(df, output_dir=args.output_dir)
    
    print("\nExtracting step-by-step perturbation level data...")
    df_levels = extract_intensity_level_data(args.directories)
    
    if not df_levels.empty:
        print("Compiling Intensity Degradation Curves...")
        plot_iou_intensity_curves(df_levels, output_dir=args.output_dir)
    
    print("\nAll supplementary plots generated successfully.")

if __name__ == "__main__":
    main()
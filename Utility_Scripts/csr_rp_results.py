# import os
# import json
# import argparse
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from math import exp

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

# def extract_aligned_stepwise_data(directories):
#     """
#     Crawls directories, aligns IC and OC data by noise step, calculates instant 
#     discounts, step-wise E_IC, E_OC, and the Harmonic Mean (R_k).
#     """
#     data_points = []
    
#     for model_name, base_dir in directories.items():
#         print(f"Extracting aligned step-wise data for model: {model_name}...")
#         if not os.path.exists(base_dir):
#             print(f"  -> Warning: Directory not found for {model_name}: {base_dir}")
#             continue
            
#         dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
            
#         for run_folder in os.listdir(base_dir):
#             run_path = os.path.join(base_dir, run_folder)

#             # if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
#             #     continue
            
#             # ---> RESTRICT TO run_1 ONLY <---
#             if not os.path.isdir(run_path) or run_folder != "run_1":
#                 continue
                
#             for img_folder in os.listdir(run_path):
#                 img_path = os.path.join(run_path, img_folder)
#                 if not os.path.isdir(img_path):
#                     continue
                    
#                 json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
#                 if not os.path.exists(json_file):
#                     continue
                    
#                 try:
#                     data = load_json(json_file)
#                 except Exception:
#                     continue
                
#                 if "in_context_analysis" not in data or "out_of_context_analysis" not in data:
#                     continue

#                 # Process ALL noise types present in this JSON
#                 noise_types = set(data["in_context_analysis"].keys()).intersection(set(data["out_of_context_analysis"].keys()))

#                 for noise_type in noise_types:
#                     ic_levels = data["in_context_analysis"][noise_type].get("level_data", [])
#                     oc_levels = data["out_of_context_analysis"][noise_type].get("level_data", [])
                    
#                     if not ic_levels or not oc_levels:
#                         continue
                    
#                     # Ensure alignment (truncate to the shortest to be mathematically safe)
#                     min_steps = min(len(ic_levels), len(oc_levels))
                    
#                     ic_rank_history = {}
#                     oc_rank_history = {}

#                     for step_idx in range(min_steps):
#                         ic_lvl = ic_levels[step_idx]
#                         oc_lvl = oc_levels[step_idx]
                        
#                         noise_level = ic_lvl.get("noise_level", 0.0)
                        
#                         # --- IN-CONTEXT PROCESSING ---
#                         ic_csr_raw = ic_lvl.get("instant_spillover_ratio", 0.0)
#                         ic_sps = ic_lvl.get("superpixels", [])
#                         raw_iou_ic = np.mean([sp["iou"] for sp in ic_sps]) if ic_sps else 0.0
                        
#                         for sp in ic_sps:
#                             sp_id = sp["sp_id"]
#                             ic_rank_history.setdefault(sp_id, []).append(sp.get("rank", 0))
                        
#                         ic_total_jumps = sum(abs(ranks[-1] - ranks[0]) for ranks in ic_rank_history.values())
#                         ic_num_sps = len(ic_rank_history)
#                         ic_rp_raw = (ic_total_jumps / ic_num_sps) / ic_num_sps if ic_num_sps > 0 else 0.0
                        
#                         # Apply Discounts
#                         E_ic = raw_iou_ic * exp(-1.0 * ic_csr_raw) * exp(-1.0 * ic_rp_raw)
                        
#                         # --- OUT-OF-CONTEXT PROCESSING ---
#                         oc_csr_raw = oc_lvl.get("instant_spillover_ratio", 0.0)
#                         oc_sps = oc_lvl.get("superpixels", [])
#                         raw_iou_oc = np.mean([sp["iou"] for sp in oc_sps]) if oc_sps else 0.0
                        
#                         for sp in oc_sps:
#                             sp_id = sp["sp_id"]
#                             oc_rank_history.setdefault(sp_id, []).append(sp.get("rank", 0))
                        
#                         oc_total_jumps = sum(abs(ranks[-1] - ranks[0]) for ranks in oc_rank_history.values())
#                         oc_num_sps = len(oc_rank_history)
#                         oc_rp_raw = (oc_total_jumps / oc_num_sps) / oc_num_sps if oc_num_sps > 0 else 0.0
                        
#                         # Apply Discounts
#                         E_oc = raw_iou_oc * exp(-1.0 * oc_csr_raw) * exp(-1.0 * oc_rp_raw)
                        
#                         # --- HARMONIC MEAN (FINAL ROBUSTNESS) ---
#                         R_k = (2 * E_ic * E_oc) / (E_ic + E_oc + 1e-6)

#                         data_points.append({
#                             "Dataset": dataset_name,
#                             "Model": model_name,
#                             "Image": img_folder,
#                             "Noise_Type": noise_type,
#                             "Noise_Level": noise_level,
#                             "Raw_IoU_IC": raw_iou_ic,
#                             "Raw_IoU_OC": raw_iou_oc,
#                             "E_IC": E_ic,
#                             "E_OC": E_oc,
#                             "R_k": R_k,
#                             "CSR_IC": ic_csr_raw * 100.0,   # Scaled for Secondary Y
#                             "CSR_OC": oc_csr_raw * 100.0,   # Scaled for Secondary Y
#                             "RP_IC": ic_rp_raw * 100.0,     # Scaled for Secondary Y
#                             "RP_OC": oc_rp_raw * 100.0      # Scaled for Secondary Y
#                         })
                            
#     return pd.DataFrame(data_points)

# def plot_degradation_continuum(df_noise, model_name, noise_type, output_dir):
#     """
#     Plots the final harmonic robustness, discounted E_IC/E_OC, raw IoU fades, 
#     and penalties. Implements dynamic 20% axis scaling.
#     """
#     if df_noise.empty:
#         return
        
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Explicitly calculate mean across images for all numeric columns
#     numeric_cols = ["Raw_IoU_IC", "Raw_IoU_OC", "E_IC", "E_OC", "R_k", "CSR_IC", "CSR_OC", "RP_IC", "RP_OC"]
#     df_mean = df_noise.groupby(["Noise_Level"])[numeric_cols].mean().reset_index()
    
#     fig, ax1 = plt.subplots(figsize=(14, 8))
#     sns.set_theme(style="whitegrid", context="paper")
    
#     ax2 = ax1.twinx() # Create secondary axis for Penalties
    
#     ic_color = "dodgerblue"
#     oc_color = "darkorange"
    
#     # =======================================================
#     # AXIS 1: ROBUSTNESS & EVALUATION METRICS (Descending)
#     # =======================================================
    
#     # 1. Faded Raw Average IoU (What is being discounted)
#     ax1.plot(df_mean['Noise_Level'], df_mean['Raw_IoU_IC'], color=ic_color, linewidth=1, linestyle='--', alpha=0.4, label='Raw Avg IoU (IC)')
#     ax1.plot(df_mean['Noise_Level'], df_mean['Raw_IoU_OC'], color=oc_color, linewidth=1, linestyle='--', alpha=0.4, label='Raw Avg IoU (OC)')
    
#     # 2. Solid Discounted Metrics (E_IC, E_OC)
#     ax1.plot(df_mean['Noise_Level'], df_mean['E_IC'], color=ic_color, linewidth=2, linestyle='-', marker='o', markersize=8, label='E_IC (Discounted)')
#     ax1.plot(df_mean['Noise_Level'], df_mean['E_OC'], color=oc_color, linewidth=2, linestyle='-', marker='o', markersize=8, label='E_OC (Discounted)')
    
#     # 3. Final Step-Wise Robustness (Harmonic Mean)
#     ax1.plot(df_mean['Noise_Level'], df_mean['R_k'], color='black', linewidth=4, linestyle='-', marker='D', markersize=9, label='R_k (Final Robustness)')
    
#     ax1.set_xlabel(f"{noise_type.capitalize()} Noise Level", fontsize=16, fontweight='bold', labelpad=10)
#     ax1.set_ylabel("Robustness Score (0-100)", fontsize=16, fontweight='bold', labelpad=10)
#     ax1.tick_params(axis='both', which='major', labelsize=14)

#     # =======================================================
#     # AXIS 2: PENALTIES (Ascending)
#     # =======================================================
#     # Plot CSR
#     ax2.plot(df_mean['Noise_Level'], df_mean['CSR_IC'], color=ic_color, linewidth=2, linestyle=':', marker='s', markersize=5, alpha=0.8)
#     ax2.plot(df_mean['Noise_Level'], df_mean['CSR_OC'], color=oc_color, linewidth=2, linestyle=':', marker='s', markersize=5, alpha=0.8)
    
#     # Plot Base-Referenced RP
#     ax2.plot(df_mean['Noise_Level'], df_mean['RP_IC'], color=ic_color, linewidth=2, linestyle='-.', marker='^', markersize=5, alpha=0.8)
#     ax2.plot(df_mean['Noise_Level'], df_mean['RP_OC'], color=oc_color, linewidth=2, linestyle='-.', marker='^', markersize=5, alpha=0.8)
    
#     ax2.set_ylabel("Penalty Scale (CSR & RP vs Base)", fontsize=16, fontweight='bold', labelpad=10)
#     ax2.tick_params(axis='y', which='major', labelsize=14)
#     ax2.grid(False) # Prevent grid overlapping

#     # =======================================================
#     # DYNAMIC 20% SCALING LOGIC
#     # =======================================================
#     # Y1 Limits
#     y1_vals = pd.concat([df_mean['Raw_IoU_IC'], df_mean['Raw_IoU_OC'], df_mean['E_IC'], df_mean['E_OC'], df_mean['R_k']])
#     y1_min, y1_max = y1_vals.min(), y1_vals.max()
#     y1_range = y1_max - y1_min if y1_max != y1_min else 10.0
#     ax1.set_ylim(max(-5, y1_min - 0.20 * y1_range), min(105, y1_max + 0.20 * y1_range))

#     # Y2 Limits
#     y2_vals = pd.concat([df_mean['CSR_IC'], df_mean['CSR_OC'], df_mean['RP_IC'], df_mean['RP_OC']])
#     y2_min, y2_max = y2_vals.min(), y2_vals.max()
#     y2_range = y2_max - y2_min if y2_max != y2_min else 10.0
#     ax2.set_ylim(max(-5, y2_min - 0.20 * y2_range), min(105, y2_max + 0.20 * y2_range))
    
#     # =======================================================
#     # LEGEND CONSTRUCTION
#     # =======================================================
#     # We create custom legend handles for clarity due to the number of lines
#     legend_elements = [
#         Line2D([0], [0], color=ic_color, lw=2, linestyle='--', alpha=0.4, label='Raw IC (Avg IoU)'),
#         Line2D([0], [0], color=ic_color, lw=4, linestyle='-', marker='o', label='E_IC (Discounted)'),
#         Line2D([0], [0], color=oc_color, lw=2, linestyle='--', alpha=0.4, label='Raw OC (Avg IoU)'),
#         Line2D([0], [0], color=oc_color, lw=4, linestyle='-', marker='o', label='E_OC (Discounted)'),
#         Line2D([0], [0], color='black', lw=5, linestyle='-', marker='D', label='R_k (Harmonic Final)'),
#         Line2D([0], [0], color='gray', lw=2, linestyle=':', marker='s', label='Spillover Penalty (CSR)'),
#         Line2D([0], [0], color='gray', lw=2, linestyle='-.', marker='^', label='Rank Volatility (RP)'),
#     ]
    
#     # Two-row legend to prevent stretching too wide
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
#                ncol=4, fontsize=13, frameon=True, edgecolor='black')
               
#     plt.title(f"{model_name} Degradation Dynamics under {noise_type.capitalize()} Noise", fontsize=18, fontweight='bold', pad=15)
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, f"{noise_type}_degradation_continuum.png")
#     plt.savefig(save_path, bbox_inches="tight", dpi=300)
#     plt.close()

# def generate_all_plots(df, base_output_dir="./degradation_plots"):
#     models = df['Model'].unique()
    
#     for model_name in models:
#         model_output_dir = os.path.join(base_output_dir, model_name)
#         os.makedirs(model_output_dir, exist_ok=True)
#         print(f"\nGenerating dynamic degradation plots for {model_name} -> {model_output_dir}/")
        
#         df_model = df[df['Model'] == model_name]
#         noise_types = df_model['Noise_Type'].unique()
        
#         for noise_type in noise_types:
#             df_noise = df_model[df_model['Noise_Type'] == noise_type]
#             plot_degradation_continuum(df_noise, model_name, noise_type, model_output_dir)
#             print(f"  - Saved {noise_type}_degradation_continuum.png")

# def main():
#     parser = argparse.ArgumentParser(description="Plot Step-Wise Degradation Dynamics.")
#     parser.add_argument('--directories', type=json.loads, 
#                         default={
#                             "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/new_expresults_50", 
#                             "SAMv2": "/bhupender/2025/Mouse_other/new_expresults_50", 
#                             "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/new_expresults_50", 
#                             "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/new_expresults_50", 
#                             "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults_50", 
#                             "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults_50", 
#                             "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/new_expresults_50", 
#                             "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/new_expresults_50", 
#                         },
#                         help="JSON mapping of models to inference output directories")
#     parser.add_argument('--output_dir', type=str, default='./degradation_plots_run_1', help="Base directory for nested plots")
#     args = parser.parse_args()

#     print("Extracting Aligned Step-Wise Data Across All Models and Noises...")
#     df = extract_aligned_stepwise_data(args.directories)
    
#     if df.empty:
#         print("No valid data found. Check your directory paths.")
#         return
        
#     generate_all_plots(df, base_output_dir=args.output_dir)
#     print("\nAll step-wise degradation plots generated successfully.")

# if __name__ == "__main__":
#     main()




import os
import json
import argparse
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import exp

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
# 6-STEP MATHEMATICAL ENGINE
# ==========================================
def calculate_step_rp_and_iou(level_0, level_k):
    """Calculates step-wise Raw IoU and Absolute Rank Perplexity (RP) anchored to Step 0."""
    sp_0 = {sp['sp_id']: sp.get('rank', 0) for sp in level_0.get('superpixels', [])}
    sp_k = {sp['sp_id']: sp.get('rank', 0) for sp in level_k.get('superpixels', [])}
    
    N = len(sp_0)
    if N <= 1:
        rp = 0.0
    else:
        # Spearman's Footrule Normalization
        max_jumps = math.floor((N**2) / 2.0)
        total_jumps = sum(abs(sp_k.get(sp_id, sp_0[sp_id]) - sp_0[sp_id]) for sp_id in sp_0)
        rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
        
    raw_iou = np.mean([sp['iou'] for sp in level_k.get('superpixels', [])]) if level_k.get('superpixels') else 0.0
    return raw_iou, rp

def extract_aligned_stepwise_data(directories):
    """
    Crawls directories and extracts data strictly using the 6-Step Hierarchical Math.
    Calculates Step-Wise E_IC and E_OC so that the area under the curve directly matches 
    the Global Regional Scores in the master tables.
    """
    data_points = []
    
    for model_name, base_dir in directories.items():
        print(f"Extracting 6-Step aligned data for model: {model_name}...")
        if not os.path.exists(base_dir):
            print(f"  -> Warning: Directory not found for {model_name}: {base_dir}")
            continue
            
        dataset_name = DATASET_MAPPING.get(model_name, "Unknown Dataset")
            
        for run_folder in os.listdir(base_dir):
            run_path = os.path.join(base_dir, run_folder)

            if not os.path.isdir(run_path) or run_folder != "run_1":
                continue
                
            for img_folder in os.listdir(run_path):
                img_path = os.path.join(run_path, img_folder)
                if not os.path.isdir(img_path):
                    continue
                    
                json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
                if not os.path.exists(json_file):
                    continue
                    
                try:
                    data = load_json(json_file)
                except Exception:
                    continue
                
                ic_data = data.get("in_context_analysis", {})
                oc_data = data.get("out_of_context_analysis", {})
                
                if not ic_data or not oc_data:
                    continue

                shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))

                for raw_noise in shared_noises:
                    noise_type = clean_noise_name(raw_noise)
                    ic_levels = ic_data[raw_noise].get("level_data", [])
                    oc_levels = oc_data[raw_noise].get("level_data", [])
                    
                    if not ic_levels or not oc_levels:
                        continue
                    
                    # 6-Step requires Forward Filling the broken context -> max_steps
                    max_steps = max(len(ic_levels), len(oc_levels))
                    if max_steps <= 1:
                        continue
                        
                    # Extract Central Image-Wide CSR (Global Spillover Discount)
                    ic_img_csr = ic_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0)
                    oc_img_csr = oc_data[raw_noise].get("spillover_metrics", {}).get("conditional_average_spillover", 0.0)

                    ic_lvl_0 = ic_levels[0]
                    oc_lvl_0 = oc_levels[0]

                    for step_idx in range(max_steps):
                        # Forward-fill if one breaks early
                        ic_lvl_k = ic_levels[step_idx] if step_idx < len(ic_levels) else ic_levels[-1]
                        oc_lvl_k = oc_levels[step_idx] if step_idx < len(oc_levels) else oc_levels[-1]
                        
                        # Determine current actual noise level on the X-axis
                        if step_idx < len(ic_levels) and step_idx < len(oc_levels):
                            noise_level = ic_lvl_k.get("noise_level", 0.0)
                        elif step_idx < len(oc_levels):
                            noise_level = oc_lvl_k.get("noise_level", 0.0)
                        else:
                            noise_level = ic_lvl_k.get("noise_level", 0.0)
                        
                        # Calculate anchored step-wise RP and Raw IoU
                        ic_raw, ic_rp = calculate_step_rp_and_iou(ic_lvl_0, ic_lvl_k)
                        oc_raw, oc_rp = calculate_step_rp_and_iou(oc_lvl_0, oc_lvl_k)
                        
                        # Scale Raw IoU for plotting if it is in decimal format
                        ic_raw_scaled = ic_raw * 100.0 if ic_raw <= 1.0 else ic_raw
                        oc_raw_scaled = oc_raw * 100.0 if oc_raw <= 1.0 else oc_raw
                        
                        # Apply Discounts Step-by-Step (RP is local, CSR is global)
                        E_ic = ic_raw_scaled * exp(-ic_rp) * exp(-ic_img_csr)
                        E_oc = oc_raw_scaled * exp(-oc_rp) * exp(-oc_img_csr)
                        
                        # Harmonic Mean (Final Robustness at this step)
                        R_k = (2 * E_ic * E_oc) / (E_ic + E_oc + 1e-6)

                        data_points.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Image": img_folder,
                            "Noise_Type": noise_type,
                            "Noise_Level": noise_level,
                            "Raw_IoU_IC": ic_raw_scaled,
                            "Raw_IoU_OC": oc_raw_scaled,
                            "E_IC": E_ic,
                            "E_OC": E_oc,
                            "R_k": R_k,
                            "CSR_IC": ic_img_csr,   # Scaled for Secondary Y
                            "CSR_OC": oc_img_csr ,   # Scaled for Secondary Y
                            "RP_IC": ic_rp ,         # Scaled for Secondary Y
                            "RP_OC": oc_rp          # Scaled for Secondary Y
                        })
                            
    return pd.DataFrame(data_points)

# ==========================================
# PLOTTING FUNCTIONS (COMPLETELY UNTOUCHED)
# ==========================================
def plot_degradation_continuum(df_noise, model_name, noise_type, output_dir):
    """
    Plots the final harmonic robustness, discounted E_IC/E_OC, raw IoU fades, 
    and penalties. Implements dynamic 20% axis scaling.
    """
    if df_noise.empty:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Explicitly calculate mean across images for all numeric columns
    numeric_cols = ["Raw_IoU_IC", "Raw_IoU_OC", "E_IC", "E_OC", "R_k", "CSR_IC", "CSR_OC", "RP_IC", "RP_OC"]
    df_mean = df_noise.groupby(["Noise_Level"])[numeric_cols].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="paper")
    
    ax2 = ax1.twinx() # Create secondary axis for Penalties
    
    ic_color = "dodgerblue"
    oc_color = "darkorange"
    
    # =======================================================
    # AXIS 1: ROBUSTNESS & EVALUATION METRICS (Descending)
    # =======================================================
    
    # 1. Faded Raw Average IoU (What is being discounted)
    ax1.plot(df_mean['Noise_Level'], df_mean['Raw_IoU_IC'], color=ic_color, linewidth=1, linestyle='--', alpha=0.4, label='Raw Avg IoU (IC)')
    ax1.plot(df_mean['Noise_Level'], df_mean['Raw_IoU_OC'], color=oc_color, linewidth=1, linestyle='--', alpha=0.4, label='Raw Avg IoU (OC)')
    
    # 2. Solid Discounted Metrics (E_IC, E_OC)
    ax1.plot(df_mean['Noise_Level'], df_mean['E_IC'], color=ic_color, linewidth=2, linestyle='-', marker='o', markersize=8, label='E_IC (Discounted)')
    ax1.plot(df_mean['Noise_Level'], df_mean['E_OC'], color=oc_color, linewidth=2, linestyle='-', marker='o', markersize=8, label='E_OC (Discounted)')
    
    # 3. Final Step-Wise Robustness (Harmonic Mean)
    ax1.plot(df_mean['Noise_Level'], df_mean['R_k'], color='black', linewidth=4, linestyle='-', marker='D', markersize=9, label='R_k (Final Robustness)')
    
    ax1.set_xlabel(f"{noise_type.capitalize()} Noise Level", fontsize=16, fontweight='bold', labelpad=10)
    ax1.set_ylabel("Robustness Score (0-100)", fontsize=16, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # =======================================================
    # AXIS 2: PENALTIES (Ascending)
    # =======================================================
    # Plot CSR
    ax2.plot(df_mean['Noise_Level'], df_mean['CSR_IC'], color=ic_color, linewidth=2, linestyle=':', marker='s', markersize=5, alpha=0.8)
    ax2.plot(df_mean['Noise_Level'], df_mean['CSR_OC'], color=oc_color, linewidth=2, linestyle=':', marker='s', markersize=5, alpha=0.8)
    
    # Plot Base-Referenced RP
    ax2.plot(df_mean['Noise_Level'], df_mean['RP_IC'], color=ic_color, linewidth=2, linestyle='-.', marker='^', markersize=5, alpha=0.8)
    ax2.plot(df_mean['Noise_Level'], df_mean['RP_OC'], color=oc_color, linewidth=2, linestyle='-.', marker='^', markersize=5, alpha=0.8)
    
    ax2.set_ylabel("Penalty Scale (CSR & RP vs Base)", fontsize=16, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.grid(False) # Prevent grid overlapping

    # =======================================================
    # DYNAMIC 20% SCALING LOGIC
    # =======================================================
    # Y1 Limits
    y1_vals = pd.concat([df_mean['Raw_IoU_IC'], df_mean['Raw_IoU_OC'], df_mean['E_IC'], df_mean['E_OC'], df_mean['R_k']])
    y1_min, y1_max = y1_vals.min(), y1_vals.max()
    y1_range = y1_max - y1_min if y1_max != y1_min else 10.0
    ax1.set_ylim(max(-5, y1_min - 0.20 * y1_range), min(105, y1_max + 0.20 * y1_range))

    # Y2 Limits
    y2_vals = pd.concat([df_mean['CSR_IC'], df_mean['CSR_OC'], df_mean['RP_IC'], df_mean['RP_OC']])
    y2_min, y2_max = y2_vals.min(), y2_vals.max()
    y2_range = y2_max - y2_min if y2_max != y2_min else 10.0
    ax2.set_ylim(max(-5, y2_min - 0.20 * y2_range), min(105, y2_max + 0.20 * y2_range))
    
    # =======================================================
    # LEGEND CONSTRUCTION
    # =======================================================
    # We create custom legend handles for clarity due to the number of lines
    legend_elements = [
        Line2D([0], [0], color=ic_color, lw=2, linestyle='--', alpha=0.4, label='Raw IC (Avg IoU)'),
        Line2D([0], [0], color=ic_color, lw=4, linestyle='-', marker='o', label='E_IC (Discounted)'),
        Line2D([0], [0], color=oc_color, lw=2, linestyle='--', alpha=0.4, label='Raw OC (Avg IoU)'),
        Line2D([0], [0], color=oc_color, lw=4, linestyle='-', marker='o', label='E_OC (Discounted)'),
        Line2D([0], [0], color='black', lw=5, linestyle='-', marker='D', label='R_k (Harmonic Final)'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', marker='s', label='Spillover Penalty (CSR)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-.', marker='^', label='Rank Volatility (RP)'),
    ]
    
    # Two-row legend to prevent stretching too wide
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
               ncol=4, fontsize=13, frameon=True, edgecolor='black')
               
    plt.title(f"{model_name} Degradation Dynamics under {noise_type.capitalize()} Noise", fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{noise_type}_degradation_continuum.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def generate_all_plots(df, base_output_dir="./degradation_plots"):
    models = df['Model'].unique()
    
    for model_name in models:
        model_output_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"\nGenerating dynamic degradation plots for {model_name} -> {model_output_dir}/")
        
        df_model = df[df['Model'] == model_name]
        noise_types = df_model['Noise_Type'].unique()
        
        for noise_type in noise_types:
            df_noise = df_model[df_model['Noise_Type'] == noise_type]
            plot_degradation_continuum(df_noise, model_name, noise_type, model_output_dir)
            print(f"  - Saved {noise_type}_degradation_continuum.png")

def main():
    parser = argparse.ArgumentParser(description="Plot Step-Wise Degradation Dynamics.")
    parser.add_argument('--directories', type=json.loads, 
                        default={
                            "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/new_expresults_50", 
                            "SAMv2": "/bhupender/2025/Mouse_other/new_expresults_50", 
                            "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/new_expresults_50", 
                            "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/new_expresults_50", 
                            "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults_50", 
                            "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults_50", 
                            "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/new_expresults_50", 
                            "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/new_expresults_50", 
                        },
                        help="JSON mapping of models to inference output directories")
    parser.add_argument('--output_dir', type=str, default='./degradation_plots_run_1', help="Base directory for nested plots")
    args = parser.parse_args()

    print("Extracting 6-Step Aligned Data Across All Models and Noises...")
    df = extract_aligned_stepwise_data(args.directories)
    
    if df.empty:
        print("No valid data found. Check your directory paths.")
        return
        
    generate_all_plots(df, base_output_dir=args.output_dir)
    print("\nAll step-wise degradation plots generated successfully.")

if __name__ == "__main__":
    main()
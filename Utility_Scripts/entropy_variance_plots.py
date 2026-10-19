# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =================================================================================
# # 1. CONFIGURATION & MAPPING
# # =================================================================================
# OUTPUT_DIR = "./cross_model_mean_entropy_plots"

# # The directories pointing to where the Phase 4 ablation JSONs are saved
# DIRECTORIES = {
#     "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/sota_prism_ablation_results", 
#     "SAMv2": "/bhupender/2025/Mouse_other/sota_prism_ablation_results", 
#     "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/sota_prism_ablation_results", 
#     "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/sota_prism_ablation_results", 
#     "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/sota_prism_ablation_results", 
#     "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/sota_prism_ablation_results", 
#     # "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/sota_prism_ablation_results", 
#     # "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/sota_prism_ablation_results", 
# }

# # Grouping the models by dataset for paired comparison
# DATASET_PAIRS = {
#     "Pascal-VOC": ("SAMv1", "SAMv2"),
#     "ISIC-2016": ("MSDUNet", "TransFuse"),
#     "Synapse": ("D-LKA", "TransDeeplab"),
#     # "XCAD": ("CDARL", "DARL")
# }

# # Color palette for strict visual differentiation
# COLORS = {
#     "M1_SOTA": "#ff9999",  # Light Salmon
#     "M1_PRISM": "#66b3ff", # Light Blue
#     "M2_SOTA": "#cc0000",  # Dark Red
#     "M2_PRISM": "#0055ff"  # Dark Blue
# }

# # =================================================================================
# # 2. DATA EXTRACTION & MEAN CALCULATION
# # =================================================================================
# def get_mean_and_sig(json_path):
#     """
#     Parses the JSON to extract raw image-wise entropies, computes the 
#     arithmetic mean dynamically, and maps the statistical significance markers.
#     """
#     if not os.path.exists(json_path):
#         return None
        
#     with open(json_path, 'r') as f:
#         data = json.load(f)
        
#     # Extract significance markers mapping
#     sig_map = {}
#     for stat in data.get('statistical_summary', []):
#         noise = stat['Noise_Type'].upper()
#         sig_map[noise] = stat.get('significance_marker', 'ns')
        
#     # Extract raw entropies and compute mean dynamically
#     raw_entropies = data.get('raw_image_entropies', [])
#     if not raw_entropies:
#         return None
        
#     df_raw = pd.DataFrame(raw_entropies)
#     df_raw['Noise Type'] = df_raw['Noise Type'].str.upper()
    
#     # Calculate arithmetic mean grouped by Noise Type
#     mean_df = df_raw.groupby('Noise Type').agg(
#         SOTA_Mean=('SOTA Entropy', 'mean'),
#         PRISM_Mean=('PRISM Entropy', 'mean')
#     ).reset_index()
    
#     # Attach the significance marker
#     mean_df['sig'] = mean_df['Noise Type'].map(sig_map)
    
#     return mean_df.set_index('Noise Type')


# def load_paired_dataset_data(dataset_name, model_1, model_2):
#     """Loads and merges the computed mean data for two models on the same dataset."""
#     json_path_1 = os.path.join(DIRECTORIES[model_1], "Ablation_Statistical_Results.json")
#     json_path_2 = os.path.join(DIRECTORIES[model_2], "Ablation_Statistical_Results.json")
    
#     df1 = get_mean_and_sig(json_path_1)
#     df2 = get_mean_and_sig(json_path_2)
    
#     if df1 is None:
#         print(f"WARNING: Missing or empty data for {model_1} at {json_path_1}")
#         return None
#     if df2 is None:
#         print(f"WARNING: Missing or empty data for {model_2} at {json_path_2}")
#         return None
        
#     common_noises = df1.index.intersection(df2.index)
#     if len(common_noises) == 0:
#         print(f"WARNING: No common noise types found between {model_1} and {model_2}")
#         return None
    
#     merged_data = []
#     for noise in common_noises:
#         merged_data.append({
#             'Noise_Type': noise,
#             f'{model_1}_SOTA': df1.loc[noise, 'SOTA_Mean'],
#             f'{model_1}_PRISM': df1.loc[noise, 'PRISM_Mean'],
#             f'{model_2}_SOTA': df2.loc[noise, 'SOTA_Mean'],
#             f'{model_2}_PRISM': df2.loc[noise, 'PRISM_Mean'],
#             f'{model_1}_sig': df1.loc[noise, 'sig'],
#             f'{model_2}_sig': df2.loc[noise, 'sig']
#         })
        
#     return pd.DataFrame(merged_data)

# # =================================================================================
# # 3. PLOTTING FUNCTIONS
# # =================================================================================
# def plot_per_noise_mean(df, dataset_name, model_1, model_2, out_dir):
#     """Generates the 4-bar mean entropy plot for each individual noise type."""
#     for _, row in df.iterrows():
#         noise = row['Noise_Type']
#         plt.figure(figsize=(10, 6))
        
#         labels = [f'{model_1}\nSOTA', f'{model_1}\nPRISM', f'{model_2}\nSOTA', f'{model_2}\nPRISM']
#         means = [
#             row[f'{model_1}_SOTA'], row[f'{model_1}_PRISM'],
#             row[f'{model_2}_SOTA'], row[f'{model_2}_PRISM']
#         ]
#         colors = [COLORS["M1_SOTA"], COLORS["M1_PRISM"], COLORS["M2_SOTA"], COLORS["M2_PRISM"]]
        
#         bars = plt.bar(labels, means, color=colors, edgecolor='black', width=0.6)
        
#         # Attach Significance markers above PRISM bars
#         plt.text(1, means[1] + (max(means)*0.02), f"{row[f'{model_1}_sig']}", ha='center', va='bottom', fontweight='bold', color='red')
#         plt.text(3, means[3] + (max(means)*0.02), f"{row[f'{model_2}_sig']}", ha='center', va='bottom', fontweight='bold', color='red')
        
#         plt.ylabel(r'Mean Rank Entropy ($\bar{H}$) [bits]', fontsize=14, fontweight='bold')
#         plt.title(f'Cross-Model Mean Rank Stability: {dataset_name}\nNoise: {noise} (Lower is Better)', fontsize=16, fontweight='bold')
#         plt.grid(axis='y', linestyle=':', alpha=0.7)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"{dataset_name}_CrossModel_Mean_{noise}.pdf"), dpi=300)
#         plt.close()

# def plot_collective_mean(df, dataset_name, model_1, model_2, out_dir):
#     """Generates the comprehensive multi-noise mean entropy plot for the dataset."""
#     plt.figure(figsize=(16, 8))
    
#     x = np.arange(len(df))  
#     width = 0.2  
    
#     plt.bar(x - 1.5*width, df[f'{model_1}_SOTA'], width, label=f'{model_1} (SOTA)', color=COLORS["M1_SOTA"], edgecolor='black')
#     plt.bar(x - 0.5*width, df[f'{model_1}_PRISM'], width, label=f'{model_1} (PRISM)', color=COLORS["M1_PRISM"], edgecolor='black')
#     plt.bar(x + 0.5*width, df[f'{model_2}_SOTA'], width, label=f'{model_2} (SOTA)', color=COLORS["M2_SOTA"], edgecolor='black')
#     plt.bar(x + 1.5*width, df[f'{model_2}_PRISM'], width, label=f'{model_2} (PRISM)', color=COLORS["M2_PRISM"], edgecolor='black')
    
#     for idx, row in df.iterrows():
#         y_val_m1 = max(row[f'{model_1}_SOTA'], row[f'{model_1}_PRISM'])
#         plt.text(idx - 0.5*width, y_val_m1 + (y_val_m1*0.05), f"{row[f'{model_1}_sig']}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
#         y_val_m2 = max(row[f'{model_2}_SOTA'], row[f'{model_2}_PRISM'])
#         plt.text(idx + 1.5*width, y_val_m2 + (y_val_m2*0.05), f"{row[f'{model_2}_sig']}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

#     plt.xlabel('Perturbation Types', fontsize=14, fontweight='bold')
#     plt.ylabel(r'Mean Rank Entropy ($\bar{H}$) [bits]', fontsize=14, fontweight='bold')
#     plt.title(f'Global Cross-Model Mean Entropy Comparison ({dataset_name})\nLower Entropy = Superior Rank Stability', fontsize=16, fontweight='bold')
    
#     plt.xticks(x, df['Noise_Type'], fontsize=12, rotation=15)
#     plt.legend(fontsize=12, loc='upper right')
#     plt.grid(axis='y', linestyle=':', alpha=0.6)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"{dataset_name}_Mean_SUMMARY.pdf"), dpi=300)
#     plt.close()

# # =================================================================================
# # 4. EXECUTION LOOP
# # =================================================================================
# if __name__ == "__main__":
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     for dataset, (model_1, model_2) in DATASET_PAIRS.items():
#         print(f"\n--- Processing Dataset: {dataset} ({model_1} vs {model_2}) ---")
        
#         dataset_out_dir = os.path.join(OUTPUT_DIR, dataset)
#         os.makedirs(dataset_out_dir, exist_ok=True)
        
#         df_merged = load_paired_dataset_data(dataset, model_1, model_2)
        
#         if df_merged is None or df_merged.empty:
#             print(f"Skipping {dataset}: Could not aggregate valid statistical data.")
#             continue
            
#         print(f"Generating Individual Noise Plots for {dataset}...")
#         plot_per_noise_mean(df_merged, dataset, model_1, model_2, dataset_out_dir)
        
#         print(f"Generating Collective Summary Plot for {dataset}...")
#         plot_collective_mean(df_merged, dataset, model_1, model_2, dataset_out_dir)
        
#     print(f"\nSuccess! All visual proofs saved to {os.path.abspath(OUTPUT_DIR)}")




import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =================================================================================
# 1. CONFIGURATION & DIRECTORIES
# =================================================================================
OUTPUT_DIR = "./cross_model_mean_entropy_plots"

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

# The paired grouping that dictates adjacency and color families
DATASET_PAIRS = {
    "Pascal-VOC": ("SAMv1", "SAMv2"),
    "ISIC-2016": ("MSDUNet", "TransFuse"),
    "Synapse": ("D-LKA", "TransDeeplab"),
    "XCAD": ("CDARL", "DARL")
}

# Light, distinct color palettes where Model 1 and Model 2 of a dataset share a color family but different tones
COLORS = {
    "SAMv1": "#a8d8ea", "SAMv2": "#6796ca",               # Blues
    "MSDUNet": "#c5e3d6", "TransFuse": "#6cb994",         # Greens
    "D-LKA": "#d5c6e0", "TransDeeplab": "#9584c8",        # Purples
    "CDARL": "#deb2b5", "DARL": "#e0786b"                 # Pinks/Oranges
}

# =================================================================================
# 2. DATA EXTRACTION
# =================================================================================
def normalize_0_1(df, col="Entropy"):
    df = df.copy()
    vmin = df[col].min()
    vmax = df[col].max()
    if vmax == vmin:
        df[col] = 0.0
    else:
        df[col] = (df[col] - vmin) / (vmax - vmin)
    return df

def get_mean_entropy(json_path):
    """Parses JSON to extract raw entropies and computes the arithmetic mean."""
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    raw_entropies = data.get('raw_image_entropies', [])
    if not raw_entropies:
        return None
        
    df_raw = pd.DataFrame(raw_entropies)
    df_raw['Noise Type'] = df_raw['Noise Type'].str.lower()
    
    mean_df = df_raw.groupby('Noise Type').agg(
        SOTA_Mean=('SOTA Entropy', 'mean'),
        PRISM_Mean=('PRISM Entropy', 'mean')
    ).reset_index()
    
    return mean_df.set_index('Noise Type')

def compile_master_dataframe():
    """Aggregates all available datasets into a single unpivoted structure."""
    master_data = []
    
    for dataset, (m1, m2) in DATASET_PAIRS.items():
        path1 = os.path.join(DIRECTORIES.get(m1, ""), "Ablation_Statistical_Results.json")
        path2 = os.path.join(DIRECTORIES.get(m2, ""), "Ablation_Statistical_Results.json")
        
        df1 = get_mean_entropy(path1)
        df2 = get_mean_entropy(path2)
        
        if df1 is None or df2 is None:
            print(f"Skipping {dataset}: Missing data for {m1} or {m2}.")
            continue
            
        common_noises = df1.index.intersection(df2.index)
        for noise in common_noises:
            # SOTA Entries
            master_data.append({'Noise': noise, 'Dataset': dataset, 'Model': m1, 'Metric': 'SOTA', 'Entropy': df1.loc[noise, 'SOTA_Mean']})
            master_data.append({'Noise': noise, 'Dataset': dataset, 'Model': m2, 'Metric': 'SOTA', 'Entropy': df2.loc[noise, 'SOTA_Mean']})
            # PRISM Entries
            master_data.append({'Noise': noise, 'Dataset': dataset, 'Model': m1, 'Metric': 'PRISM', 'Entropy': df1.loc[noise, 'PRISM_Mean']})
            master_data.append({'Noise': noise, 'Dataset': dataset, 'Model': m2, 'Metric': 'PRISM', 'Entropy': df2.loc[noise, 'PRISM_Mean']})
            
    return pd.DataFrame(master_data)

# =================================================================================
# 3. COLLECTIVE MASTER PLOT (SIDE-BY-SIDE)
# =================================================================================
def plot_collective_master_graph(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    noises = df['Noise'].unique()
    models = df['Model'].unique()
    num_models = len(models)

    df = normalize_0_1(df, "Entropy")
    
    # Layout mathematics for strictly side-by-side (SOTA, PRISM, SOTA, PRISM...)
    bar_width = 0.70
    total_bars_per_noise = num_models * 2
    offsets = np.arange(total_bars_per_noise) * bar_width - (total_bars_per_noise * bar_width) / 2 + bar_width / 2
    
    fig, ax = plt.subplots(figsize=(44, 12))
    x_positions = np.arange(len(noises)) * 13.0  # Slightly wider spacing to fit all 16 bars comfortably
    
    for x_idx, noise in enumerate(noises):
        noise_data = df[df['Noise'] == noise]
        x_base = x_positions[x_idx]
        
        for i, model in enumerate(models):
            # 1. SOTA BAR (Bold/Solid)
            sota_val = noise_data[(noise_data['Model'] == model) & (noise_data['Metric'] == 'SOTA')]['Entropy'].values
            if len(sota_val) > 0:
                x_sota = x_base + offsets[2 * i]
                ax.bar(x_sota, sota_val[0], width=bar_width, color=COLORS[model], 
                       edgecolor='black', linewidth=2.0, linestyle='solid')
                ax.text(x_sota, sota_val[0] + 0.02, f"{sota_val[0]:.2f}", rotation=90, va='bottom', ha='center', fontsize=20, color='black')

            # 2. PRISM BAR (Dashed/Hatched)
            prism_val = noise_data[(noise_data['Model'] == model) & (noise_data['Metric'] == 'PRISM')]['Entropy'].values
            if len(prism_val) > 0:
                x_prism = x_base + offsets[2 * i + 1]
                ax.bar(x_prism, prism_val[0], width=bar_width, color=COLORS[model], 
                       edgecolor='black', linewidth=2.0, linestyle='dashed', hatch='//')
                ax.text(x_prism, prism_val[0] + 0.02, f"{prism_val[0]:.2f}", rotation=90, va='bottom', ha='center', fontsize=20, color='black')

    left_edge = x_positions[0] + offsets[0] - (bar_width / 2)
    right_edge = x_positions[-1] + offsets[-1] + (bar_width / 2)

    padding = bar_width * 2
    ax.set_xlim(left_edge - padding, right_edge + padding)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(noises, fontsize=30, fontweight='bold')
    ax.tick_params(axis='x', pad=25) 
    
    ax.set_ylabel(r'Entropy ($\bar{H}$)', fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=30)
    ax.set_ylim(0, df['Entropy'].max() * 1.15)
    
    plt.title('SOTA vs. PRISM Mean Entropy Comparison', fontsize=32, fontweight='bold', pad=15)
    

    legend_handles = []
     # 1. Model Color Legends
    for dataset, (m1, m2) in DATASET_PAIRS.items():
        if m1 in models:
            legend_handles.append(mpatches.Patch(color=COLORS[m1], label=f'{m1}'))
            legend_handles.append(mpatches.Patch(color=COLORS[m2], label=f'{m2}'))
             
     # 2. Style Guide for SOTA vs PRISM
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0, linestyle='solid', label='SOTA'))
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0, linestyle='dashed', hatch='//', label='PRISM'))
     # 1) First row: model name color patches (one patch per model present)
    model_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in models if m in COLORS]
 
     # 2) Second row: style guide for SOTA vs PRISM
    style_handles = [
         mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0, label='SOTA (Solid)'),
         mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0, hatch='//', label='PRISM (Dashed)')
     ]
 
     # Place first (models) legend as a full-width row above the axes
    first_legend = ax.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                              ncol=max(1, len(model_handles)), fontsize=24, frameon=False)
    ax.add_artist(first_legend)

     # Place second (style) legend as a second row just below the first
    ax.legend(handles=style_handles, loc='upper center', bbox_to_anchor=(0.5, 0.97),
              ncol=2, fontsize=24, frameon=False)
            
    # ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.88), 
    #           title=" ", fontsize=28, title_fontsize=24, frameon=False, 
    #           edgecolor='black', ncol=len(models) // 2 + 1)

            
    # ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.88), 
    #           title=" ", fontsize=28, title_fontsize=24, frameon=False, 
    #           edgecolor='black', ncol=len(models) // 2 + 1)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Master_Collective_Mean_Entropy_Graph.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Master collective graph compiled and saved to -> {save_path}")

# =================================================================================
# 4. EXECUTION
# =================================================================================
if __name__ == "__main__":
    print("Compiling Master Entropy Data...")
    df_master = compile_master_dataframe()
    
    if df_master.empty:
        print("FATAL: No valid data found. Check your directories.")
    else:
        print(f"Loaded {len(df_master)} data points. Generating Collective Graph...")
        plot_collective_master_graph(df_master)
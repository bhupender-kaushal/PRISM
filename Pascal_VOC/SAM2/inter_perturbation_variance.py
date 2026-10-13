import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# 1. PARSERS (Reusing your exact logic to pull the data)
# -------------------------------------------------------------------------
def parse_continuous_framework(dir_dict, dataset_map):
    records = []
    for model_name, path_str in dir_dict.items():
        base_path = Path(path_str)
        run_1_path = base_path / "run_1"
        if not run_1_path.exists(): continue
        
        for p in run_1_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            image_id = p.stem.replace('_analysis', '')
            for key, content in data.items():
                key_lower = key.lower()
                if 'in_context' in key_lower or 'ic' in key_lower: context_type = 'Continuous IC'
                elif 'out_of_context' in key_lower or 'oc' in key_lower: context_type = 'Continuous OC'
                else: continue 

                for noise_type, metrics in content.get('level_data_per_noise', {}).items():
                    steps = metrics.get('level_data', [])
                    for step in steps:
                        records.append({
                            'model': model_name,
                            'image_id': image_id,
                            'paradigm': context_type,
                            'perturbation_type': noise_type.capitalize(),
                            'global_iou': step.get('global_iou', 100.0)
                        })
    return pd.DataFrame(records)

def parse_id_framework(dir_dict, dataset_map):
    records = []
    for model_name, path_str in dir_dict.items():
        base_path = Path(path_str)
        if not base_path.exists(): continue
        
        for p in base_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            if 'eigen_guided_analysis' not in data: continue
            image_id = p.stem.replace('_analysis', '')
            trajectories = data['eigen_guided_analysis'].get('trajectories', {})

            for baseline_name, ranking_modes in trajectories.items():
                for rank_type, modes in ranking_modes.items():
                    if rank_type != 'eigen_ranked': continue # Focus on EigenCAM for standard I/D
                    
                    for mode_key in ['deletion']: # Focus on Deletion for standard I/D
                        steps = modes.get(mode_key, [])
                        for step in steps:
                            records.append({
                                'model': model_name,
                                'image_id': image_id,
                                'paradigm': 'I/D Deletion (EigenCAM)',
                                'perturbation_type': baseline_name.capitalize(),
                                'global_iou': step.get('global_iou', 0.0)
                            })
    return pd.DataFrame(records)

# -------------------------------------------------------------------------
# 2. VARIANCE CALCULATION ENGINE
# -------------------------------------------------------------------------
def calculate_metric_consistency(df_id, df_cont):
    # Combine both dataframes
    df_all = pd.concat([df_id, df_cont], ignore_index=True)
    
    # Step 1: Calculate the Trajectory AUC (Mean IoU) for every single Image + Perturbation combo
    # This gives us a single robustness score (0-100) for how well the image survived that specific noise
    trajectory_means = df_all.groupby(['model', 'paradigm', 'perturbation_type', 'image_id'])['global_iou'].mean().reset_index()
    
    # Step 2: Average those scores across all images to get the Model's performance on that specific noise
    model_noise_means = trajectory_means.groupby(['model', 'paradigm', 'perturbation_type'])['global_iou'].mean().reset_index()
    
    # Step 3: Calculate the Coefficient of Variation (CV) across the different perturbation types
    # I/D variance is calculated across its 8 canvases. Continuous variance is calculated across its noise types.
    consistency_df = model_noise_means.groupby(['model', 'paradigm'])['global_iou'].agg(
        mean_iou='mean',
        std_iou='std'
    ).reset_index()
    
    # CV = (Standard Deviation / Mean) * 100
    consistency_df['cv_percent'] = (consistency_df['std_iou'] / consistency_df['mean_iou']) * 100
    return consistency_df

# -------------------------------------------------------------------------
# 3. PLOTTING ENGINE
# -------------------------------------------------------------------------
def plot_consistency_bar_chart(cv_df, output_dir):
    plt.figure(figsize=(14, 7))
    
    # We want a grouped bar chart: X-axis is Model, Y-axis is CV, Hue is the Paradigm
    sns.barplot(
        data=cv_df, 
        x='model', 
        y='cv_percent', 
        hue='paradigm', 
        palette={'I/D Deletion (EigenCAM)': '#d62728', 'Continuous IC': '#1f77b4', 'Continuous OC': '#2ca02c'},
        edgecolor='black',
        linewidth=1.5
    )
    
    plt.title("Evaluation Metric Consistency:\nCoefficient of Variation (CV) Across Perturbation Types", fontsize=16, fontweight='bold')
    plt.xlabel("Architecture", fontsize=13)
    plt.ylabel("Coefficient of Variation (CV %) - Lower is Better", fontsize=13)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Evaluation Paradigm", fontsize=11, title_fontsize=12)
    
    # Add text annotation explaining the math
    plt.text(
        x=plt.gca().get_xlim()[1] * 0.98, y=plt.gca().get_ylim()[1] * 0.95,
        s="High CV = Fragile to canvas changes (Inconsistent)\nLow CV = Reliable textural evaluation (Consistent)",
        fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "metric_consistency_cv.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    # Export the hard numbers to CSV for the paper table
    cv_df.round(2).to_csv(os.path.join(output_dir, "metric_consistency_cv.csv"), index=False)
    print(f"[SAVED] Consistency Graph: {out_file}")

# -------------------------------------------------------------------------
# 4. CONTROLLER
# -------------------------------------------------------------------------
if __name__ == "__main__":
    DATASET_MAPPING = {
        "SAMv1": "Pascal-VOC", "SAMv2": "Pascal-VOC",
        "MSDUNet": "ISIC-2016", "TransFuse": "ISIC-2016",
        "D-LKA": "Synapse", "TransDeeplab": "Synapse",
        "CDARL": "XCAD", "DARL": "XCAD"
    }

    # Use your exact directory dictionaries here
    CONTINUOUS_DIRS = {
        "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out_original", 
        "SAMv2": "/bhupender/2025/Mouse_other/prism_out_original", 
        "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out_original", 
        "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_original", 
        "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out_original", 
        "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out_original", 
        "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out_original", 
        "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out_original", 
    }
    
    ID_DIRS = {
        "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out_insertion_deletion", 
        "SAMv2": "/bhupender/2025/Mouse_other/prism_out_insertion_deletion", 
        "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out_insertion_deletion", 
        "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_insertion_deletion", 
        "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out_insertion_deletion", 
        "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out_insertion_deletion", 
        "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out_insertion_deletion", 
        "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out_insertion_deletion", 
    }
    
    OUT_DIR = "./comparative_pdf_reports"
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(">>> Parsing Data for Variance Calculation...")
    df_cont = parse_continuous_framework(CONTINUOUS_DIRS, DATASET_MAPPING)
    df_id = parse_id_framework(ID_DIRS, DATASET_MAPPING)

    if not df_cont.empty and not df_id.empty:
        print(">>> Calculating Coefficient of Variation (CV)...")
        cv_df = calculate_metric_consistency(df_id, df_cont)
        plot_consistency_bar_chart(cv_df, OUT_DIR)
    else:
        print("[ERROR] Dataframes are empty. Check paths.")
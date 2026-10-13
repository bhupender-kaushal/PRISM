import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import trapz

# -------------------------------------------------------------------------
# 1. RANK BIASED OVERLAP (RBO) CALCULATOR
# -------------------------------------------------------------------------
def calculate_rbo(list1, list2, p=0.9):
    if not list1 or not list2: return 0.0
    set1, set2 = set(), set()
    overlap_sum = 0.0
    depth = min(len(list1), len(list2))
    for i in range(depth):
        set1.add(list1[i])
        set2.add(list2[i])
        agreement = len(set1.intersection(set2)) / (i + 1)
        overlap_sum += (p ** i) * agreement
    return float((1 - p) * overlap_sum)

def extract_ranked_list(superpixels_data):
    sorted_sps = sorted(superpixels_data, key=lambda x: x.get('step_prob_rank', 9999))
    return [sp['sp_id'] for sp in sorted_sps]

# -------------------------------------------------------------------------
# 2. GENERALIZED PARSERS 
# -------------------------------------------------------------------------
def parse_continuous_framework(dir_dict, dataset_map, p_val=0.9):
    records = []
    for model_name, path_str in dir_dict.items():
        base_path = Path(path_str)
        run_1_path = base_path / "run_1"
        if not run_1_path.exists():
            print(f"[WARN] Continuous 'run_1' Path missing: {run_1_path}")
            continue
        dataset_name = dataset_map.get(model_name, "Unknown")
        
        for p in run_1_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            image_id = p.stem.replace('_analysis', '')
            for key, content in data.items():
                key_lower = key.lower()
                if 'in_context' in key_lower or 'ic' in key_lower: context_type = 'Continuous IC'
                elif 'out_of_context' in key_lower or 'oc' in key_lower: context_type = 'Continuous OC'
                elif 'global' in key_lower: context_type = 'Continuous Global'
                else: continue 

                for noise_type, metrics in content.get('level_data_per_noise', {}).items():
                    steps = metrics.get('level_data', [])
                    if not steps: continue
                    baseline_sps = steps[0].get('superpixels', [])
                    baseline_ranks = extract_ranked_list(baseline_sps)
                    
                    for step in steps:
                        ssim_val = step.get('ssim', 1.0)
                        norm_severity = max(0.0, min(1.0, (1.0 - ssim_val) / 0.30))
                        current_ranks = extract_ranked_list(step.get('superpixels', []))
                        
                        records.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'image_id': image_id,
                            'paradigm': context_type,
                            'perturbation': noise_type.capitalize(),
                            'severity': norm_severity,
                            'global_iou': step.get('global_iou', 100.0),
                            'rbo': calculate_rbo(baseline_ranks, current_ranks, p=p_val)
                        })
    return pd.DataFrame(records)

def parse_id_framework(dir_dict, dataset_map, p_val=0.9):
    records = []
    for model_name, path_str in dir_dict.items():
        base_path = Path(path_str)
        if not base_path.exists():
            print(f"[WARN] I/D Path missing: {path_str}")
            continue
        dataset_name = dataset_map.get(model_name, "Unknown")
        
        for p in base_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            if 'eigen_guided_analysis' not in data: continue
            image_id = p.stem.replace('_analysis', '')
            trajectories = data['eigen_guided_analysis'].get('trajectories', {})

            for baseline_name, ranking_modes in trajectories.items():
                for rank_type, modes in ranking_modes.items():
                    if rank_type not in ['eigen_ranked', 'prob_ranked']: continue
                    rank_name = 'EigenCAM' if 'eigen' in rank_type else 'Probability'
                    
                    for mode_key in ['deletion', 'insertion']:
                        steps = modes.get(mode_key, [])
                        if not steps: continue
                        baseline_ranks = extract_ranked_list(steps[0].get('superpixels', []))
                        
                        paradigm_name = f"I/D {mode_key.capitalize()} ({rank_name})"
                        
                        for step in steps:
                            frac = step.get('fraction_perturbed', 0.0)
                            severity = frac if mode_key == 'deletion' else (1.0 - frac)
                            current_ranks = extract_ranked_list(step.get('superpixels', []))
                            
                            records.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'image_id': image_id,
                                'paradigm': paradigm_name,
                                'perturbation': baseline_name.capitalize(),
                                'severity': severity,
                                'global_iou': step.get('global_iou', 0.0),
                                'rbo': calculate_rbo(baseline_ranks, current_ranks, p=p_val)
                            })
    return pd.DataFrame(records)

# -------------------------------------------------------------------------
# 3. AUC COMPUTATION ENGINE
# -------------------------------------------------------------------------
def compute_trajectory_aucs(df_all):
    """Calculates the Area Under the Curve for IoU and RBO (Rank Flip) trajectories."""
    print(">>> Calculating Rank Flip AUC (RBO) and Performance AUC (IoU)...")
    auc_records = []
    
    groups = df_all.groupby(['model', 'paradigm', 'perturbation', 'image_id'])
    
    for (model, paradigm, perturbation, img_id), group in groups:
        group = group.sort_values('severity')
        if len(group) < 2: continue
        
        x = group['severity'].values
        
        y_iou = group['global_iou'].values / 100.0
        auc_iou = trapz(y=y_iou, x=x)
        
        y_rbo = group['rbo'].values
        auc_rbo = trapz(y=y_rbo, x=x)
        
        auc_records.append({
            'model': model,
            'image_id': img_id,
            'paradigm': paradigm,
            'perturbation': perturbation,
            'IoU_AUC': auc_iou,
            'Rank_Flip_AUC': auc_rbo
        })
        
    return pd.DataFrame(auc_records)

# -------------------------------------------------------------------------
# 4. LINE PLOTTING ENGINES (Replaced Violins)
# -------------------------------------------------------------------------
def plot_degradation_trajectories(pdf_obj, df, y_col, title, y_label):
    """Plots the actual degradation curve (Severity vs Metric) with shaded variance."""
    if df.empty: return
    
    # Creates a multi-panel grid, one panel for each perturbation type
    g = sns.relplot(
        data=df, x='severity', y=y_col, hue='model', col='perturbation', 
        col_wrap=3, kind='line', marker='o', errorbar=('ci', 95), 
        height=4, aspect=1.3, linewidth=2
    )
    
    g.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    g.set_axis_labels("Severity Fraction (0.0 to 1.0)", y_label)
    
    if 'iou' in y_col.lower():
        g.set(ylim=(-5, 105))
    else:
        g.set(ylim=(-0.05, 1.05))
        
    pdf_obj.savefig(g.fig, bbox_inches='tight')
    plt.close(g.fig)

def plot_scalar_auc_lines(pdf_obj, df_auc, y_col, title, y_label):
    """Plots the final integrated AUC scores across different perturbation types."""
    if df_auc.empty: return
    fig = plt.figure(figsize=(16, 7))
    
    sns.lineplot(
        data=df_auc, x='perturbation', y=y_col, hue='model', 
        marker='s', markersize=8, linewidth=2.5, errorbar=('ci', 95)
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Perturbation Type", fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(title="Architecture", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    pdf_obj.savefig(fig)
    plt.close(fig)

# -------------------------------------------------------------------------
# 5. MASTER CONTROLLER
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    DATASET_MAPPING = {
        "SAMv1": "Pascal-VOC", "SAMv2": "Pascal-VOC",
        "MSDUNet": "ISIC-2016", "TransFuse": "ISIC-2016",
        "D-LKA": "Synapse", "TransDeeplab": "Synapse",
        "CDARL": "XCAD", "DARL": "XCAD"
    }

    # Your existing directories
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

    print(">>> Parsing Data...")
    df_cont = parse_continuous_framework(CONTINUOUS_DIRS, DATASET_MAPPING, p_val=0.9)
    df_id = parse_id_framework(ID_DIRS, DATASET_MAPPING, p_val=0.9)

    df_all = pd.concat([df_cont, df_id], ignore_index=True)

    if not df_all.empty:
        # --- 1. Master Degradation Trajectory PDFs (The Curves) ---
        iou_curve_path = os.path.join(OUT_DIR, "Master_IoU_Degradation_Curves.pdf")
        rbo_curve_path = os.path.join(OUT_DIR, "Master_RBO_Decay_Curves.pdf")
        
        with PdfPages(iou_curve_path) as pdf_iou, PdfPages(rbo_curve_path) as pdf_rbo:
            for paradigm in df_all['paradigm'].unique():
                p_df = df_all[df_all['paradigm'] == paradigm]
                plot_degradation_trajectories(
                    pdf_iou, p_df, y_col='global_iou',
                    title=f"IoU Degradation Trajectory: {paradigm}", y_label="Global IoU (%)"
                )
                plot_degradation_trajectories(
                    pdf_rbo, p_df, y_col='rbo',
                    title=f"Rank Hierarchy Decay (RBO): {paradigm}", y_label="Rank Biased Overlap (RBO)"
                )
        
        # --- 2. AUC Computation & Scalar Line Plots ---
        df_auc = compute_trajectory_aucs(df_all)
        df_auc.to_csv(os.path.join(OUT_DIR, "AUC_Analysis_Summary.csv"), index=False)
        
        auc_pdf_path = os.path.join(OUT_DIR, "Master_AUC_Analysis.pdf")
        with PdfPages(auc_pdf_path) as pdf_auc:
            for paradigm in df_auc['paradigm'].unique():
                p_auc = df_auc[df_auc['paradigm'] == paradigm]
                
                # Rank Flip AUC Line Plot
                plot_scalar_auc_lines(
                    pdf_auc, p_auc, y_col='Rank_Flip_AUC',
                    title=f"Rank Flip AUC Across Perturbations: {paradigm}", y_label="Rank Flip AUC (0.0 to 1.0)"
                )
                
                # Performance AUC Line Plot
                plot_scalar_auc_lines(
                    pdf_auc, p_auc, y_col='IoU_AUC',
                    title=f"Performance AUC Across Perturbations: {paradigm}", y_label="Performance AUC (0.0 to 1.0)"
                )

    print(f"\n[DONE] Master Curve PDFs and AUC Summaries successfully compiled in: {OUT_DIR}")
import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
SINGLE_NOISE_MODE = 'Gaussian_blur' 
CUMULATIVE_FRACTIONS = [0.3, 0.5]

# NEW: Separate flag to control which baselines Insertion/Deletion should evaluate
ID_BASELINES = ['mean', 'gaussian_blur'] 

# ==========================================
# 2. MATH & EXTRACTION ENGINES
# ==========================================
GAMMA = 1.0
LAMBDA_P = 1.0

def calculate_holistic_s_factor(ic_levels, oc_levels):
    IC_vals = [float(lvl.get("discounted_utility", lvl.get("global_iou", 0.0))) for lvl in ic_levels]
    OC_vals = [float(lvl.get("discounted_utility", lvl.get("global_iou", 0.0))) for lvl in oc_levels]

    N_min = min(len(IC_vals), len(OC_vals))
    if N_min < 2: return np.nan
    
    IC_0, OC_0 = max(1e-6, IC_vals[0]), max(1e-6, OC_vals[0])
    W_ic, A_ic = 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1)
        w_k = math.exp(-GAMMA * eta_k)
        A_ic += w_k * (IC_vals[k] / IC_0)
        W_ic += w_k
    norm_A_ic = min(1.0, A_ic / W_ic if W_ic > 0 else 0.0)

    W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1)
        w_k = math.exp(-GAMMA * eta_k)
        oc_hat = OC_vals[k] / OC_0
        A_oc += w_k * oc_hat
        weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
        W_oc += w_k
    norm_A_oc = min(1.0, A_oc / W_oc if W_oc > 0 else 0.0)
    
    G_gap = max(0.0, norm_A_oc - norm_A_ic)
    E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0
    
    # FIXED: Corrupted math syntax
    return math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift)) * 100.0

def extract_noises(analysis_dict):
    if not isinstance(analysis_dict, dict): return {}
    return analysis_dict.get('level_data_per_noise', analysis_dict)

def clean_noise_name(raw_name):
    lower = str(raw_name).strip().lower()
    no_space = lower.replace(" ", "")
    if ('salt' in lower and 'pepper' in lower) or ('s&p' in no_space): return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def is_noise_match(target, current):
    if not target or not current: return False
    t = str(target).strip().lower().replace(' ', '').replace('_', '')
    c = str(current).strip().lower().replace(' ', '').replace('_', '')
    return (t in c) or (c in t)

def find_id_trajectories(node):
    if isinstance(node, dict):
        if 'trajectories' in node and isinstance(node['trajectories'], dict):
            return node['trajectories']
        for v in node.values():
            res = find_id_trajectories(v)
            if res: return res
    return {}

# ==========================================
# 3. DATA PARSER
# ==========================================
def extract_all_scores(cont_dirs, id_dirs, pairings):
    records = []
    
    for dataset, (model_A, model_B) in pairings.items():
        for model_name in [model_A, model_B]:
            
            # --- CONTINUOUS FRAMEWORKS ---
            if model_name in cont_dirs:
                run_1_path = Path(cont_dirs[model_name]) / "run_1"
                if run_1_path.exists():
                    for p in run_1_path.rglob('*_analysis.json'):
                        image_id = p.stem.replace('_analysis', '')
                        try:
                            with open(p, 'r') as f: data = json.load(f)
                            
                            # IC/OC S-Factor
                            ic_noises = extract_noises(data.get('in_context_analysis', {}))
                            oc_noises = extract_noises(data.get('out_of_context_analysis', {}))
                            for raw_noise in set(ic_noises.keys()).intersection(oc_noises.keys()):
                                clean_noise = clean_noise_name(raw_noise)
                                ic_lvls = ic_noises[raw_noise].get('level_data', []) if isinstance(ic_noises[raw_noise], dict) else ic_noises[raw_noise]
                                oc_lvls = oc_noises[raw_noise].get('level_data', []) if isinstance(oc_noises[raw_noise], dict) else oc_noises[raw_noise]
                                
                                if SINGLE_NOISE_MODE:
                                    if not is_noise_match(SINGLE_NOISE_MODE, clean_noise): continue
                                    max_len = min(len(ic_lvls), len(oc_lvls))
                                    for i, frac in enumerate(CUMULATIVE_FRACTIONS):
                                        idx = max(1, int(frac * (max_len - 1)))
                                        score = calculate_holistic_s_factor(ic_lvls[:idx+1], oc_lvls[:idx+1])
                                        if not np.isnan(score):
                                            records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'S-Factor (IC/OC)', 'noise': f'Cumul_{int(frac*100)}%_v{i+1}', 'score': score})
                                else:
                                    score = calculate_holistic_s_factor(ic_lvls, oc_lvls)
                                    if not np.isnan(score):
                                        records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'S-Factor (IC/OC)', 'noise': clean_noise, 'score': score})
                            
                            # Global Continuous AUC
                            global_key = next((k for k in data.keys() if 'global' in k.lower() and 'analysis' in k.lower()), None)
                            if global_key:
                                gl_noises = extract_noises(data[global_key])
                                for raw_noise, metrics in gl_noises.items():
                                    clean_noise = clean_noise_name(raw_noise)
                                    lvls = metrics.get('level_data', []) if isinstance(metrics, dict) else metrics
                                    
                                    if SINGLE_NOISE_MODE:
                                        if not is_noise_match(SINGLE_NOISE_MODE, clean_noise): continue
                                        for i, frac in enumerate(CUMULATIVE_FRACTIONS):
                                            idx = max(1, int(frac * (len(lvls) - 1)))
                                            sub_lvls = lvls[:idx+1]
                                            x_axis = [k / (len(lvls) - 1) for k in range(len(sub_lvls))]
                                            y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in sub_lvls]
                                            records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Global Continuous AUC', 'noise': f'Cumul_{int(frac*100)}%_v{i+1}', 'score': trapz(y=y_axis, x=x_axis)})
                                    else:
                                        if len(lvls) > 1:
                                            x_axis = [k / (len(lvls) - 1) for k in range(len(lvls))]
                                            y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in lvls]
                                            records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Global Continuous AUC', 'noise': clean_noise, 'score': trapz(y=y_axis, x=x_axis)})
                        except Exception: pass

            # --- STANDARD I/D FRAMEWORK ---
            if model_name in id_dirs:
                id_path = Path(id_dirs[model_name])
                if id_path.exists():
                    for p in id_path.rglob('*_analysis.json'):
                        image_id = p.stem.replace('_analysis', '')
                        try:
                            with open(p, 'r') as f: data = json.load(f)
                            trajectories = find_id_trajectories(data)
                            
                            for baseline, ranks in trajectories.items():
                                # Evaluate strictly on requested I/D Baselines
                                if baseline.lower() not in [b.lower() for b in ID_BASELINES]: continue
                                if 'prob_ranked' not in ranks: continue
                                traj = ranks['prob_ranked']
                                
                                # Deletion Processing
                                del_lvls = traj.get('deletion', [])
                                if isinstance(del_lvls, dict): del_lvls = del_lvls.get('level_data', [])
                                if len(del_lvls) > 1:
                                    if SINGLE_NOISE_MODE:
                                        for i, frac in enumerate(CUMULATIVE_FRACTIONS):
                                            idx = max(1, int(frac * (len(del_lvls) - 1)))
                                            sub_lvls = del_lvls[:idx+1]
                                            x_axis = [float(step.get('fraction_perturbed', 0.0)) for step in sub_lvls]
                                            y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in sub_lvls]
                                            records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Standard I/D AUC (Deletion)', 'noise': f'{baseline.capitalize()}_{int(frac*100)}%_v{i+1}', 'score': trapz(y=y_axis, x=x_axis)})
                                    else:
                                        x_axis = [float(step.get('fraction_perturbed', 0.0)) for step in del_lvls]
                                        y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in del_lvls]
                                        records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Standard I/D AUC (Deletion)', 'noise': baseline.capitalize(), 'score': trapz(y=y_axis, x=x_axis)})
                                
                                # Insertion Processing
                                ins_lvls = traj.get('insertion', [])
                                if isinstance(ins_lvls, dict): ins_lvls = ins_lvls.get('level_data', [])
                                if len(ins_lvls) > 1:
                                    if SINGLE_NOISE_MODE:
                                        for i, frac in enumerate(CUMULATIVE_FRACTIONS):
                                            idx = max(1, int(frac * (len(ins_lvls) - 1)))
                                            sub_lvls = ins_lvls[:idx+1]
                                            x_axis = [float(step.get('fraction_perturbed', 0.0)) for step in sub_lvls]
                                            y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in sub_lvls]
                                            records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Standard I/D AUC (Insertion)', 'noise': f'{baseline.capitalize()}_{int(frac*100)}%_v{i+1}', 'score': trapz(y=y_axis, x=x_axis)})
                                    else:
                                        x_axis = [float(step.get('fraction_perturbed', 0.0)) for step in ins_lvls]
                                        y_axis = [float(step.get('global_iou', 0.0)) / 100.0 for step in ins_lvls]
                                        records.append({'model': model_name, 'image_id': image_id, 'paradigm': 'Standard I/D AUC (Insertion)', 'noise': baseline.capitalize(), 'score': trapz(y=y_axis, x=x_axis)})
                        except Exception: pass
                        
    return pd.DataFrame(records)

# ==========================================
# 4. KENDALL'S W & SPEARMAN'S RHO ENGINES
# ==========================================
def compute_kendalls_w(rank_matrix):
    n, m = rank_matrix.shape
    if n < 2 or m < 2: return np.nan
    R_i = np.sum(rank_matrix, axis=1)
    R_mean = np.mean(R_i)
    S = np.sum((R_i - R_mean)**2)
    
    # FIXED: Corrupted math syntax
    denom = (m**2 * (n**3 - n))
    return (12 * S) / denom if denom > 0 else 0.0

def compute_mean_spearman(df_pivot):
    if df_pivot.shape[1] < 2: return np.nan
    corr_matrix = df_pivot.corr(method='spearman').values
    idx = np.triu_indices_from(corr_matrix, k=1)
    return np.nanmean(corr_matrix[idx])

def run_image_ranking_consistency(df_all):
    results = []
    
    print("\n>>> DIAGNOSTIC: Image Counts Surviving Pivot Grid DropNA")
    for model in df_all['model'].unique():
        m_df = df_all[df_all['model'] == model]
        
        for paradigm in m_df['paradigm'].unique():
            p_df = m_df[m_df['paradigm'] == paradigm]
            
            try:
                pivot = p_df.pivot_table(index='image_id', columns='noise', values='score')
                initial_images = len(pivot)
                pivot_clean = pivot.dropna(axis=0, how='any')
                surviving_images = len(pivot_clean)
                
                print(f"[{model}] {paradigm}: {initial_images} initial -> {surviving_images} overlapping images")
                
                if surviving_images < 2 or pivot_clean.shape[1] < 2: 
                    print(f"   -> SKIPPING {model} {paradigm} (Not enough overlap)")
                    continue
                

                sampled_df = pivot_clean.sort_index()
                
                # Calculate Ranks
                rank_matrix = np.zeros_like(sampled_df.values)
                for col_idx in range(sampled_df.shape[1]):
                    rank_matrix[:, col_idx] = rankdata(-sampled_df.iloc[:, col_idx].values)
                
                # Calculate Stats
                w_score = compute_kendalls_w(rank_matrix)
                spearman_score = compute_mean_spearman(sampled_df)
                
                # Print the deterministic raw rank lists
                raw_ranks_df = pd.DataFrame(
                    rank_matrix.astype(int), 
                    columns=sampled_df.columns
                )
                print(f"\n=== RAW RANK LISTS FOR {model} | {paradigm} ===")
                print(raw_ranks_df.to_string(index=False))
                
                results.append({
                    'Model': model,
                    'Paradigm': paradigm,
                    'Kendalls_W': w_score,
                    'Spearmans_Rho': spearman_score,
                    'Num_Images': surviving_images,
                    'Num_Noises_Evaluated': pivot_clean.shape[1]
                })
            except Exception as e: pass
                
    return pd.DataFrame(results).round(3)

# ==========================================
# 5. MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    PAIRINGS = {
        "Pascal-VOC": ("SAMv1", "SAMv2"),
        "ISIC-2016": ("MSDUNet", "TransFuse"),
        "Synapse": ("D-LKA", "TransDeeplab"),
        "XCAD": ("CDARL", "DARL")
    }
    
    CONT_DIRS = {
        "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out_original", 
        # "SAMv2": "/bhupender/2025/Mouse_other/prism_out_original", 
        # "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out_original", 
        # "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_original", 
        # "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out_original", 
        # "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out_original", 
        # "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out_original", 
        # "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out_original", 
    }
    
    ID_DIRS = {
        "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/prism_out_insertion_deletion", 
        # "SAMv2": "/bhupender/2025/Mouse_other/prism_out_insertion_deletion", 
        # "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/prism_out_insertion_deletion", 
        # "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_insertion_deletion", 
        # "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out_insertion_deletion", 
        # "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out_insertion_deletion", 
        # "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/prism_out_insertion_deletion", 
        # "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/prism_out_insertion_deletion", 
    }
    
    OUT_DIR = "./meta_evaluation_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    mode_str = f"Intra-Noise Cumulative Stability (Noise: {SINGLE_NOISE_MODE})" if SINGLE_NOISE_MODE else "Cross-Noise Universal Stability"
    print(f">>> Extracting Data | Mode: {mode_str}")
    
    df_all = extract_all_scores(CONT_DIRS, ID_DIRS, PAIRINGS)
    
    if not df_all.empty:
        print("\n>>> Calculating Deterministic Kendall's W and Spearman's Rho (Full Dataset)...")
        w_results = run_image_ranking_consistency(df_all)
        
        if not w_results.empty:
            prefix = "Cumulative_" if SINGLE_NOISE_MODE else "CrossNoise_"
            w_results.to_csv(os.path.join(OUT_DIR, f"{prefix}Table3_Image_Ranking_Consistency.csv"), index=False)
            
            print(f"\n=== INTERNAL IMAGE RANKING CONSISTENCY ({mode_str}) ===")
            print(w_results.to_string(index=False))
            
            paradigm_palette = {
                'Standard I/D AUC (Deletion)': '#d62728', 
                'Standard I/D AUC (Insertion)': '#ff7f0e',
                'Global Continuous AUC': '#2ca02c', 
                'S-Factor (IC/OC)': '#1f77b4'
            }
            
            print("\n>>> Generating Standalone Model Plots (Side-by-Side Subplots)...")
            for model in w_results['Model'].unique():
                model_df = w_results[w_results['Model'] == model].copy()
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                title_suffix = f" (Cumulative {SINGLE_NOISE_MODE})" if SINGLE_NOISE_MODE else " (Cross-Noise)"
                fig.suptitle(f"Image Ranking Consistency: {model}{title_suffix}", fontsize=18, fontweight='bold', y=1.02)
                
                ax1 = sns.barplot(
                    data=model_df, x='Paradigm', y='Kendalls_W', hue='Paradigm', 
                    palette=paradigm_palette, edgecolor='black', linewidth=1.2, dodge=False, ax=axes[0]
                )
                for container in ax1.containers:
                    ax1.bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
                axes[0].set_title("Kendall's W", fontsize=14, fontweight='bold')
                axes[0].set_ylabel("Score (1.0 = Perfect Rank Agreement)", fontsize=13)
                axes[0].set_xlabel("")
                axes[0].set_ylim(0, 1.15)
                axes[0].tick_params(axis='x', rotation=15)
                if ax1.legend_: ax1.legend_.remove()
                
                ax2 = sns.barplot(
                    data=model_df, x='Paradigm', y='Spearmans_Rho', hue='Paradigm', 
                    palette=paradigm_palette, edgecolor='black', linewidth=1.2, dodge=False, ax=axes[1]
                )
                for container in ax2.containers:
                    ax2.bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
                axes[1].set_title("Spearman's Rho (Mean Pairwise)", fontsize=14, fontweight='bold')
                axes[1].set_ylabel("Correlation (1.0 = Perfect Agreement)", fontsize=13)
                axes[1].set_xlabel("")
                axes[1].set_ylim(0, 1.15)
                axes[1].tick_params(axis='x', rotation=15)
                if ax2.legend_: ax2.legend_.remove()
                
                plt.tight_layout()
                
                pdf_path = os.path.join(OUT_DIR, f"{prefix}Fig2_{model}_Ranking_Consistency.pdf")
                plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  -> Saved: {pdf_path}")
                
            print(f"\n[DONE] Saved CSV and all independent Model Plots to: {OUT_DIR}")
        else:
            print("Error: DataFrame returned empty after dropna checks. See diagnostic printouts above.")
    else:
        print("Error: No data extracted. Check paths and JSON structure.")
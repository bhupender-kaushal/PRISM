import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. USER MATH: SPOT S-FACTOR 
# ==========================================
LAMBDA_P = 1.0

def calculate_spot_s_factor(ic_0, oc_0, ic_k, oc_k):
    """
    Calculates a strictly localized S_factor for a single severity spot (step k).
    This removes integration smoothing to ensure a fair apples-to-apples comparison 
    against the raw IoU spots of the I/D framework.
    """
    ic_0 = max(1e-6, float(ic_0))
    oc_0 = max(1e-6, float(oc_0))
    ic_k = float(ic_k)
    oc_k = float(oc_k)
    
    # Bounded relative utility at step k
    ic_hat = min(1.0, ic_k / ic_0)
    oc_hat = min(1.0, oc_k / oc_0)
    
    # Pure spot gap and absolute drift
    G_gap = max(0.0, oc_hat - ic_hat)
    E_drift = abs(oc_hat - 1.0)
    
    return math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift)) * 100.0

# ==========================================
# 2. DYNAMIC PER-IMAGE PARSERS
# ==========================================
def extract_noises(analysis_dict):
    if not isinstance(analysis_dict, dict): return {}
    if 'level_data_per_noise' in analysis_dict:
        return analysis_dict['level_data_per_noise']
    return analysis_dict

def get_trajectory_percentage_tier(i, N_image_trajectory):
    if N_image_trajectory <= 1: return '100_pct'
    pct = i / (N_image_trajectory - 1)
    
    if pct == 0.0: return None 
    elif pct <= 0.25: return '25_pct'
    elif pct <= 0.50: return '50_pct'
    elif pct <= 0.75: return '75_pct'
    else: return '100_pct'

def parse_hybrid_framework(cont_dirs, id_dirs):
    print(">>> Extracting Spot Data via Dynamic Trajectory Step Percentages...")
    records = []
    
    # --- 1. Continuous Framework ---
    for model_name, path_str in cont_dirs.items():
        run_1_path = Path(path_str) / "run_1"
        if not run_1_path.exists(): continue
        
        for p in run_1_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            image_id = p.stem.replace('_analysis', '')
            
            ic_noises = extract_noises(data.get('in_context_analysis', {}))
            oc_noises = extract_noises(data.get('out_of_context_analysis', {}))
            
            # A. Spot S-Factor for IC/OC Composite
            shared_noises = set(ic_noises.keys()).intersection(set(oc_noises.keys()))
            for noise in shared_noises:
                ic_item, oc_item = ic_noises[noise], oc_noises[noise]
                ic_lvls = ic_item.get('level_data', []) if isinstance(ic_item, dict) else (ic_item if isinstance(ic_item, list) else [])
                oc_lvls = oc_item.get('level_data', []) if isinstance(oc_item, dict) else (oc_item if isinstance(oc_item, list) else [])
                
                N_min = min(len(ic_lvls), len(oc_lvls))
                if N_min > 1:
                    # Anchor baseline values
                    ic_0 = ic_lvls[0].get("discounted_utility", ic_lvls[0].get("global_iou", 0.0))
                    oc_0 = oc_lvls[0].get("discounted_utility", oc_lvls[0].get("global_iou", 0.0))
                    
                    # Iterate through the exact same spots used by Global/ID
                    for i in range(1, N_min):
                        tier = get_trajectory_percentage_tier(i, N_min)
                        if tier:
                            ic_k = ic_lvls[i].get("discounted_utility", ic_lvls[i].get("global_iou", 0.0))
                            oc_k = oc_lvls[i].get("discounted_utility", oc_lvls[i].get("global_iou", 0.0))
                            
                            spot_s = calculate_spot_s_factor(ic_0, oc_0, ic_k, oc_k)
                            
                            records.append({
                                'model': model_name, 'image_id': image_id,
                                'paradigm': 'IC/OC Composite (Spot S-Factor)',
                                'noise': noise.capitalize(), 'severity_tier': tier,
                                'final_score': spot_s
                            })
            
            # B. Raw Spot IoU for Continuous Global
            for key, content in data.items():
                if 'global' in key.lower() and 'analysis' in key.lower():
                    gl_noises = extract_noises(content)
                    for noise, metrics in gl_noises.items():
                        lvls = metrics.get('level_data', []) if isinstance(metrics, dict) else (metrics if isinstance(metrics, list) else [])
                        N_gl = len(lvls)
                        for i, step in enumerate(lvls):
                            tier = get_trajectory_percentage_tier(i, N_gl)
                            if tier:
                                records.append({
                                    'model': model_name, 'image_id': image_id,
                                    'paradigm': 'Continuous Global (IoU)',
                                    'noise': noise.capitalize(), 'severity_tier': tier,
                                    'final_score': step.get('global_iou', 100.0)
                                })
                                
    # --- 2. Standard I/D Framework ---
    for model_name, path_str in id_dirs.items():
        base_path = Path(path_str)
        if not base_path.exists(): continue
        
        for p in base_path.rglob('*_analysis.json'):
            with open(p, 'r') as f: data = json.load(f)
            if 'eigen_guided_analysis' not in data: continue
            image_id = p.stem.replace('_analysis', '')
            
            for baseline, ranks in data['eigen_guided_analysis'].get('trajectories', {}).items():
                modes = ranks.get('eigen_ranked', {}).get('deletion', [])
                lvls = modes if isinstance(modes, list) else modes.get('level_data', [])
                
                N_id = len(lvls)
                for i, mode in enumerate(lvls):
                    tier = get_trajectory_percentage_tier(i, N_id)
                    if tier:
                        records.append({
                            'model': model_name, 'image_id': image_id,
                            'paradigm': 'I/D Deletion (IoU)',
                            'noise': baseline.capitalize(), 'severity_tier': tier,
                            'final_score': mode.get('global_iou', 0.0)
                        })
                        
    return pd.DataFrame(records)

# ==========================================
# 3. STATISTICAL ENGINES
# ==========================================
def compute_kendalls_w(rank_matrix):
    n, m = rank_matrix.shape
    if n < 2 or m < 2: return np.nan
    R_i = np.sum(rank_matrix, axis=1)
    R_mean = np.mean(R_i)
    S = np.sum((R_i - R_mean)**2)
    denom = (m**2 * (n**3 - n))
    return (12 * S) / denom if denom > 0 else 0.0

def run_global_kendalls_w(df_all, iterations=100, n_images=10):
    print(f"\n>>> Running Global Kendall's W (All Perturbations Combined)...")
    results = []
    
    df_all = df_all.copy()
    df_all['rater_id'] = df_all['noise'] + "_" + df_all['severity_tier'].astype(str)
    
    for paradigm in df_all['paradigm'].unique():
        p_df = df_all[df_all['paradigm'] == paradigm]
        
        try:
            agg_df = p_df.pivot_table(index='image_id', columns='rater_id', values='final_score').dropna(axis=0, how='any')
        except Exception:
            continue
            
        valid_images = agg_df.index.tolist()
        if len(valid_images) < n_images: continue
            
        w_scores = []
        for _ in range(iterations):
            sampled_df = agg_df.loc[np.random.choice(valid_images, size=n_images, replace=False)]
            rank_matrix = np.zeros_like(sampled_df.values)
            for col_idx in range(sampled_df.shape[1]):
                rank_matrix[:, col_idx] = rankdata(-sampled_df.iloc[:, col_idx].values)
            w_scores.append(compute_kendalls_w(rank_matrix))
            
        results.append({
            'Paradigm': paradigm,
            'Total_Raters': agg_df.shape[1],
            'Global_Kendalls_W': np.mean(w_scores),
            'Std_Dev': np.std(w_scores)
        })
        
    return pd.DataFrame(results).round(3)

def run_specific_kendalls_w(df_all, iterations=100, n_images=10):
    print(f"\n>>> Running Specific Kendall's W (Per-Perturbation, excluding trivial I/D canvases)...")
    results = []
    
    for paradigm in df_all['paradigm'].unique():
        p_df = df_all[df_all['paradigm'] == paradigm]
        
        for noise in p_df['noise'].unique():
            if 'I/D' in paradigm and noise.lower() in ['black', 'mean']:
                continue
                
            pn_df = p_df[p_df['noise'] == noise]
            agg_df = pn_df.groupby(['image_id', 'severity_tier'])['final_score'].mean().unstack().dropna()
            
            valid_images = agg_df.index.tolist()
            if len(valid_images) < n_images: continue
                
            w_scores = []
            for _ in range(iterations):
                sampled_df = agg_df.loc[np.random.choice(valid_images, size=n_images, replace=False)]
                rank_matrix = np.zeros_like(sampled_df.values)
                for col_idx in range(sampled_df.shape[1]):
                    rank_matrix[:, col_idx] = rankdata(-sampled_df.iloc[:, col_idx].values)
                w_scores.append(compute_kendalls_w(rank_matrix))
                
            results.append({
                'Paradigm': paradigm,
                'Perturbation': noise,
                'Specific_Kendalls_W': np.mean(w_scores),
                'Std_Dev': np.std(w_scores)
            })
            
    return pd.DataFrame(results).round(3)

def compute_jsd_matrix(df_all, output_dir):
    print(">>> Calculating Jensen-Shannon Divergence (JSD)...")
    results = []
    bins = np.linspace(0, 100, 101)
    eps = 1e-10
    
    for model in df_all['model'].unique():
        m_df = df_all[df_all['model'] == model]
        
        def get_dist(paradigm_name):
            vals = m_df[m_df['paradigm'] == paradigm_name]['final_score']
            if vals.empty: return np.ones(100)/100.0
            d = np.histogram(vals, bins=bins, density=True)[0]
            d = d + eps
            return d / d.sum()

        dist_id = get_dist('I/D Deletion (IoU)')
        dist_icoc = get_dist('IC/OC Composite (Spot S-Factor)')
        dist_gl = get_dist('Continuous Global (IoU)')
        
        results.append({
            'Model': model,
            'JSD (I/D vs IC/OC)': jensenshannon(dist_id, dist_icoc),
            'JSD (I/D vs Global)': jensenshannon(dist_id, dist_gl),
            'JSD (IC/OC vs Global)': jensenshannon(dist_icoc, dist_gl)
        })
        
    jsd_df = pd.DataFrame(results).round(3)
    jsd_df.to_csv(os.path.join(output_dir, "JSD_Distribution_Divergence.csv"), index=False)
    return jsd_df

# ==========================================
# 4. PLOTTING ENGINE 
# ==========================================
def plot_meta_evaluation_pdfs(jsd_df, w_global, w_specific, output_dir):
    print(">>> Generating PDF Visualizations...")
    
    paradigm_palette = {
        'I/D Deletion (IoU)': '#d62728', 
        'IC/OC Composite (Spot S-Factor)': '#1f77b4', 
        'Continuous Global (IoU)': '#ff7f0e'
    }

    plt.figure(figsize=(10, 6))
    sns.heatmap(jsd_df.set_index('Model').T, annot=True, cmap="YlOrRd", vmin=0, vmax=1, fmt=".2f", linewidths=1)
    plt.title("Distribution Shift (JSD) Between Evaluation Metrics", fontsize=15, fontweight='bold', pad=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "JSD_Distribution_Heatmap.pdf"), format='pdf')
    plt.close()
    
    if not w_global.empty:
        plt.figure(figsize=(9, 6))
        sns.barplot(data=w_global, x='Paradigm', y='Global_Kendalls_W', palette=paradigm_palette, edgecolor='black', linewidth=1.5)
        plt.title("Global Metric Consistency\n(Rank Stability Across ALL Perturbations & Severities)", fontsize=14, fontweight='bold')
        plt.ylabel("Kendall's W Concordance", fontsize=12)
        plt.xlabel("")
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Kendalls_W_Global_Stability.pdf"), format='pdf')
        plt.close()

    if not w_specific.empty:
        plt.figure(figsize=(14, 7))
        sns.barplot(data=w_specific, x='Perturbation', y='Specific_Kendalls_W', hue='Paradigm', palette=paradigm_palette, edgecolor='black', linewidth=1.2)
        plt.title("Perturbation-Specific Metric Consistency\n(Image Rank Stability Within Individual Trajectories)", fontsize=15, fontweight='bold')
        plt.ylabel("Kendall's W Concordance", fontsize=12)
        plt.xlabel("Perturbation Type", fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Kendalls_W_Specific_Stability.pdf"), format='pdf')
        plt.close()

# ==========================================
# 5. MASTER CONTROLLER
# ==========================================
if __name__ == "__main__":
    
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
    
    OUT_DIR = "./meta_evaluation_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df_hybrid = parse_hybrid_framework(CONTINUOUS_DIRS, ID_DIRS)
    
    if not df_hybrid.empty:
        jsd_df = compute_jsd_matrix(df_hybrid, OUT_DIR)
        
        w_global = run_global_kendalls_w(df_hybrid, iterations=100, n_images=10)
        if not w_global.empty:
            w_global = w_global.sort_values(by='Global_Kendalls_W', ascending=False)
            w_global.to_csv(os.path.join(OUT_DIR, "Global_Kendalls_W.csv"), index=False)
        
        w_specific = run_specific_kendalls_w(df_hybrid, iterations=100, n_images=10)
        if not w_specific.empty:
            w_specific.to_csv(os.path.join(OUT_DIR, "Specific_Kendalls_W.csv"), index=False)
        
        plot_meta_evaluation_pdfs(jsd_df, w_global, w_specific, OUT_DIR)
        
        print("\n=== GLOBAL STABILITY SUMMARY ===")
        print(w_global.to_string(index=False) if not w_global.empty else "No Global data generated.")
        
        print("\n=== SPECIFIC STABILITY SUMMARY ===")
        if not w_specific.empty:
            print(w_specific.groupby('Paradigm')['Specific_Kendalls_W'].mean().reset_index().to_string(index=False))
        else:
            print("No Specific data generated.")
            
        print(f"\n[DONE] Saved all CSVs and PDFs to: {OUT_DIR}")
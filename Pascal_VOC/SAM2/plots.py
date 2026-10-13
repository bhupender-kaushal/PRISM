import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from pathlib import Path

# Set seaborn style for publication-ready plots
sns.set_theme(style="whitegrid")
COMMON_X = np.linspace(0.0, 1.0, 100)

# ==========================================
# CONFIGURATION & DIRECTORIES
# ==========================================
PAIRINGS = {
    "Pascal-VOC": ("SAMv1", "SAMv2"),
    "ISIC-2016": ("MSDUNet", "TransFuse"),
    "Synapse": ("D-LKA", "TransDeeplab"),
    "XCAD": ("CDARL", "DARL")
}

CONT_DIRS = {
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

OUT_DIR = "./Dataset_PDF_Reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# EXTRACTION & MATH FUNCTIONS
# ==========================================
def extract_noises(analysis_dict):
    if not isinstance(analysis_dict, dict): return {}
    return analysis_dict.get('level_data_per_noise', analysis_dict)

def clean_noise_name(raw_name):
    lower = str(raw_name).strip().lower()
    no_space = lower.replace(" ", "")
    if ('salt' in lower and 'pepper' in lower) or ('s&p' in no_space): return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def get_interpolated(x_raw, y_raw):
    if len(x_raw) < 2 or len(y_raw) < 2: return np.full_like(COMMON_X, np.nan)
    x_raw, indices = np.unique(x_raw, return_index=True)
    y_raw = np.array(y_raw)[indices]
    f = interp1d(x_raw, y_raw, kind='linear', fill_value=(y_raw[0], y_raw[-1]), bounds_error=False)
    return f(COMMON_X)

def calculate_rbo_trajectory(levels):
    if not levels or not isinstance(levels[0], dict): return [], []
    
    sps_0 = levels[0].get('superpixels', [])
    if not sps_0: return [], []
    
    rank_map_0 = {sp['sp_id']: sp.get('step_prob_rank', sp.get('rank', 0)) for sp in sps_0}
    N = len(rank_map_0)
    if N <= 1: return [], []
    
    max_jumps = math.floor((N**2) / 2.0)
    y_rbo, x_axis = [], []
    
    for k, lvl in enumerate(levels):
        if not isinstance(lvl, dict): continue
        sps_k = lvl.get('superpixels', [])
        if not sps_k: continue
        
        rank_map_k = {sp['sp_id']: sp.get('step_prob_rank', sp.get('rank', 0)) for sp in sps_k}
        
        total_jumps = sum(abs(rank_map_k.get(sp_id, rank_map_0[sp_id]) - rank_map_0[sp_id]) for sp_id in rank_map_0)
        rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
        
        y_rbo.append(1.0 - rp)
        x_axis.append(k / (len(levels) - 1))
        
    return x_axis, y_rbo

# ==========================================
# PLOTTING ENGINE
# ==========================================
def plot_panel(ax, title, xlabel, ylabel, data_dict, models, metrics, colors, styles):
    plotted = False
    for metric in metrics:
        for model in models:
            matrix = np.array([arr for arr in data_dict[model][metric] if not np.isnan(arr).all()])
            if len(matrix) > 0:
                mean_curve = np.nanmean(matrix, axis=0)
                ax.plot(COMMON_X, mean_curve, label=f'{model} ({metric})', 
                        color=colors.get(metric, '#000000'), linestyle=styles[model], linewidth=2.5)
                plotted = True
                
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    if plotted: ax.legend(loc='best', fontsize=9, ncol=2)

def generate_dataset_report(dataset_name, model_A, model_B):
    model_styles = {model_A: '-', model_B: '--'}
    
    # 1. Discover Continuous Noises (FIXED: Saving strictly cleaned names)
    discovered_noises = set()
    for model_name in [model_A, model_B]:
        cont_path = CONT_DIRS.get(model_name)
        if cont_path and os.path.exists(cont_path):
            run_1_path = Path(cont_path) / "run_1"
            if not run_1_path.exists(): continue
            for p in run_1_path.rglob('*_analysis.json'):
                try:
                    with open(p, 'r') as f: data = json.load(f)
                    ic_noises = extract_noises(data.get('in_context_analysis', {}))
                    for k in ic_noises.keys():
                        discovered_noises.add(clean_noise_name(k))
                except: pass
                
    clean_noises = sorted(list(discovered_noises))
    
    total_rows = 2 + len(clean_noises)
    fig, axes = plt.subplots(total_rows, 2, figsize=(16, 5 * total_rows))
    fig.suptitle(f'{dataset_name} Comprehensive Evaluation: {model_A} vs {model_B}', 
                 fontsize=20, fontweight='bold', y=1.0 - (0.05 / total_rows))

    # ----------------------------------------------------
    # ROWS 0 & 1: I/D (BLACK & MEAN)
    # ----------------------------------------------------
    id_baselines = ['black', 'mean']
    id_data = {baseline: {model_A: {'Insertion_Perf': [], 'Insertion_RBO': [], 'Deletion_Perf': [], 'Deletion_RBO': []},
                          model_B: {'Insertion_Perf': [], 'Insertion_RBO': [], 'Deletion_Perf': [], 'Deletion_RBO': []}}
               for baseline in id_baselines}
    
    for model_name in [model_A, model_B]:
        id_path = Path(ID_DIRS.get(model_name, ""))
        if not id_path.exists(): continue
        for p in id_path.rglob('*_analysis.json'):
            try:
                with open(p, 'r') as f: data = json.load(f)
                trajectories = data.get('eigen_guided_analysis', {}).get('trajectories', {})
                
                for baseline in id_baselines:
                    if baseline in trajectories:
                        ranks = trajectories[baseline]
                        if 'eigen_ranked' not in ranks: continue
                        traj = ranks['eigen_ranked']
                        
                        # Deletion
                        del_lvls = traj.get('deletion', [])
                        if isinstance(del_lvls, dict): del_lvls = del_lvls.get('level_data', [])
                        if len(del_lvls) > 1:
                            x = [float(s.get('fraction_perturbed', 0.0)) for s in del_lvls]
                            y_perf = [float(s.get('global_iou', 0.0)) / 100.0 for s in del_lvls] 
                            x_rbo, y_rbo = calculate_rbo_trajectory(del_lvls)
                            id_data[baseline][model_name]['Deletion_Perf'].append(get_interpolated(x, y_perf))
                            id_data[baseline][model_name]['Deletion_RBO'].append(get_interpolated(x_rbo, y_rbo))
                        
                        # Insertion
                        ins_lvls = traj.get('insertion', [])
                        if isinstance(ins_lvls, dict): ins_lvls = ins_lvls.get('level_data', [])
                        if len(ins_lvls) > 1:
                            x = [float(s.get('fraction_perturbed', 0.0)) for s in ins_lvls]
                            y_perf = [float(s.get('global_iou', 0.0)) / 100.0 for s in ins_lvls]
                            x_rbo, y_rbo = calculate_rbo_trajectory(ins_lvls)
                            id_data[baseline][model_name]['Insertion_Perf'].append(get_interpolated(x, y_perf))
                            id_data[baseline][model_name]['Insertion_RBO'].append(get_interpolated(x_rbo, y_rbo))
            except Exception: pass

    ins_colors = {'Insertion_Perf': '#2ca02c', 'Deletion_Perf': '#d62728', 'Insertion_RBO': '#2ca02c', 'Deletion_RBO': '#d62728'}
    
    plot_panel(axes[0, 0], 'Standard I/D Performance (Black Canvas)', 'Fraction Perturbed / Revealed', 'Normalized IoU', 
               id_data['black'], [model_A, model_B], ['Insertion_Perf', 'Deletion_Perf'], ins_colors, model_styles)
    plot_panel(axes[0, 1], 'Standard I/D Rank Flipping (Black Canvas)', 'Fraction Perturbed / Revealed', 'Stability Score (1.0 = Perfect)', 
               id_data['black'], [model_A, model_B], ['Insertion_RBO', 'Deletion_RBO'], ins_colors, model_styles)
               
    plot_panel(axes[1, 0], 'Standard I/D Performance (Mean Canvas)', 'Fraction Perturbed / Revealed', 'Normalized IoU', 
               id_data['mean'], [model_A, model_B], ['Insertion_Perf', 'Deletion_Perf'], ins_colors, model_styles)
    plot_panel(axes[1, 1], 'Standard I/D Rank Flipping (Mean Canvas)', 'Fraction Perturbed / Revealed', 'Stability Score (1.0 = Perfect)', 
               id_data['mean'], [model_A, model_B], ['Insertion_RBO', 'Deletion_RBO'], ins_colors, model_styles)


    # ----------------------------------------------------
    # ROWS 2+: CONTINUOUS EXTRACTIONS (Global, IC, OC)
    # ----------------------------------------------------
    for row_idx, noise in enumerate(clean_noises, start=2):
        noise_data = {
            model_A: {'Global_Perf': [], 'Global_RBO': [], 'IC_Perf': [], 'IC_RBO': [], 'OC_Perf': [], 'OC_RBO': []},
            model_B: {'Global_Perf': [], 'Global_RBO': [], 'IC_Perf': [], 'IC_RBO': [], 'OC_Perf': [], 'OC_RBO': []}
        }
        
        for model_name in [model_A, model_B]:
            if model_name not in CONT_DIRS: continue
            run_1_path = Path(CONT_DIRS[model_name]) / "run_1"
            if not run_1_path.exists(): continue
            
            for p in run_1_path.rglob('*_analysis.json'):
                try:
                    with open(p, 'r') as f: data = json.load(f)
                    ic_noises = extract_noises(data.get('in_context_analysis', {}))
                    oc_noises = extract_noises(data.get('out_of_context_analysis', {}))
                    
                    # Target correct string mapping regardless of original json casing
                    target_ic_raw = next((k for k in ic_noises.keys() if clean_noise_name(k) == noise), None)
                    target_oc_raw = next((k for k in oc_noises.keys() if clean_noise_name(k) == noise), None)
                    
                    if target_ic_raw:
                        ic_lvls = ic_noises[target_ic_raw].get('level_data', []) if isinstance(ic_noises[target_ic_raw], dict) else ic_noises[target_ic_raw]
                        if len(ic_lvls) > 1:
                            x = [k / (len(ic_lvls) - 1) for k in range(len(ic_lvls))]
                            y_perf = [float(lvl.get("global_iou", 0.0)) / 100.0 for lvl in ic_lvls]
                            x_rbo, y_rbo = calculate_rbo_trajectory(ic_lvls)
                            noise_data[model_name]['IC_Perf'].append(get_interpolated(x, y_perf))
                            noise_data[model_name]['IC_RBO'].append(get_interpolated(x_rbo, y_rbo))
                            
                    if target_oc_raw:
                        oc_lvls = oc_noises[target_oc_raw].get('level_data', []) if isinstance(oc_noises[target_oc_raw], dict) else oc_noises[target_oc_raw]
                        if len(oc_lvls) > 1:
                            x = [k / (len(oc_lvls) - 1) for k in range(len(oc_lvls))]
                            y_perf = [float(lvl.get("global_iou", 0.0)) / 100.0 for lvl in oc_lvls]
                            x_rbo, y_rbo = calculate_rbo_trajectory(oc_lvls)
                            noise_data[model_name]['OC_Perf'].append(get_interpolated(x, y_perf))
                            noise_data[model_name]['OC_RBO'].append(get_interpolated(x_rbo, y_rbo))

                    global_key = next((k for k in data.keys() if 'global' in k.lower() and 'analysis' in k.lower()), None)
                    if global_key:
                        gl_noises = extract_noises(data[global_key])
                        target_gl_raw = next((k for k in gl_noises.keys() if clean_noise_name(k) == noise), None)
                        if target_gl_raw:
                            gl_lvls = gl_noises[target_gl_raw].get('level_data', []) if isinstance(gl_noises[target_gl_raw], dict) else gl_noises[target_gl_raw]
                            if len(gl_lvls) > 1:
                                x = [k / (len(gl_lvls) - 1) for k in range(len(gl_lvls))]
                                y_perf = [float(s.get('global_iou', 0.0)) / 100.0 for s in gl_lvls]
                                x_rbo, y_rbo = calculate_rbo_trajectory(gl_lvls)
                                noise_data[model_name]['Global_Perf'].append(get_interpolated(x, y_perf))
                                noise_data[model_name]['Global_RBO'].append(get_interpolated(x_rbo, y_rbo))
                except Exception: pass

        perf_colors = {'Global_Perf': '#2ca02c', 'IC_Perf': '#1f77b4', 'OC_Perf': '#9467bd'}
        plot_panel(axes[row_idx, 0], f'Continuous Performance ({noise})', 'Normalized Severity Step', 'Normalized IoU', 
                   noise_data, [model_A, model_B], ['Global_Perf', 'IC_Perf', 'OC_Perf'], perf_colors, model_styles)
        
        rbo_colors = {'Global_RBO': '#2ca02c', 'IC_RBO': '#1f77b4', 'OC_RBO': '#9467bd'}
        plot_panel(axes[row_idx, 1], f'Continuous Stability ({noise})', 'Normalized Severity Step', 'Stability Score (1.0 = Perfect)', 
                   noise_data, [model_A, model_B], ['Global_RBO', 'IC_RBO', 'OC_RBO'], rbo_colors, model_styles)

    plt.tight_layout(rect=[0, 0, 1, 1.0 - (0.03 / total_rows)])
    pdf_path = os.path.join(OUT_DIR, f"{dataset_name.replace(' ', '_')}_SinglePage_Report.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=300)
    plt.close()
    print(f"  -> Successfully Saved: {pdf_path}")

# ==========================================
# MASTER EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Starting Multi-Row Layout Generation for {len(PAIRINGS)} Datasets...\n")
    for dataset, (model_A, model_B) in PAIRINGS.items():
        print(f"Processing Dataset: {dataset} ({model_A} vs {model_B})")
        generate_dataset_report(dataset, model_A, model_B)
    print(f"\nAll single-page PDF reports successfully saved to '{OUT_DIR}'.")
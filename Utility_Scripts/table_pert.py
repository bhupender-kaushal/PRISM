import os
import json
import numpy as np
import pandas as pd
import argparse
import math

# ==========================================
# DATASET MAPPING
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

# --- MATHEMATICAL CONSTANTS ---
GAMMA = 1.0      # Exponential decay weight for integral
LAMBDA_P = 0.25  # For the Legacy Exponential formulation only

# --- CONSTRAINT: Evaluate only the first 5 runs ---
VALID_RUNS = {"run_1", "run_2", "run_3", "run_4", "run_5"}

def clean_noise_name(raw_name):
    lower_name = str(raw_name).strip().lower()
    no_space_name = lower_name.replace(" ", "")
    if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
        return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def load_metrics(directory):
    # Initialize store to separate 'global' average from 'noises' breakdown
    run_store = {run: {'global': {k: [] for k in ['E0', 'S_exp', 'M_exp', 'S_geo', 'M_geo', 'S_euc', 'M_euc']}, 
                       'noises': {}} for run in VALID_RUNS}

    if not os.path.exists(directory): return {}

    for run_folder in os.listdir(directory):
        if run_folder not in VALID_RUNS: 
            continue
            
        run_path = os.path.join(directory, run_folder)
        if not os.path.isdir(run_path): continue
            
        for img_folder in os.listdir(run_path):
            img_path = os.path.join(run_path, img_folder)
            if not os.path.isdir(img_path): continue
                
            json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
            if not os.path.exists(json_file): continue
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            ic_data = data.get('in_context_analysis', {})
            oc_data = data.get('out_of_context_analysis', {})
            shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))
            
            img_metrics_global = {k: [] for k in ['E0', 'S_exp', 'M_exp', 'S_geo', 'M_geo', 'S_euc', 'M_euc']}
            
            for raw_noise in shared_noises:
                noise = clean_noise_name(raw_noise)
                
                ic_levels = ic_data[raw_noise].get('level_data', [])
                oc_levels = oc_data[raw_noise].get('level_data', [])
                
                if not ic_levels or not oc_levels: continue
                max_steps = max(len(ic_levels), len(oc_levels))
                if max_steps <= 1: continue
                
                # Extract pre-computed Discounted Utility (out of 100)
                IC_vals = [lvl.get("discounted_utility", 0.0) for lvl in ic_levels]
                OC_vals = [lvl.get("discounted_utility", 0.0) for lvl in oc_levels]

                # Pad arrays dynamically if one stopped early due to SSIM triggers
                while len(IC_vals) < max_steps: IC_vals.append(IC_vals[-1])
                while len(OC_vals) < max_steps: OC_vals.append(OC_vals[-1])
                    
                ic_0, oc_0 = max(1e-6, IC_vals[0]), max(1e-6, OC_vals[0])
                E_0 = (ic_0 + oc_0) / 2.0
                
                # Trajectory Integral Math
                W, weighted_sep_sum, weighted_drift_sum = 0.0, 0.0, 0.0
                for k in range(1, max_steps):
                    eta_k = k / (max_steps - 1)
                    w_k = math.exp(-GAMMA * eta_k)
                    
                    ic_hat = IC_vals[k] / ic_0
                    oc_hat = OC_vals[k] / oc_0
                    
                    weighted_sep_sum += w_k * max(0.0, oc_hat - ic_hat)
                    weighted_drift_sum += w_k * ((oc_hat - 1.0) ** 2)
                    W += w_k
                    
                G_ic_oc = weighted_sep_sum / W if W > 0 else 0.0
                E_oc_drift = math.sqrt(weighted_drift_sum / W) if W > 0 else 0.0
                G_deficit = max(0.0, 1.0 - G_ic_oc)
                    
                # =========================================================
                # THREE COMBINERS CALCULATED SIMULTANEOUSLY (Scaled 0-100)
                # =========================================================
                S_exp = math.exp(-LAMBDA_P * G_deficit) * math.exp(-LAMBDA_P * E_oc_drift) * 100.0
                S_geo = math.sqrt(max(0.0, G_ic_oc) * max(0.0, 1.0 - E_oc_drift)) * 100.0
                S_euc = (1.0 - math.sqrt((G_deficit**2 + min(1.0, E_oc_drift)**2) / 2.0)) * 100.0
                
                metrics = {
                    'E0': E_0,
                    'S_exp': S_exp, 'M_exp': E_0 * (S_exp / 100.0),
                    'S_geo': S_geo, 'M_geo': E_0 * (S_geo / 100.0),
                    'S_euc': S_euc, 'M_euc': E_0 * (S_euc / 100.0)
                }

                # 1. Store locally for Global Average
                for k, v in metrics.items():
                    img_metrics_global[k].append(v)

                # 2. Store directly in Per-Noise Tracker
                if noise not in run_store[run_folder]['noises']:
                    run_store[run_folder]['noises'][noise] = {k: [] for k in ['E0', 'S_exp', 'M_exp', 'S_geo', 'M_geo', 'S_euc', 'M_euc']}
                for k, v in metrics.items():
                    run_store[run_folder]['noises'][noise][k].append(v)
                
            # Image-Level Averaging for Global
            if img_metrics_global['E0']:
                for key in img_metrics_global:
                    run_store[run_folder]['global'][key].append(np.mean(img_metrics_global[key]))

    # Condense the 5 runs to final means
    processed_global = {k: [] for k in ['E0', 'S_exp', 'M_exp', 'S_geo', 'M_geo', 'S_euc', 'M_euc']}
    processed_noises = {}

    for run_name, r_data in run_store.items():
        # Compile globals
        for key, vals in r_data['global'].items():
            processed_global[key].extend(vals)
        
        # Compile per-noise
        for noise, n_data in r_data['noises'].items():
            if noise not in processed_noises:
                processed_noises[noise] = {k: [] for k in ['E0', 'S_exp', 'M_exp', 'S_geo', 'M_geo', 'S_euc', 'M_euc']}
            for key, vals in n_data.items():
                processed_noises[noise][key].extend(vals)
            
    final_global = {k: np.mean(v) if v else np.nan for k, v in processed_global.items()}
    final_noises = {noise: {k: np.mean(v) if v else np.nan for k, v in n_data.items()} for noise, n_data in processed_noises.items()}
    
    return {'global': final_global, 'by_noise': final_noises}

def fv(val):
    if val is None or np.isnan(val): return "-"
    return f"{val:.1f}"

def print_latex_table(data_dict, title, caption):
    print("\n" + "="*95)
    print(title)
    print("="*95)
    
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(r"\begin{tabular}{lc | cc | cc | cc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Base ($E_0$)} & \multicolumn{2}{c|}{1. Soft Exponential ($\lambda=0.25$)} & \multicolumn{2}{c|}{2. Geometric Mean} & \multicolumn{2}{c}{3. Euclidean Distance} \\")
    print(r"& & $S_{factor}$ & $M_{final}$ & $S_{factor}$ & $M_{final}$ & $S_{factor}$ & $M_{final}$ \\")
    print(r"\midrule")
    
    for model, metrics in data_dict.items():
        if metrics is None or pd.isna(metrics.get('E0')): continue
        
        e0 = fv(metrics['E0'])
        s_ex, m_ex = fv(metrics['S_exp']), fv(metrics['M_exp'])
        s_ge, m_ge = fv(metrics['S_geo']), fv(metrics['M_geo'])
        s_eu, m_eu = fv(metrics['S_euc']), fv(metrics['M_euc'])
        
        print(f"{model} & {e0} & {s_ex} & {m_ex} & {s_ge} & {m_ge} & {s_eu} & {m_eu} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

def generate_all_tables(all_data):
    # 1. Output the Master Global Table
    global_data = {model: d['global'] for model, d in all_data.items() if d}
    print_latex_table(
        global_data, 
        "GLOBAL FORMULATION COMPARISON TABLE (Runs 1 to 5)", 
        "Comparison of Robustness Metric Formulations (Aggregated across all noises)"
    )
    
    # 2. Output the Noise-Specific Tables
    all_noises = set()
    for model, d in all_data.items():
        if d and 'by_noise' in d:
            all_noises.update(d['by_noise'].keys())
            
    for noise in sorted(all_noises):
        noise_data = {}
        for model, d in all_data.items():
            if d and 'by_noise' in d and noise in d['by_noise']:
                noise_data[model] = d['by_noise'][noise]
            else:
                noise_data[model] = None
                
        print_latex_table(
            noise_data, 
            f"FORMULATION COMPARISON TABLE - {noise.upper()}", 
            f"Comparison of Robustness Metric Formulations for {noise} Perturbations"
        )

def main():
    parser = argparse.ArgumentParser()
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
                        })
    args = parser.parse_args()

    all_data = {}
    for method, directory in args.directories.items():
        if os.path.exists(directory):
            print(f"Aggregating Runs 1-5 for {method}...")
            all_data[method] = load_metrics(directory)
        else:
            print(f"Skipping {method} - directory not found.")

    generate_all_tables(all_data)

if __name__ == "__main__":
    main()
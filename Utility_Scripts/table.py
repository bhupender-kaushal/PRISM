# import os
# import json
# import argparse
# import math
# import numpy as np
# import pandas as pd
# import logging

# # ==========================================
# # CONFIGURATION & CONSTANTS
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
# LAMBDA_P = 1.0

# def clean_noise_name(raw_name):
#     lower_name = str(raw_name).strip().lower()
#     no_space_name = lower_name.replace(" ", "")
#     if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
#         return 'Salt & Pepper'
#     return str(raw_name).strip().replace('_', ' ').title()

# # ==========================================
# # METRIC CALCULATORS
# # ==========================================
# def calculate_step_rp_and_iou(level_0, level_k):
#     """Calculates step-wise Raw IoU and Absolute Rank Perplexity (RP) for Table 2."""
#     sp_0 = {sp['sp_id']: sp.get('rank', 0) for sp in level_0.get('superpixels', [])}
#     sp_k = {sp['sp_id']: sp.get('rank', 0) for sp in level_k.get('superpixels', [])}
    
#     N = len(sp_0)
#     if N <= 1: return 0.0, 0.0
    
#     max_jumps = math.floor((N**2) / 2.0)
#     total_jumps = sum(abs(sp_k.get(sp_id, sp_0[sp_id]) - sp_0[sp_id]) for sp_id in sp_0)
#     rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
#     raw_iou = np.mean([sp['iou'] for sp in level_0.get('superpixels', [])]) if level_0.get('superpixels') else 0.0
#     return raw_iou, rp

# def calculate_holistic_metrics(ic_levels, oc_levels, ic_csr):
#     """Calculates pure structural Robustness (S_factor), decoupling E_0."""
#     IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels]
#     OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels]

#     N_ic, N_oc = len(IC_vals), len(OC_vals)
#     IC_0 = max(1e-6, IC_vals[0]) if N_ic > 0 else 1e-6
#     OC_0 = max(1e-6, OC_vals[0]) if N_oc > 0 else 1e-6
#     E_0_img = (IC_0 + OC_0) / 2.0

#     # 1. Independent IC Integration
#     W_ic, A_ic = 0.0, 0.0
#     for k in range(1, N_ic):
#         eta_k = k / (N_ic - 1) if N_ic > 1 else 0.0
#         w_k = math.exp(-GAMMA * eta_k)
#         A_ic += w_k * (IC_vals[k] / IC_0)
#         W_ic += w_k
#     norm_A_ic = A_ic / W_ic if W_ic > 0 else (IC_vals[0]/IC_0 if N_ic == 1 else 0.0)

#     # 2. Independent OC Integration & Drift
#     W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
#     for k in range(1, N_oc):
#         eta_k = k / (N_oc - 1) if N_oc > 1 else 0.0
#         w_k = math.exp(-GAMMA * eta_k)
#         oc_hat = OC_vals[k] / OC_0
        
#         A_oc += w_k * oc_hat
#         weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
#         W_oc += w_k

#     # 3. Strictly Bounded Gap & Drift
#     norm_A_oc = A_oc / W_oc if W_oc > 0 else (OC_vals[0]/OC_0 if N_oc == 1 else 0.0)
#     bounded_ic = min(1.0, norm_A_ic)
#     bounded_oc = min(1.0, norm_A_oc)
    
#     G_gap = max(0.0, bounded_oc - bounded_ic)
#     E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0
    
#     # 4. Pure S_factor as Final Score
#     S_factor = math.exp(-LAMBDA_P * (max(0.0, 1 - G_gap) + E_drift)) * 100.0
    
#     # 5. Extract RP and Discounted IoU for Table 2
#     avg_rp = 0.0
#     initial_iou = 0.0
#     if N_ic > 1:
#         initial_iou, _ = calculate_step_rp_and_iou(ic_levels[0], ic_levels[0])
#         rps = [calculate_step_rp_and_iou(ic_levels[0], lvl)[1] for lvl in ic_levels[1:]]
#         avg_rp = np.mean(rps) if rps else 0.0
        
#     discounted_iou = initial_iou * math.exp(-avg_rp) * math.exp(-ic_csr) * 100.0
    
#     return E_0_img, G_gap, E_drift, S_factor, initial_iou * 100.0, avg_rp, ic_csr, discounted_iou

# # ==========================================
# # DATA EXTRACTION
# # ==========================================
# def load_metrics_from_model(directory):
#     logger = logging.getLogger('tablegen')
#     run_store = {}
#     metric_keys = ['E_0', 'G', 'E_d', 'Robustness', 'Init_IoU', 'RP', 'CSR', 'Disc_IoU']

#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith("_analysis.json"):
#                 parts = root.replace('\\', '/').split('/')
#                 if "run_1" not in parts: continue
                        
#                 if "run_1" not in run_store:
#                     run_store["run_1"] = {'global': {k: [] for k in metric_keys}, 'by_noise': {}}
                    
#                 json_path = os.path.join(root, file)
                
#                 try:
#                     with open(json_path, 'r') as f:
#                         data = json.load(f)
#                         ic_data = data.get('in_context_analysis', {})
#                         oc_data = data.get('out_of_context_analysis', {})
#                         shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))
                        
#                         img_metrics = {k: [] for k in metric_keys}
                        
#                         for raw_noise in shared_noises:
#                             noise = clean_noise_name(raw_noise)
#                             if noise not in run_store["run_1"]['by_noise']:
#                                 run_store["run_1"]['by_noise'][noise] = {k: [] for k in metric_keys}
                            
#                             ic_levels = ic_data[raw_noise].get('level_data', [])
#                             oc_levels = oc_data[raw_noise].get('level_data', [])
#                             if not ic_levels or not oc_levels: continue
                            
#                             ic_csr = ic_data[raw_noise].get('spillover_metrics', {}).get('conditional_average_spillover', 0.0)
                            
#                             E0, G, E_d, S_fac, Init_IoU, RP, CSR, Disc_IoU = calculate_holistic_metrics(ic_levels, oc_levels, ic_csr)
                            
#                             res = [E0, G, E_d, S_fac, Init_IoU, RP, CSR, Disc_IoU]
#                             for i, k in enumerate(metric_keys):
#                                 run_store["run_1"]['by_noise'][noise][k].append(res[i])
#                                 img_metrics[k].append(res[i])
                                
#                         if img_metrics['Robustness']:
#                             for k in metric_keys:
#                                 run_store["run_1"]['global'][k].append(np.mean(img_metrics[k]))
                                
#                 except Exception as e:
#                     logger.warning(f"Failed to process {json_path}: {e}")

#     # Aggregate
#     processed = {'global': {}, 'by_noise': {}}
#     if "run_1" in run_store:
#         r_data = run_store["run_1"]
#         for k in metric_keys:
#             processed['global'][k] = np.mean(r_data['global'][k]) if r_data['global'][k] else np.nan
        
#         for noise, n_data in r_data['by_noise'].items():
#             processed['by_noise'][noise] = {}
#             for k in metric_keys:
#                 processed['by_noise'][noise][k] = np.mean(n_data[k]) if n_data[k] else np.nan
        
#     return processed

# def fv(val, dec=2):
#     return f"{val:.{dec}f}" if val is not None and not np.isnan(val) else "-"

# # ==========================================
# # LATEX GENERATORS
# # ==========================================
# def generate_table_1(all_data):
#     """Table 1: Dataset, Model, Gap, Drift, Robustness Score (S_factor)"""
#     print("\n" + "% "*45)
#     print("% TABLE 1: Structural Robustness (Grouped by Dataset)")
#     print("% "*45)
    
#     print(r"\begin{table*}[h]")
#     print(r"\centering")
#     print(r"\caption{Global Structural Robustness Grouped by Dataset}")
#     print(r"\begin{tabular}{ll ccc}")
#     print(r"\toprule")
#     print(r"Dataset & Model & Separation Gap ($G \uparrow$) & Background Drift ($E_d \downarrow$) & Robustness ($S_{factor} \uparrow$) \\")
#     print(r"\midrule")
    
#     ds_map = {}
#     for model in all_data:
#         ds = DATASET_MAPPING.get(model, "Unknown")
#         if ds not in ds_map: ds_map[ds] = []
#         ds_map[ds].append(model)
        
#     for dataset, models in ds_map.items():
#         for i, model in enumerate(models):
#             g = all_data[model].get('global', {})
#             if not g or pd.isna(g.get('Robustness')): continue
            
#             ds_str = f"\\multirow{{{len(models)}}}{{*}}{{{dataset}}}" if i == 0 else ""
#             print(f"{ds_str} & {model} & {fv(g['G'])} & {fv(g['E_d'])} & {fv(g['Robustness'])} \\\\")
#         print(r"\midrule")
        
#     print(r"\bottomrule")
#     print(r"\end{tabular}")
#     print(r"\end{table*}")


# def generate_table_2(all_data):
#     """Table 2: Model, Initial IoU, CSR, RP, Discounted IoU"""
#     print("\n" + "% "*45)
#     print("% TABLE 2: Global Legacy Metrics (IoU, CSR, RP)")
#     print("% "*45)
    
#     print(r"\begin{table*}[h]")
#     print(r"\centering")
#     print(r"\caption{Baseline Utility and Contextual Penalties (Global Averages)}")
#     print(r"\begin{tabular}{l cccc}")
#     print(r"\toprule")
#     print(r"Model & Initial IoU ($\%$) & Context Spillover (CSR) & Rank Perplexity (RP) & Discounted IoU ($\%$) \\")
#     print(r"\midrule")
    
#     for model, m_data in all_data.items():
#         g = m_data.get('global', {})
#         if not g or pd.isna(g.get('Init_IoU')): continue
        
#         print(f"{model} & {fv(g['Init_IoU'])} & {fv(g['CSR'], 4)} & {fv(g['RP'], 4)} & {fv(g['Disc_IoU'])} \\\\")
        
#     print(r"\bottomrule")
#     print(r"\end{tabular}")
#     print(r"\end{table*}")


# def generate_table_3(all_data):
#     """Table 3: Model, [Noise 1: G, E_d, S], [Noise 2: G, E_d, S]"""
#     print("\n" + "% "*45)
#     print("% TABLE 3: Perturbation-Specific Robustness Matrix")
#     print("% "*45)
    
#     ds_map = {}
#     for model in all_data:
#         ds = DATASET_MAPPING.get(model, "Unknown")
#         if ds not in ds_map: ds_map[ds] = []
#         ds_map[ds].append(model)
        
#     for dataset, models in ds_map.items():
#         noises = set()
#         for m in models: noises.update(all_data[m].get('by_noise', {}).keys())
#         noises = sorted(list(noises))
        
#         for i in range(0, len(noises), 2):
#             chunk = noises[i:i+2]
#             n1 = chunk[0]
#             n2 = chunk[1] if len(chunk) > 1 else None
#             col_format = "l ccc ccc" if n2 else "l ccc"
            
#             print(f"\n% --- {dataset} | {n1}" + (f" & {n2}" if n2 else "") + " ---")
#             print(r"\begin{table*}[h]")
#             print(r"\centering")
#             print(f"\\caption{{{dataset}: Structural Dynamics under {n1 + (' and ' + n2 if n2 else '')}}}")
#             print(f"\\begin{{tabular}}{{{col_format}}}")
#             print(r"\toprule")
            
#             if n2:
#                 print(r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{" + n1 + r"} & \multicolumn{3}{c}{" + n2 + r"} \\")
#                 print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
#                 print(r"& Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) & Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) \\")
#             else:
#                 print(r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{" + n1 + r"} \\")
#                 print(r"\cmidrule(lr){2-4}")
#                 print(r"& Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) \\")
                
#             print(r"\midrule")
            
#             for model in models:
#                 m_data = all_data[model].get('by_noise', {})
#                 row_str = f"{model}"
                
#                 for noise in chunk:
#                     if noise in m_data:
#                         d = m_data[noise]
#                         row_str += f" & {fv(d['G'])} & {fv(d['E_d'])} & {fv(d['Robustness'])}"
#                     else:
#                         row_str += " & - & - & -"
#                 row_str += r" \\"
#                 print(row_str)
                
#             print(r"\bottomrule")
#             print(r"\end{tabular}")
#             print(r"\end{table*}")

# def main():
#     parser = argparse.ArgumentParser()
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
#                         })
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger('tablegen')

#     all_data = {}
#     for method, directory in args.directories.items():
#         logger.info(f"Extracting Matrix Data for model: {method}")
#         if not os.path.exists(directory):
#             continue
#         all_data[method] = load_metrics_from_model(directory)

#     generate_table_1(all_data)
#     generate_table_2(all_data)
#     generate_table_3(all_data)

# if __name__ == "__main__":
#     main()



import os
import json
import argparse
import math
import numpy as np
import pandas as pd
import logging

# ==========================================
# CONFIGURATION & CONSTANTS
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
LAMBDA_P = 1.0

def clean_noise_name(raw_name):
    lower_name = str(raw_name).strip().lower()
    no_space_name = lower_name.replace(" ", "")
    if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
        return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def extract_noises(analysis_dict):
    """Safely extracts the noise dictionary regardless of JSON nesting variations."""
    if not isinstance(analysis_dict, dict): return {}
    return analysis_dict.get('level_data_per_noise', analysis_dict)

# ==========================================
# METRIC CALCULATORS
# ==========================================
def calculate_step_rp_and_iou(level_0, level_k):
    """Calculates step-wise Raw IoU and Absolute Rank Perplexity (RP) for Table 2."""
    if not isinstance(level_0, dict) or not isinstance(level_k, dict):
        return 0.0, 0.0
        
    sp_0 = {sp['sp_id']: sp.get('rank', 0) for sp in level_0.get('superpixels', [])}
    sp_k = {sp['sp_id']: sp.get('rank', 0) for sp in level_k.get('superpixels', [])}
    
    N = len(sp_0)
    if N <= 1: return 0.0, 0.0
    
    max_jumps = math.floor((N**2) / 2.0)
    total_jumps = sum(abs(sp_k.get(sp_id, sp_0[sp_id]) - sp_0[sp_id]) for sp_id in sp_0)
    rp = total_jumps / max_jumps if max_jumps > 0 else 0.0
    raw_iou = np.mean([sp['iou'] for sp in level_0.get('superpixels', [])]) if level_0.get('superpixels') else 0.0
    return raw_iou, rp

def calculate_holistic_metrics(ic_levels, oc_levels, ic_csr):
    """Calculates pure structural Robustness (S_factor), decoupling E_0."""
    IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels if isinstance(lvl, dict)]
    OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels if isinstance(lvl, dict)]

    N_min = min(len(IC_vals), len(OC_vals))
    if N_min == 0: return 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0
    
    IC_0 = max(1e-6, IC_vals[0])
    OC_0 = max(1e-6, OC_vals[0])
    E_0_img = (IC_0 + OC_0) / 2.0

    # 1. Independent IC Integration
    W_ic, A_ic = 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1) if N_min > 1 else 0.0
        w_k = math.exp(-GAMMA * eta_k)
        A_ic += w_k * (IC_vals[k] / IC_0)
        W_ic += w_k
    norm_A_ic = A_ic / W_ic if W_ic > 0 else (IC_vals[0]/IC_0 if N_min == 1 else 0.0)

    # 2. Independent OC Integration & Drift
    W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1) if N_min > 1 else 0.0
        w_k = math.exp(-GAMMA * eta_k)
        oc_hat = OC_vals[k] / OC_0
        
        A_oc += w_k * oc_hat
        weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
        W_oc += w_k

    # 3. Strictly Bounded Gap & Drift
    norm_A_oc = A_oc / W_oc if W_oc > 0 else (OC_vals[0]/OC_0 if N_min == 1 else 0.0)
    bounded_ic = min(1.0, norm_A_ic)
    bounded_oc = min(1.0, norm_A_oc)
    
    G_gap = max(0.0, bounded_oc - bounded_ic)
    E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0
    
    # 4. Pure S_factor as Final Score (Fixed the 1G_gap typo)
    S_factor = math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift)) * 100.0
    
    # 5. Extract RP and Discounted IoU for Table 2
    avg_rp = 0.0
    initial_iou = 0.0
    if N_min > 1:
        initial_iou, _ = calculate_step_rp_and_iou(ic_levels[0], ic_levels[0])
        rps = [calculate_step_rp_and_iou(ic_levels[0], lvl)[1] for lvl in ic_levels[1:N_min]]
        avg_rp = np.mean(rps) if rps else 0.0
        
    discounted_iou = initial_iou * math.exp(-avg_rp) * math.exp(-ic_csr) * 100.0
    
    return E_0_img, G_gap, E_drift, S_factor, initial_iou * 100.0, avg_rp, ic_csr, discounted_iou

# ==========================================
# DATA EXTRACTION
# ==========================================
def load_metrics_from_model(directory):
    logger = logging.getLogger('tablegen')
    run_store = {}
    metric_keys = ['E_0', 'G', 'E_d', 'Robustness', 'Init_IoU', 'RP', 'CSR', 'Disc_IoU']

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_analysis.json"):
                parts = root.replace('\\', '/').split('/')
                if "run_1" not in parts: continue
                        
                if "run_1" not in run_store:
                    run_store["run_1"] = {'global': {k: [] for k in metric_keys}, 'by_noise': {}}
                    
                json_path = os.path.join(root, file)
                
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        
                        # Robust dictionary extraction
                        ic_noises = extract_noises(data.get('in_context_analysis', {}))
                        oc_noises = extract_noises(data.get('out_of_context_analysis', {}))
                        shared_noises = set(ic_noises.keys()).intersection(set(oc_noises.keys()))
                        
                        img_metrics = {k: [] for k in metric_keys}
                        
                        for raw_noise in shared_noises:
                            noise = clean_noise_name(raw_noise)
                            if noise not in run_store["run_1"]['by_noise']:
                                run_store["run_1"]['by_noise'][noise] = {k: [] for k in metric_keys}
                            
                            # Robust list extraction
                            ic_item = ic_noises[raw_noise]
                            oc_item = oc_noises[raw_noise]
                            
                            ic_levels = ic_item.get('level_data', []) if isinstance(ic_item, dict) else (ic_item if isinstance(ic_item, list) else [])
                            oc_levels = oc_item.get('level_data', []) if isinstance(oc_item, dict) else (oc_item if isinstance(oc_item, list) else [])
                            
                            if not ic_levels or not oc_levels: continue
                            
                            ic_csr = 0.0
                            if isinstance(ic_item, dict):
                                ic_csr = ic_item.get('spillover_metrics', {}).get('conditional_average_spillover', 0.0)
                            
                            E0, G, E_d, S_fac, Init_IoU, RP, CSR, Disc_IoU = calculate_holistic_metrics(ic_levels, oc_levels, ic_csr)
                            
                            res = [E0, G, E_d, S_fac, Init_IoU, RP, CSR, Disc_IoU]
                            for i, k in enumerate(metric_keys):
                                run_store["run_1"]['by_noise'][noise][k].append(res[i])
                                img_metrics[k].append(res[i])
                                
                        if img_metrics['Robustness']:
                            for k in metric_keys:
                                run_store["run_1"]['global'][k].append(np.mean(img_metrics[k]))
                                
                except Exception as e:
                    logger.warning(f"Failed to process {json_path}: {e}")

    # Aggregate
    processed = {'global': {}, 'by_noise': {}}
    if "run_1" in run_store:
        r_data = run_store["run_1"]
        for k in metric_keys:
            processed['global'][k] = np.mean(r_data['global'][k]) if r_data['global'][k] else np.nan
        
        for noise, n_data in r_data['by_noise'].items():
            processed['by_noise'][noise] = {}
            for k in metric_keys:
                processed['by_noise'][noise][k] = np.mean(n_data[k]) if n_data[k] else np.nan
        
    return processed

def fv(val, dec=2):
    return f"{val:.{dec}f}" if val is not None and not np.isnan(val) else "-"

# ==========================================
# LATEX GENERATORS
# ==========================================
def generate_table_1(all_data):
    """Table 1: Dataset, Model, Gap, Drift, Robustness Score (S_factor)"""
    print("\n" + "% "*45)
    print("% TABLE 1: Structural Robustness (Grouped by Dataset)")
    print("% "*45)
    
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\caption{Global Structural Robustness Grouped by Dataset}")
    print(r"\begin{tabular}{ll ccc}")
    print(r"\toprule")
    print(r"Dataset & Model & Separation Gap ($G \uparrow$) & Background Drift ($E_d \downarrow$) & Robustness ($S_{factor} \uparrow$) \\")
    print(r"\midrule")
    
    ds_map = {}
    for model in all_data:
        ds = DATASET_MAPPING.get(model, "Unknown")
        if ds not in ds_map: ds_map[ds] = []
        ds_map[ds].append(model)
        
    for dataset, models in ds_map.items():
        for i, model in enumerate(models):
            g = all_data[model].get('global', {})
            if not g or pd.isna(g.get('Robustness')): continue
            
            ds_str = f"\\multirow{{{len(models)}}}{{*}}{{{dataset}}}" if i == 0 else ""
            print(f"{ds_str} & {model} & {fv(g['G'])} & {fv(g['E_d'])} & {fv(g['Robustness'])} \\\\")
        print(r"\midrule")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def generate_table_2(all_data):
    """Table 2: Model, Initial IoU, CSR, RP, Discounted IoU"""
    print("\n" + "% "*45)
    print("% TABLE 2: Global Legacy Metrics (IoU, CSR, RP)")
    print("% "*45)
    
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\caption{Baseline Utility and Contextual Penalties (Global Averages)}")
    print(r"\begin{tabular}{l cccc}")
    print(r"\toprule")
    print(r"Model & Initial IoU ($\%$) & Context Spillover (CSR) & Rank Perplexity (RP) & Discounted IoU ($\%$) \\")
    print(r"\midrule")
    
    for model, m_data in all_data.items():
        g = m_data.get('global', {})
        if not g or pd.isna(g.get('Init_IoU')): continue
        
        print(f"{model} & {fv(g['Init_IoU'])} & {fv(g['CSR'], 4)} & {fv(g['RP'], 4)} & {fv(g['Disc_IoU'])} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def generate_table_3(all_data):
    """Table 3: Model, [Noise 1: G, E_d, S], [Noise 2: G, E_d, S]"""
    print("\n" + "% "*45)
    print("% TABLE 3: Perturbation-Specific Robustness Matrix")
    print("% "*45)
    
    ds_map = {}
    for model in all_data:
        ds = DATASET_MAPPING.get(model, "Unknown")
        if ds not in ds_map: ds_map[ds] = []
        ds_map[ds].append(model)
        
    for dataset, models in ds_map.items():
        noises = set()
        for m in models: noises.update(all_data[m].get('by_noise', {}).keys())
        noises = sorted(list(noises))
        
        for i in range(0, len(noises), 2):
            chunk = noises[i:i+2]
            n1 = chunk[0]
            n2 = chunk[1] if len(chunk) > 1 else None
            col_format = "l ccc ccc" if n2 else "l ccc"
            
            print(f"\n% --- {dataset} | {n1}" + (f" & {n2}" if n2 else "") + " ---")
            print(r"\begin{table*}[h]")
            print(r"\centering")
            print(f"\\caption{{{dataset}: Structural Dynamics under {n1 + (' and ' + n2 if n2 else '')}}}")
            print(f"\\begin{{tabular}}{{{col_format}}}")
            print(r"\toprule")
            
            if n2:
                print(r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{" + n1 + r"} & \multicolumn{3}{c}{" + n2 + r"} \\")
                print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
                print(r"& Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) & Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) \\")
            else:
                print(r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{" + n1 + r"} \\")
                print(r"\cmidrule(lr){2-4}")
                print(r"& Gap ($G$) & Drift ($E_d$) & Score ($S_{fac}$) \\")
                
            print(r"\midrule")
            
            for model in models:
                m_data = all_data[model].get('by_noise', {})
                row_str = f"{model}"
                
                for noise in chunk:
                    if noise in m_data:
                        d = m_data[noise]
                        row_str += f" & {fv(d['G'])} & {fv(d['E_d'])} & {fv(d['Robustness'])}"
                    else:
                        row_str += " & - & - & -"
                row_str += r" \\"
                print(row_str)
                
            print(r"\bottomrule")
            print(r"\end{tabular}")
            print(r"\end{table*}")

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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('tablegen')

    all_data = {}
    for method, directory in args.directories.items():
        logger.info(f"Extracting Matrix Data for model: {method}")
        if not os.path.exists(directory):
            continue
        all_data[method] = load_metrics_from_model(directory)

    generate_table_1(all_data)
    generate_table_2(all_data)
    generate_table_3(all_data)

if __name__ == "__main__":
    main()
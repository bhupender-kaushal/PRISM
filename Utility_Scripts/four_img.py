import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

MAIN_PERTURBATION_KEYWORDS = ["gaussian noise", "gaussian", "gaussian blur", "rician", "rician noise"]

GAMMA = 1.0
LAMBDA_P = 1.0

def clean_noise_name(raw_name):
    lower_name = str(raw_name).strip().lower()
    no_space_name = lower_name.replace(" ", "")
    if ('salt' in lower_name and 'pepper' in lower_name) or ('s&p' in no_space_name):
        return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# ==========================================
# 2. MATHEMATICAL CORE (S_factor as Final Score)
# ==========================================
def compute_image_metrics(ic_levels, oc_levels):
    IC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in ic_levels]
    OC_vals = [float(lvl.get("discounted_utility", 0.0)) for lvl in oc_levels]

    N_ic, N_oc = len(IC_vals), len(OC_vals)
    if N_ic == 0 or N_oc == 0: return None

    IC_0 = max(1e-6, IC_vals[0])
    OC_0 = max(1e-6, OC_vals[0])
    E_0 = (IC_0 + OC_0) / 2.0

    W_ic, A_ic = 0.0, 0.0
    for k in range(1, N_ic):
        eta_k = k / (N_ic - 1) if N_ic > 1 else 0.0
        w_k = math.exp(-GAMMA * eta_k)
        A_ic += w_k * (IC_vals[k] / IC_0)
        W_ic += w_k
    norm_A_ic = A_ic / W_ic if W_ic > 0 else (IC_vals[0] / IC_0 if N_ic == 1 else 0.0)

    W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
    for k in range(1, N_oc):
        eta_k = k / (N_oc - 1) if N_oc > 1 else 0.0
        w_k = math.exp(-GAMMA * eta_k)
        oc_hat = OC_vals[k] / OC_0
        A_oc += w_k * oc_hat
        weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
        W_oc += w_k
    
    norm_A_oc = A_oc / W_oc if W_oc > 0 else (OC_vals[0]/OC_0 if N_oc == 1 else 0.0)
    
    bounded_oc = min(1.0, norm_A_oc)
    bounded_ic = min(1.0, norm_A_ic)
    E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0
    G_gap = max(0.0, bounded_oc - bounded_ic)
    
    # NEW LOGIC: S_factor is the absolute final robustness score (scaled to 100%)
    s_final = math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift)) * 100.0

    def normalize_curve(levels):
        ssims = [float(lvl.get("ssim", 1.0)) for lvl in levels]
        evals = [float(lvl.get("discounted_utility", 0.0)) for lvl in levels]
        drops = [1.0 - s for s in ssims]
        for i in range(1, len(drops)): drops[i] = max(drops[i], drops[i - 1])
        max_drop = drops[-1] if drops[-1] > 0 else 1e-6
        norm_x = [d / max_drop for d in drops]
        grid = np.linspace(0.0, 1.0, 50)
        return grid, np.interp(grid, norm_x, evals)

    grid_ic, curve_ic = normalize_curve(ic_levels)
    grid_oc, curve_oc = normalize_curve(oc_levels)

    return {
        "E_0": E_0, "G": G_gap, "E_d": E_drift, "S_final": s_final,
        "curve_ic": curve_ic, "curve_oc": curve_oc, "grid": grid_ic
    }

def scan_model_all_noises(base_dir):
    records = []
    for root, dirs, files in os.walk(base_dir):
        if "run_" not in root: continue
        for img_folder in dirs:
            img_path = os.path.join(root, img_folder)
            json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
            if not os.path.exists(json_file): continue
            
            try: data = load_json(json_file)
            except: continue
            
            ic_data = data.get("in_context_analysis", {})
            oc_data = data.get("out_of_context_analysis", {})
            shared_noises = set(ic_data.keys()).intersection(set(oc_data.keys()))
            
            for raw_noise in shared_noises:
                ic_levels = ic_data[raw_noise].get("level_data", [])
                oc_levels = oc_data[raw_noise].get("level_data", [])
                if not ic_levels or not oc_levels: continue
                
                res = compute_image_metrics(ic_levels, oc_levels)
                if res:
                    res["image_id"] = img_folder
                    res["noise_type"] = clean_noise_name(raw_noise)
                    records.append(res)
                    
    if not records: return {}
    df_all = pd.DataFrame(records)
    return {noise: df_noise for noise, df_noise in df_all.groupby("noise_type")}

def get_best_and_worst(df):
    """
    Selects Best and Worst based strictly on Structural Robustness (S_final).
    Requires E_0 >= 40% to ensure we aren't rewarding an empty prediction.
    """
    if df.empty: return None, None
    viable_df = df[df["E_0"] >= 40.0]
    
    if viable_df.empty:
        # Fallback if no images meet viability threshold
        best_row = df.sort_values(by="S_final", ascending=False).iloc[0]
        worst_row = df.sort_values(by="S_final", ascending=True).iloc[0]
    else:
        best_row = viable_df.sort_values(by="S_final", ascending=False).iloc[0]
        worst_row = viable_df.sort_values(by="S_final", ascending=True).iloc[0]
        
    return best_row, worst_row

# ==========================================
# 3. DATASET-WISE MATRIX PLOTTING
# ==========================================
def generate_dataset_matrix(dataset_name, models, model_data_dict, target_noises, output_path, is_main=False):
    if not models or not target_noises: return
    
    num_cols = len(target_noises)
    num_models = len(models)
    num_rows = num_models * 2  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows))
    
    if num_rows == 1 and num_cols == 1: axes = np.array([[axes]])
    elif num_rows == 1: axes = np.expand_dims(axes, axis=0)
    elif num_cols == 1: axes = np.expand_dims(axes, axis=1)

    ic_color, oc_color = "dodgerblue", "darkorange"

    for m_idx, model in enumerate(models):
        row_best = m_idx * 2
        row_worst = row_best + 1
        
        axes[row_best, 0].set_ylabel(f"{model}\nBEST CASE", fontsize=16, fontweight='bold', labelpad=15)
        axes[row_worst, 0].set_ylabel(f"{model}\nWORST CASE", fontsize=16, fontweight='bold', labelpad=15)
        
        for c_idx, noise in enumerate(target_noises):
            if m_idx == 0:
                axes[row_best, c_idx].set_title(f"{noise}", fontsize=22, fontweight='bold', pad=15)
                
            df_noise = model_data_dict[model].get(noise, pd.DataFrame())
            best_case, worst_case = get_best_and_worst(df_noise)
            
            for is_worst, case_data in enumerate([best_case, worst_case]):
                ax = axes[row_worst if is_worst else row_best, c_idx]
                
                if case_data is None:
                    ax.set_visible(False)
                    continue
                    
                grid = case_data["grid"]
                ax.plot(grid, case_data["curve_ic"], color=ic_color, lw=3)
                ax.plot(grid, case_data["curve_oc"], color=oc_color, lw=3)
                ax.axhline(y=case_data["E_0"], color="gray", linestyle="--", lw=2, alpha=0.8)
                
                # We no longer plot S_final as a flat line across the whole Y-axis since it's decoupled from E_0
                
                ax.set_ylim(-5, 105)
                ax.set_xlim(-0.02, 1.02)
                
                # Updated info string to reflect pure S_final
                info_str = (
                    f"ID: {case_data['image_id']}"
                    f"Gap ($G$): {case_data['G']:.2f} | Drift ($E_d$): {case_data['E_d']:.2f}\n"
                    f"Robustness Score: {case_data['S_final']:.1f}%"
                )
                
                ax.text(0.5, 0.15, info_str, transform=ax.transAxes, ha='center', va='top', 
                        fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
                
                ax.grid(False)
                if row_worst != num_rows - 1 or not is_worst:
                    ax.set_xticklabels([])

    fig.supxlabel("SSIM Drop (0.0 = Clean Image, 1.0 = Max Perturbed)", fontsize=24, fontweight='bold', y=0.01)
    
    legend_elements = [
        Line2D([0], [0], color=ic_color, lw=4, label='In-Context (IC) Utility'),
        Line2D([0], [0], color=oc_color, lw=4, label='Out-of-Context (OC) Utility'),
        Line2D([0], [0], color="gray", lw=2, linestyle='--', label='Initial IoU')
    ]
    
    legend_y = 0.98 if num_rows > 2 else 1.05
    title_y = 1.03 if num_rows > 2 else 1.20
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=3, fontsize=18, frameon=False)
    
    section_label = "MAIN MANUSCRIPT" if is_main else "SUPPLEMENTARY"
    plt.suptitle(f"{dataset_name}", fontsize=30, fontweight='bold', y=title_y)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.15, top=0.92 if num_rows > 2 else 0.85, bottom=0.05)
    
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {dataset_name} ({section_label}): {os.path.basename(output_path)}")

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

    out_dir = './evaluation_plots'
    os.makedirs(out_dir, exist_ok=True)

    print("Extracting Trajectories for Dataset Matrices (Pure Structural Robustness)...\n" + "="*70)
    
    model_data_dict = {}
    for model_name, base_dir in args.directories.items():
        if not os.path.exists(base_dir):
            continue
        print(f"Scanning {model_name}...")
        dfs_per_noise = scan_model_all_noises(base_dir)
        if dfs_per_noise:
            model_data_dict[model_name] = dfs_per_noise

    dataset_grouping = {}
    for m in model_data_dict:
        ds = DATASET_MAPPING.get(m, "Unknown_Dataset")
        if ds not in dataset_grouping: dataset_grouping[ds] = []
        dataset_grouping[ds].append(m)

    for dataset_name, models in dataset_grouping.items():
        print(f"\nProcessing Dataset: {dataset_name}")
        
        all_noises = set()
        for m in models:
            all_noises.update(model_data_dict[m].keys())
        all_noises = sorted(list(all_noises))
        
        main_noises = [n for n in all_noises if n.lower() in MAIN_PERTURBATION_KEYWORDS]
        
        if main_noises:
            out_main = os.path.join(out_dir, f"{dataset_name}_Main_BestWorst.pdf")
            generate_dataset_matrix(dataset_name, models, model_data_dict, main_noises, out_main, is_main=True)
            
        if all_noises:
            out_supp = os.path.join(out_dir, f"{dataset_name}_Supp_BestWorst_All.pdf")
            generate_dataset_matrix(dataset_name, models, model_data_dict, all_noises, out_supp, is_main=False)

if __name__ == "__main__":
    main()
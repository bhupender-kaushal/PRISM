import os
import json
import argparse
import math
import numpy as np
import pandas as pd
import seaborn as sns
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

GAMMA = 1.0
LAMBDA_P = 1.0

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def process_noise_trajectory(levels, region_name):
    if not levels: return pd.DataFrame()
    noises = [float(lvl.get("noise_level", 0.0)) for lvl in levels]
    evals = [float(lvl.get("discounted_utility", 0.0)) for lvl in levels]
    
    clean_noises, clean_evals = [], []
    for n, e in zip(noises, evals):
        if not clean_noises or n > clean_noises[-1]:
            clean_noises.append(n)
            clean_evals.append(e)
            
    if not clean_noises: return pd.DataFrame()
    
    fixed_grid = np.linspace(0.0, 3.0, 100)
    interp_evals = np.interp(fixed_grid, clean_noises, clean_evals)
    
    return pd.DataFrame({"X_Axis": fixed_grid, "E_val": interp_evals, "Region": region_name})

def process_ssim_lifecycle(levels, region_name):
    if not levels: return pd.DataFrame()
    ssims = [float(lvl.get("ssim", 1.0)) for lvl in levels]
    evals = [float(lvl.get("discounted_utility", 0.0)) for lvl in levels]
    
    drops = [1.0 - s for s in ssims]
    for i in range(1, len(drops)):
        drops[i] = max(drops[i], drops[i-1])
        
    max_drop = drops[-1]
    if max_drop <= 0: return pd.DataFrame()
    
    norm_x = [d / max_drop for d in drops]
    fixed_grid = np.linspace(0.0, 1.0, 100)
    interp_evals = np.interp(fixed_grid, norm_x, evals)
    
    return pd.DataFrame({"X_Axis": fixed_grid, "E_val": interp_evals, "Region": region_name})

def extract_aligned_stepwise_data(directories):
    noise_data = []
    ssim_data = []
    
    for model_name, base_dir in directories.items():
        print(f"Extracting JSON payload for model: {model_name}...")
        if not os.path.exists(base_dir): continue
        dataset_name = DATASET_MAPPING.get(model_name, "Unknown")
            
        for run_folder in os.listdir(base_dir):
            if run_folder != "run_1": continue
            run_path = os.path.join(base_dir, run_folder)
            
            for img_folder in os.listdir(run_path):
                img_path = os.path.join(run_path, img_folder)
                if not os.path.isdir(img_path): continue
                    
                json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
                if not os.path.exists(json_file): continue
                try: data = load_json(json_file)
                except Exception: continue
                
                if "in_context_analysis" not in data or "out_of_context_analysis" not in data: continue
                noise_types = set(data["in_context_analysis"].keys()).intersection(set(data["out_of_context_analysis"].keys()))

                for noise_type in noise_types:
                    ic_levels = data["in_context_analysis"][noise_type].get("level_data", [])
                    oc_levels = data["out_of_context_analysis"][noise_type].get("level_data", [])
                    if not ic_levels or not oc_levels: continue

                    max_steps = max(len(ic_levels), len(oc_levels))
                    IC_vals = [lvl.get("discounted_utility", 0.0) for lvl in ic_levels]
                    OC_vals = [lvl.get("discounted_utility", 0.0) for lvl in oc_levels]
                    while len(IC_vals) < max_steps: IC_vals.append(IC_vals[-1])
                    while len(OC_vals) < max_steps: OC_vals.append(OC_vals[-1])
                    
                    IC_0, OC_0 = max(1e-6, IC_vals[0]), max(1e-6, OC_vals[0])
                    N_ic, N_oc = len(IC_vals), len(OC_vals)
                    E_0_img = (IC_0 + OC_0) / 2.0

                    # 2. Independent IC Integration
                    W_ic, A_ic = 0.0, 0.0
                    for k in range(1, N_ic):
                        eta_k = k / (N_ic - 1) if N_ic > 1 else 0.0
                        w_k = math.exp(-GAMMA * eta_k)
                        ic_hat = IC_vals[k] / IC_0
                        
                        A_ic += w_k * ic_hat
                        W_ic += w_k
                        
                    norm_A_ic = A_ic / W_ic if W_ic > 0 else (IC_vals[0]/IC_0 if N_ic == 1 else 0.0)

                    # 3. Independent OC Integration (Area & Drift)
                    W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
                    for k in range(1, N_oc):
                        eta_k = k / (N_oc - 1) if N_oc > 1 else 0.0
                        w_k = math.exp(-GAMMA * eta_k)
                        oc_hat = OC_vals[k] / OC_0
                        
                        A_oc += w_k * oc_hat
                        weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
                        W_oc += w_k

                    # 4. Global Gap Calculation (Strictly Bounded to prevent hallucination loopholes)
                    norm_A_oc = A_oc / W_oc if W_oc > 0 else (OC_vals[0]/OC_0 if N_oc == 1 else 0.0)
                    
                    # CRITICAL FIX: Clamp components so artificial spikes above 1.0 cannot game the gap separation
                    bounded_oc = min(1.0, norm_A_oc)
                    bounded_ic = min(1.0, norm_A_ic)
                    G_gap = max(0.0, bounded_oc - bounded_ic)

                    E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0
                    discount_rate = math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift))


                    final_score = (E_0_img / 100.0) * discount_rate * 100.0

                    df_n_ic = process_noise_trajectory(ic_levels, 'IC')
                    df_n_oc = process_noise_trajectory(oc_levels, 'OC')
                    df_s_ic = process_ssim_lifecycle(ic_levels, 'IC')
                    df_s_oc = process_ssim_lifecycle(oc_levels, 'OC')
                    
                    if df_n_ic.empty or df_s_ic.empty: continue
                    
                    meta = {
                        'Dataset': dataset_name, 'Model': model_name, 'Image': img_folder, 
                        'Noise_Type': noise_type, 'E_0': E_0_img, 'S_Factor': discount_rate, 'Final_Score': final_score
                    }
                    
                    for df, store in [(pd.concat([df_n_ic, df_n_oc]), noise_data), (pd.concat([df_s_ic, df_s_oc]), ssim_data)]:
                        for k, v in meta.items(): df[k] = v
                        store.extend(df.to_dict('records'))
                            
    return pd.DataFrame(noise_data), pd.DataFrame(ssim_data)

# ==========================================
# 2. AGGREGATED "WHOLE MODEL" PLOTTING
# ==========================================
def plot_aggregated_model(ax, df_model, model_name, plot_type):
    if df_model.empty:
        ax.set_visible(False)
        return

    ic_color, oc_color = "dodgerblue", "darkorange"
    noise_types = df_model['Noise_Type'].unique()

    # 1. Plot individual noises as faint envelope lines
    for noise in noise_types:
        df_noise = df_model[df_model['Noise_Type'] == noise]
        df_ic = df_noise[df_noise['Region'] == 'IC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)
        df_oc = df_noise[df_noise['Region'] == 'OC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)
        
        if not df_ic.empty and not df_oc.empty:
            ax.plot(df_ic['X_Axis'], df_ic['E_val'], color=ic_color, lw=1.5, alpha=0.25)
            ax.plot(df_oc['X_Axis'], df_oc['E_val'], color=oc_color, lw=1.5, alpha=0.25)

    # 2. Plot the Global Mean Trajectory
    df_global_ic = df_model[df_model['Region'] == 'IC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)
    df_global_oc = df_model[df_model['Region'] == 'OC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)

    markevery = max(1, len(df_global_ic) // 10)
    ax.plot(df_global_ic['X_Axis'], df_global_ic['E_val'], color=ic_color, lw=4, marker='o', markersize=6, markevery=markevery, label='Global IC Mean')
    ax.plot(df_global_oc['X_Axis'], df_global_oc['E_val'], color=oc_color, lw=4, marker='o', markersize=6, markevery=markevery, label='Global OC Mean')

    # 3. Whole Model Metrics
    global_e0 = df_model['E_0'].mean()
    global_s_factor = df_model['S_Factor'].mean()
    global_final_score = df_model['Final_Score'].mean()

    if plot_type == "SSIM":
        ax.set_xlim(-0.02, 1.02)
        ax.axhline(y=global_e0, color='gray', linestyle='--', lw=2.5, alpha=0.8, label=f'Overall $E_0$ ({global_e0:.1f}%)')
        ax.axhline(y=global_final_score, color='black', linestyle='-', lw=4, label=f'Overall Discounted Score ({global_final_score:.1f}%)')
        title_str = f"{model_name} | Whole Model Aggregation\n$E_0$: {global_e0:.1f}% | Global S-Factor: {global_s_factor:.2f} | Final: {global_final_score:.1f}%"
    else:
        ax.set_xlim(-0.1, 3.1)
        title_str = f"{model_name} | Whole Model Aggregation"

    ax.set_ylim(-5, 105)
    ax.set_title(title_str, fontsize=17 if plot_type=="SSIM" else 20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

def generate_aggregated_model_plots(df_data, output_dir, plot_type):
    """Generates a single plot containing all perturbation types per model."""
    datasets = df_data['Dataset'].unique()
    for dataset in datasets:
        df_ds = df_data[df_data['Dataset'] == dataset]
        models = sorted(df_ds['Model'].unique())
        
        fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 6.5))
        if len(models) == 1: axes = [axes]
        
        for idx, model in enumerate(models):
            df_model = df_ds[df_ds['Model'] == model]
            plot_aggregated_model(axes[idx], df_model, model, plot_type)
            
        x_label = "Normalized SSIM Drop (0.0 to 1.0)" if plot_type == "SSIM" else "Absolute Noise (0.0 to 3.0)"
        fig.supxlabel(x_label, fontsize=24, fontweight='bold', y=-0.05)
        fig.supylabel("Normalized Utility ($E_{val}$) %", fontsize=24, fontweight='bold', x=0.08)
        
        legend_elements = [
            Line2D([0], [0], color="dodgerblue", lw=4, label='Global IC Trajectory'),
            Line2D([0], [0], color="darkorange", lw=4, label='Global OC Trajectory'),
            Line2D([0], [0], color="gray", lw=2, linestyle='--', label='Initial Target Utility ($E_0$)'),
            Line2D([0], [0], color="black", lw=4, linestyle='-', label='Final Discounted Utility')
        ] if plot_type == "SSIM" else [
            Line2D([0], [0], color="dodgerblue", lw=4, label='Global IC Trajectory'),
            Line2D([0], [0], color="darkorange", lw=4, label='Global OC Trajectory')
        ]
        
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(legend_elements), fontsize=16, frameon=False)
        plt.suptitle(f"{dataset} | Overall Model Robustness Envelope ({plot_type} Scale)", fontsize=30, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{dataset}_Overall_Model_Envelope_{plot_type}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved {dataset} Whole-Model Aggregation ({plot_type})")

# ==========================================
# 3. COMPREHENSIVE GRID PLOTTING (Unchanged)
# ==========================================
def plot_single_subplot(ax, df_sub, model_name, noise_type, plot_type):
    if df_sub.empty:
        ax.set_visible(False)
        return

    df_ic = df_sub[df_sub['Region'] == 'IC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)
    df_oc = df_sub[df_sub['Region'] == 'OC'].groupby('X_Axis', as_index=False).mean(numeric_only=True)
    
    if df_ic.empty or df_oc.empty:
        ax.set_visible(False)
        return

    e0_mean = df_sub['E_0'].mean()
    s_factor_mean = df_sub['S_Factor'].mean()
    final_score_mean = df_sub['Final_Score'].mean()

    ic_color, oc_color = "dodgerblue", "darkorange"
    markevery = max(1, len(df_ic) // 10)

    ax.plot(df_ic['X_Axis'], df_ic['E_val'], color=ic_color, lw=3, marker='o', markersize=5, markevery=markevery)
    ax.plot(df_oc['X_Axis'], df_oc['E_val'], color=oc_color, lw=3, marker='o', markersize=5, markevery=markevery)
    ax.fill_between(df_ic['X_Axis'], df_ic['E_val'], 100.0, color=ic_color, alpha=0.08)
    ax.fill_between(df_oc['X_Axis'], 0.0, df_oc['E_val'], color=oc_color, alpha=0.08)

    if plot_type == "SSIM":
        ax.set_xlim(-0.02, 1.02)
        ax.axhline(y=e0_mean, color='gray', linestyle='--', lw=2)
        ax.axhline(y=final_score_mean, color='black', linestyle='-', lw=4)
        title_extra = f"\n$E_0$: {e0_mean:.1f}% | S-Factor: {s_factor_mean:.2f} | Final: {final_score_mean:.1f}%"
    else:
        ax.set_xlim(-0.1, 3.1)
        title_extra = ""

    ax.set_ylim(-5, 105)
    ax.set_title(f"{model_name} | {noise_type.capitalize()}{title_extra}", fontsize=18 if plot_type=="SSIM" else 22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

def generate_comprehensive_grid(df_data, dataset_name, output_dir, plot_type):
    models = sorted(df_data['Model'].unique())
    noise_types = sorted(df_data['Noise_Type'].unique())
    
    num_models = len(models)
    noises_per_row = 3
    num_rows = math.ceil(len(noise_types) / noises_per_row)
    num_cols = noises_per_row * num_models
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5.5, num_rows * 6))
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)
    if num_cols == 1: axes = np.expand_dims(axes, axis=1)

    for r in range(num_rows):
        for c_noise in range(noises_per_row):
            noise_idx = r * noises_per_row + c_noise
            if noise_idx >= len(noise_types):
                for m_idx in range(num_models): axes[r, c_noise * num_models + m_idx].set_visible(False)
                continue
                
            noise = noise_types[noise_idx]
            for m_idx, model in enumerate(models):
                ax = axes[r, c_noise * num_models + m_idx]
                df_sub = df_data[(df_data['Model'] == model) & (df_data['Noise_Type'] == noise)]
                plot_single_subplot(ax, df_sub, model, noise, plot_type)

    x_label = "Normalized SSIM Drop (0.0 to 1.0)" if plot_type == "SSIM" else "Absolute Noise (0.0 to 3.0)"
    fig.supxlabel(x_label, fontsize=32, fontweight='bold', y=-0.02)
    fig.supylabel("Normalized Utility ($E_{val}$) %", fontsize=32, fontweight='bold', x=0.02)
    plt.suptitle(f"{dataset_name} | Comprehensive {plot_type} Profiles", fontsize=38, fontweight='bold', y=1.05)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.35, hspace=0.45, bottom=0.08, left=0.06, top=0.90)
    
    save_path = os.path.join(output_dir, f"{dataset_name}_Comprehensive_{plot_type}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Comprehensive {dataset_name} Grid ({plot_type})")

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

    # Create directory structures
    base_out = './evaluation_plots_noise'
    dirs = {
        'noise_per': os.path.join(base_out, 'per_perturbation', 'noise_based'),
        'ssim_per': os.path.join(base_out, 'per_perturbation', 'ssim_based'),
        'comp': os.path.join(base_out, 'comprehensive')
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    print("Extracting Data Arrays...")
    df_noise, df_ssim = extract_aligned_stepwise_data(args.directories)
    
    if df_noise.empty or df_ssim.empty:
        print("Data extraction failed. Check paths.")
        return

    sns.set_theme(style="whitegrid", context="paper")

    print("\n--- Generating Overall Model Aggregated Plots ---")
    generate_aggregated_model_plots(df_noise, dirs['noise_per'], "Noise")
    generate_aggregated_model_plots(df_ssim, dirs['ssim_per'], "SSIM")

    print("\n--- Generating Comprehensive Grids ---")
    for dataset in df_ssim['Dataset'].unique():
        generate_comprehensive_grid(df_noise[df_noise['Dataset'] == dataset], dataset, dirs['comp'], "Noise")
        generate_comprehensive_grid(df_ssim[df_ssim['Dataset'] == dataset], dataset, dirs['comp'], "SSIM")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
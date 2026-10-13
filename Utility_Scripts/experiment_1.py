import os
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ==========================================
# CONFIGURATION
# ==========================================
MODELS = {
    "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out",
    "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out"
}

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def get_shared_images(models_dict):
    """Finds images present in run_1 for all specified models."""
    image_sets = []
    for name, path in models_dict.items():
        run_path = os.path.join(path, "run_1")
        if not os.path.exists(run_path):
            print(f"Directory missing for {name}: {run_path}")
            return []
        
        imgs = set()
        for img in os.listdir(run_path):
            json_file = os.path.join(run_path, img, f"{img}_analysis.json")
            if os.path.exists(json_file):
                imgs.add(img)
        image_sets.append(imgs)
        
    return list(set.intersection(*image_sets))

def get_max_shared_drop(models_dict, target_image):
    """
    Finds the maximum SSIM drop that is shared across ALL noise types 
    for BOTH models and BOTH regions (IC & OC) on this specific image.
    """
    min_ssims = []
    
    for name, path in models_dict.items():
        json_file = os.path.join(path, "run_1", target_image, f"{target_image}_analysis.json")
        data = load_json(json_file)
        
        # Check IC SSIMs
        for noise, noise_data in data.get("in_context_analysis", {}).items():
            levels = noise_data.get("level_data", [])
            if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
            
        # Check OC SSIMs
        for noise, noise_data in data.get("out_of_context_analysis", {}).items():
            levels = noise_data.get("level_data", [])
            if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
            
    # The highest of these minimums is the strict "shared floor" reached by EVERY trajectory
    shared_min_ssim = max(min_ssims)
    max_shared_drop = 1.0 - shared_min_ssim
    return max_shared_drop, shared_min_ssim

def main():
    print("Finding a shared image for the experiment...")
    shared_images = get_shared_images(MODELS)
    if not shared_images:
        print("No shared images found between the specified models.")
        return
        
    target_image = random.choice(shared_images)
    print(f"Selected Image: {target_image}")
    
    max_shared_drop, shared_min_ssim = get_max_shared_drop(MODELS, target_image)
    print(f"Shared Minimum SSIM: {shared_min_ssim:.4f}")
    print(f"Maximum Shared Drop: {max_shared_drop:.4f} (This will map to 1.0 on X-axis)")

    # 100 uniformly spaced points from 0.0 (No Drop) to 1.0 (Max Shared Drop)
    norm_x = np.linspace(0.0, 1.0, 100)
    
    sns.set_theme(style="whitegrid", context="paper")
    
    sample_data = load_json(os.path.join(MODELS["D-LKA"], "run_1", target_image, f"{target_image}_analysis.json"))
    noise_types = sorted(list(sample_data.get("in_context_analysis", {}).keys()))
    
    num_models = len(MODELS)
    num_noises = len(noise_types)
    
    fig, axes = plt.subplots(num_models, num_noises, figsize=(5 * num_noises, 6 * num_models))
    if num_models == 1: axes = np.expand_dims(axes, axis=0)
    if num_noises == 1: axes = np.expand_dims(axes, axis=1)

    for row_idx, (model_name, model_path) in enumerate(MODELS.items()):
        json_file = os.path.join(model_path, "run_1", target_image, f"{target_image}_analysis.json")
        data = load_json(json_file)
        
        ic_data = data.get("in_context_analysis", {})
        oc_data = data.get("out_of_context_analysis", {})
        
        for col_idx, noise in enumerate(noise_types):
            ax = axes[row_idx, col_idx]
            
            ic_levels = ic_data.get(noise, {}).get("level_data", [])
            oc_levels = oc_data.get(noise, {}).get("level_data", [])
            
            if not ic_levels or not oc_levels:
                ax.set_visible(False)
                continue
                
            # Extract independent SSIM trajectories
            ssims_ic = [float(lvl.get("ssim", 1.0)) for lvl in ic_levels]
            ssims_oc = [float(lvl.get("ssim", 1.0)) for lvl in oc_levels]
            
            # Extract empirical utility metrics
            ic_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in ic_levels]
            oc_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in oc_levels]
            
            # Map empirical SSIMs to Normalized Drop independent of each other
            emp_drops_ic = [(1.0 - s) / max_shared_drop for s in ssims_ic]
            emp_drops_oc = [(1.0 - s) / max_shared_drop for s in ssims_oc]
            
            # Force monotonicity to prevent numpy interpolation crashes
            for i in range(1, len(emp_drops_ic)):
                if emp_drops_ic[i] <= emp_drops_ic[i-1]: emp_drops_ic[i] = emp_drops_ic[i-1] + 1e-6
            for i in range(1, len(emp_drops_oc)):
                if emp_drops_oc[i] <= emp_drops_oc[i-1]: emp_drops_oc[i] = emp_drops_oc[i-1] + 1e-6
                    
            # Interpolate both independent arrays onto the shared uniform axis
            interp_ic = np.interp(norm_x, emp_drops_ic, ic_utils)
            interp_oc = np.interp(norm_x, emp_drops_oc, oc_utils)
            
            # Area calculations (Trapezoidal Rule)
            auc_ic = np.trapz(interp_ic, norm_x)
            auc_oc = np.trapz(interp_oc, norm_x)
            
            # Final Vulnerability Score (0 = Perfect, 1 = Total Collapse)
            final_score = (auc_oc + (1.0 - auc_ic)) / 2.0
            
            # Plot the trajectories
            ax.plot(norm_x, interp_ic, color='dodgerblue', lw=3, label="IC Utility")
            ax.plot(norm_x, interp_oc, color='darkorange', lw=3, label="OC Utility")
            
            # Visual area fills
            ax.fill_between(norm_x, interp_ic, 1.0, color='dodgerblue', alpha=0.1) # (1 - AUC_IC) Area
            ax.fill_between(norm_x, 0, interp_oc, color='darkorange', alpha=0.1)   # AUC_OC Area
            
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.05)
            
            ax.set_title(f"{model_name} | {noise.capitalize()}\nFinal. Score: {final_score:.3f}", 
                         fontsize=16, fontweight='bold')
            ax.tick_params(labelsize=12)

    fig.supxlabel("Normalized SSIM Drop ($0 = No Drop$, $1 = Max Shared Drop$)", fontsize=20, fontweight='bold')
    fig.supylabel("Normalized Expected Utility ($E_k$)", fontsize=20, fontweight='bold')
    
    plt.suptitle(f"Experimental Vulnerability AUC Integration\nImage: {target_image}", fontsize=24, fontweight='bold', y=1.05)
    
    legend_elements = [
        Line2D([0], [0], color='dodgerblue', lw=3, label='IC Utility Curve'),
        Line2D([0], [0], color='darkorange', lw=3, label='OC Utility Curve'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='dodgerblue', alpha=0.3, markersize=15, label='Area: (1 - AUC_IC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', alpha=0.3, markersize=15, label='Area: AUC_OC')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=16, frameon=False)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.85)
    
    save_path = "./Experimental_AUC_Vulnerability.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Success! Saved experimental plot to {save_path}")

if __name__ == "__main__":
    main()



# import os
# import json
# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D

# # ==========================================
# # CONFIGURATION
# # ==========================================
# MODELS = {
#     "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out",
#     "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out"
# }

# GAMMA = 1.0 # Decay weight for the trend integrals

# def load_json(file_path):
#     with open(file_path, "r") as file:
#         return json.load(file)

# def get_shared_images(models_dict):
#     """Finds images present in run_1 for all specified models."""
#     image_sets = []
#     for name, path in models_dict.items():
#         run_path = os.path.join(path, "run_1")
#         if not os.path.exists(run_path):
#             print(f"Directory missing for {name}: {run_path}")
#             return []
        
#         imgs = set()
#         for img in os.listdir(run_path):
#             json_file = os.path.join(run_path, img, f"{img}_analysis.json")
#             if os.path.exists(json_file):
#                 imgs.add(img)
#         image_sets.append(imgs)
        
#     return list(set.intersection(*image_sets))

# def get_max_shared_drop(models_dict, target_image):
#     """
#     Finds the maximum SSIM drop that is shared across ALL noise types 
#     for BOTH models and BOTH regions (IC & OC) on this specific image.
#     """
#     min_ssims = []
    
#     for name, path in models_dict.items():
#         json_file = os.path.join(path, "run_1", target_image, f"{target_image}_analysis.json")
#         data = load_json(json_file)
        
#         for noise, noise_data in data.get("in_context_analysis", {}).items():
#             levels = noise_data.get("level_data", [])
#             if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
            
#         for noise, noise_data in data.get("out_of_context_analysis", {}).items():
#             levels = noise_data.get("level_data", [])
#             if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
            
#     shared_min_ssim = max(min_ssims)
#     max_shared_drop = 1.0 - shared_min_ssim
#     return max_shared_drop, shared_min_ssim

# def main():
#     print("Finding a shared image for the experiment...")
#     shared_images = get_shared_images(MODELS)
#     if not shared_images:
#         print("No shared images found between the specified models.")
#         return
        
#     target_image = random.choice(shared_images)
#     print(f"Selected Image: {target_image}")
    
#     max_shared_drop, shared_min_ssim = get_max_shared_drop(MODELS, target_image)
#     print(f"Shared Minimum SSIM: {shared_min_ssim:.4f}")
#     print(f"Maximum Shared Drop: {max_shared_drop:.4f} (Mapped to 1.0 on X-axis)")

#     # 100 uniformly spaced points from 0.0 (No Drop) to 1.0 (Max Shared Drop)
#     norm_x = np.linspace(0.0, 1.0, 100)
    
#     sns.set_theme(style="whitegrid", context="paper")
    
#     sample_data = load_json(os.path.join(MODELS["D-LKA"], "run_1", target_image, f"{target_image}_analysis.json"))
#     noise_types = sorted(list(sample_data.get("in_context_analysis", {}).keys()))
    
#     num_models = len(MODELS)
#     num_noises = len(noise_types)
    
#     fig, axes = plt.subplots(num_models, num_noises, figsize=(5.5 * num_noises, 6 * num_models))
#     if num_models == 1: axes = np.expand_dims(axes, axis=0)
#     if num_noises == 1: axes = np.expand_dims(axes, axis=1)

#     for row_idx, (model_name, model_path) in enumerate(MODELS.items()):
#         json_file = os.path.join(model_path, "run_1", target_image, f"{target_image}_analysis.json")
#         data = load_json(json_file)
        
#         ic_data = data.get("in_context_analysis", {})
#         oc_data = data.get("out_of_context_analysis", {})
        
#         for col_idx, noise in enumerate(noise_types):
#             ax = axes[row_idx, col_idx]
            
#             ic_levels = ic_data.get(noise, {}).get("level_data", [])
#             oc_levels = oc_data.get(noise, {}).get("level_data", [])
            
#             if not ic_levels or not oc_levels:
#                 ax.set_visible(False)
#                 continue
                
#             # ========================================================
#             # PART A: CALCULATE TRAJECTORY TRENDS (G & E_drift)
#             # ========================================================
#             ic_utils_raw = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in ic_levels]
#             oc_utils_raw = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in oc_levels]
            
#             max_steps = max(len(ic_utils_raw), len(oc_utils_raw))
#             while len(ic_utils_raw) < max_steps: ic_utils_raw.append(ic_utils_raw[-1])
#             while len(oc_utils_raw) < max_steps: oc_utils_raw.append(oc_utils_raw[-1])

#             ic_0, oc_0 = max(1e-6, ic_utils_raw[0]), max(1e-6, oc_utils_raw[0])

#             W, weighted_sep, weighted_drift = 0.0, 0.0, 0.0
#             for k in range(1, max_steps):
#                 eta_k = k / (max_steps - 1) if max_steps > 1 else 0
#                 w_k = math.exp(-GAMMA * eta_k)
#                 ic_hat = ic_utils_raw[k] / ic_0
#                 oc_hat = oc_utils_raw[k] / oc_0
#                 weighted_sep += w_k * max(0.0, oc_hat - ic_hat)
#                 weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
#                 W += w_k
                
#             G_ic_oc = weighted_sep / W if W > 0 else 0.0
#             E_drift = math.sqrt(weighted_drift / W) if W > 0 else 0.0
#             G_deficit = max(0.0, 1.0 - G_ic_oc)

#             # ========================================================
#             # PART B: CALCULATE NORMALIZED AUC VOLUME
#             # ========================================================
#             ssims_ic = [float(lvl.get("ssim", 1.0)) for lvl in ic_levels]
#             ssims_oc = [float(lvl.get("ssim", 1.0)) for lvl in oc_levels]
            
#             emp_drops_ic = [(1.0 - s) / max_shared_drop for s in ssims_ic]
#             emp_drops_oc = [(1.0 - s) / max_shared_drop for s in ssims_oc]
            
#             for i in range(1, len(emp_drops_ic)):
#                 if emp_drops_ic[i] <= emp_drops_ic[i-1]: emp_drops_ic[i] = emp_drops_ic[i-1] + 1e-6
#             for i in range(1, len(emp_drops_oc)):
#                 if emp_drops_oc[i] <= emp_drops_oc[i-1]: emp_drops_oc[i] = emp_drops_oc[i-1] + 1e-6
                    
#             interp_ic = np.interp(norm_x, emp_drops_ic, ic_utils_raw[:len(emp_drops_ic)])
#             interp_oc = np.interp(norm_x, emp_drops_oc, oc_utils_raw[:len(emp_drops_oc)])
            
#             auc_ic = np.trapz(interp_ic, norm_x)
#             auc_oc = np.trapz(interp_oc, norm_x)
            
#             # The base 2D Volume constraints
#             vuln_score = (auc_oc + (1.0 - auc_ic)) / 2.0
#             A_volume = 1.0 - vuln_score # Area constraint (0 to 1)

#             # ========================================================
#             # PART C: FUSE INTO 3D FORMULATIONS
#             # ========================================================
#             # 1. 3D Geometric Mean (Cube Root)
#             val_geo = A_volume * max(0.0, G_ic_oc) * max(0.0, 1.0 - E_drift)
#             S_geo_3d = (val_geo ** (1.0 / 3.0)) * 100.0
            
#             # 2. 3D Euclidean Distance
#             dist_sq = ((1.0 - A_volume)**2 + G_deficit**2 + min(1.0, E_drift)**2) / 3.0
#             S_euc_3d = (1.0 - math.sqrt(dist_sq)) * 100.0

#             # ========================================================
#             # PLOTTING
#             # ========================================================
#             ax.plot(norm_x, interp_ic, color='dodgerblue', lw=3, label="IC Utility")
#             ax.plot(norm_x, interp_oc, color='darkorange', lw=3, label="OC Utility")
            
#             ax.fill_between(norm_x, interp_ic, 1.0, color='dodgerblue', alpha=0.1) # (1 - AUC_IC) Area
#             ax.fill_between(norm_x, 0, interp_oc, color='darkorange', alpha=0.1)   # AUC_OC Area
            
#             ax.set_xlim(0, 1.0)
#             ax.set_ylim(0, 1.05)
            
#             # Multi-line title revealing the fusion
#             title_text = (f"{model_name} | {noise.capitalize()}\n"
#                           f"Vuln(AUC): {vuln_score:.3f} | Area($A$): {A_volume:.3f}\n"
#                           f"3D Geo: {S_geo_3d:.1f} | 3D Euc: {S_euc_3d:.1f}")
                          
#             ax.set_title(title_text, fontsize=14, fontweight='bold')
#             ax.tick_params(labelsize=12)

#     fig.supxlabel("Normalized SSIM Drop ($0 = No Drop$, $1 = Max Shared Drop$)", fontsize=20, fontweight='bold')
#     fig.supylabel("Normalized Expected Utility ($E_k$)", fontsize=20, fontweight='bold')
    
#     plt.suptitle(f"3D Tensor Robustness (Volume + Trend Fusion)\nImage: {target_image}", fontsize=24, fontweight='bold', y=1.05)
    
#     legend_elements = [
#         Line2D([0], [0], color='dodgerblue', lw=3, label='IC Utility Curve'),
#         Line2D([0], [0], color='darkorange', lw=3, label='OC Utility Curve'),
#         Line2D([0], [0], marker='s', color='w', markerfacecolor='dodgerblue', alpha=0.3, markersize=15, label='Area: (1 - AUC_IC)'),
#         Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', alpha=0.3, markersize=15, label='Area: AUC_OC')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=16, frameon=False)
    
#     plt.tight_layout(pad=3.0)
#     plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.82)
    
#     save_path = "./Experimental_3D_AUC_Integration.pdf"
#     plt.savefig(save_path, format="pdf", bbox_inches="tight")
#     print(f"Success! Saved experimental plot to {save_path}")

# if __name__ == "__main__":
#     main()



# import os
# import json
# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D

# # ==========================================
# # CONFIGURATION
# # ==========================================
# MODELS = {
#     "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/prism_out",
#     "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/prism_out"
# }

# GAMMA = 1.0      # Decay weight for early-stage priority
# LAMBDA_P = 1.0   # Penalty strictness for the Exponential formulation

# def load_json(file_path):
#     with open(file_path, "r") as file:
#         return json.load(file)

# def get_shared_images(models_dict):
#     """Finds images present in run_1 for all specified models."""
#     image_sets = []
#     for name, path in models_dict.items():
#         run_path = os.path.join(path, "run_1")
#         if not os.path.exists(run_path): return []
#         imgs = set([img for img in os.listdir(run_path) if os.path.exists(os.path.join(run_path, img, f"{img}_analysis.json"))])
#         image_sets.append(imgs)
#     return list(set.intersection(*image_sets))

# def get_max_shared_drop(models_dict, target_image):
#     """Finds the maximum shared SSIM drop across both models and all noises."""
#     min_ssims = []
#     for name, path in models_dict.items():
#         data = load_json(os.path.join(path, "run_1", target_image, f"{target_image}_analysis.json"))
#         for region in ["in_context_analysis", "out_of_context_analysis"]:
#             for noise, noise_data in data.get(region, {}).items():
#                 levels = noise_data.get("level_data", [])
#                 if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
#     shared_min_ssim = max(min_ssims)
#     return 1.0 - shared_min_ssim, shared_min_ssim

# def main():
#     print("Finding a shared image for the experiment...")
#     shared_images = get_shared_images(MODELS)
#     if not shared_images:
#         print("No shared images found.")
#         return
        
#     target_image = random.choice(shared_images)
#     print(f"Selected Image: {target_image}")
    
#     max_shared_drop, shared_min_ssim = get_max_shared_drop(MODELS, target_image)
#     norm_x = np.linspace(0.0, 1.0, 100)
    
#     sns.set_theme(style="whitegrid", context="paper")
#     sample_data = load_json(os.path.join(MODELS["D-LKA"], "run_1", target_image, f"{target_image}_analysis.json"))
#     noise_types = sorted(list(sample_data.get("in_context_analysis", {}).keys()))
    
#     fig, axes = plt.subplots(len(MODELS), len(noise_types), figsize=(5.5 * len(noise_types), 6 * len(MODELS)))
#     if len(MODELS) == 1: axes = np.expand_dims(axes, axis=0)
#     if len(noise_types) == 1: axes = np.expand_dims(axes, axis=1)

#     for row_idx, (model_name, model_path) in enumerate(MODELS.items()):
#         data = load_json(os.path.join(model_path, "run_1", target_image, f"{target_image}_analysis.json"))
        
#         for col_idx, noise in enumerate(noise_types):
#             ax = axes[row_idx, col_idx]
#             ic_levels = data.get("in_context_analysis", {}).get(noise, {}).get("level_data", [])
#             oc_levels = data.get("out_of_context_analysis", {}).get(noise, {}).get("level_data", [])
            
#             if not ic_levels or not oc_levels:
#                 ax.set_visible(False)
#                 continue
                
#             # --- 1. EXTRACT RAW UTILITY ---
#             ic_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in ic_levels]
#             oc_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in oc_levels]
            
#             max_steps = max(len(ic_utils), len(oc_utils))
#             while len(ic_utils) < max_steps: ic_utils.append(ic_utils[-1])
#             while len(oc_utils) < max_steps: oc_utils.append(oc_utils[-1])

#             ic_0, oc_0 = max(1e-6, ic_utils[0]), max(1e-6, oc_utils[0])

#             # --- 2. CALCULATE TRENDS (G and E_drift) ---
#             W, weighted_sep, weighted_drift = 0.0, 0.0, 0.0
#             for k in range(1, max_steps):
#                 eta_k = k / (max_steps - 1) if max_steps > 1 else 0
#                 w_k = math.exp(-GAMMA * eta_k)
#                 ic_hat = ic_utils[k] / ic_0
#                 oc_hat = oc_utils[k] / oc_0
                
#                 weighted_sep += w_k * max(0.0, oc_hat - ic_hat)
#                 weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
#                 W += w_k
                
#             G_gap = weighted_sep / W if W > 0 else 0.0
#             E_drift = math.sqrt(weighted_drift / W) if W > 0 else 0.0
#             S_stab = max(0.0, 1.0 - E_drift)

#             # --- 3. CALCULATE VOLUME (AUC) ---
#             ssims_ic = [float(lvl.get("ssim", 1.0)) for lvl in ic_levels]
#             ssims_oc = [float(lvl.get("ssim", 1.0)) for lvl in oc_levels]
            
#             emp_drops_ic = [(1.0 - s) / max_shared_drop for s in ssims_ic]
#             emp_drops_oc = [(1.0 - s) / max_shared_drop for s in ssims_oc]
            
#             for i in range(1, len(emp_drops_ic)): 
#                 if emp_drops_ic[i] <= emp_drops_ic[i-1]: emp_drops_ic[i] = emp_drops_ic[i-1] + 1e-6
#             for i in range(1, len(emp_drops_oc)): 
#                 if emp_drops_oc[i] <= emp_drops_oc[i-1]: emp_drops_oc[i] = emp_drops_oc[i-1] + 1e-6
                    
#             interp_ic = np.interp(norm_x, emp_drops_ic, ic_utils[:len(emp_drops_ic)])
#             interp_oc = np.interp(norm_x, emp_drops_oc, oc_utils[:len(emp_drops_oc)])
            
#             auc_ic = np.trapz(interp_ic, norm_x)
#             auc_oc = np.trapz(interp_oc, norm_x)
            
#             V_vol = (auc_oc + (1.0 - auc_ic)) / 2.0
#             V_vol = max(0.0, min(1.0, V_vol))

#             # --- 4. THE TWO HOLISTIC COMPARISONS ---
#             # A: Parameter-Free Geometric
#             H_geo = V_vol * math.sqrt(max(0.0, G_gap) * S_stab) * 100.0
            
#             # B: Exponential Gate
#             G_deficit = max(0.0, 1.0 - G_gap)
#             H_exp = V_vol * math.exp(-LAMBDA_P * (G_deficit + E_drift)) * 100.0

#             # --- PLOTTING ---
#             ax.plot(norm_x, interp_ic, color='dodgerblue', lw=3)
#             ax.plot(norm_x, interp_oc, color='darkorange', lw=3)
#             ax.fill_between(norm_x, interp_ic, 1.0, color='dodgerblue', alpha=0.1)
#             ax.fill_between(norm_x, 0, interp_oc, color='darkorange', alpha=0.1)
#             ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
            
#             title_text = (f"{model_name} | {noise.capitalize()}\n"
#                           f"Base Vol ($V$): {V_vol:.2f} | Gap ($G$): {G_gap:.2f} | Stab ($S$): {S_stab:.2f}\n"
#                           f"$\mathcal{{H}}_{{geo}}$ (Strict): {H_geo:.1f} | $\mathcal{{H}}_{{exp}}$ (Soft): {H_exp:.1f}")
#             ax.set_title(title_text, fontsize=13, fontweight='bold')
#             ax.tick_params(labelsize=11)

#     fig.supxlabel("Normalized SSIM Drop", fontsize=18, fontweight='bold')
#     fig.supylabel("Normalized Expected Utility ($E_k$)", fontsize=18, fontweight='bold')
#     plt.suptitle(f"Holistic Robustness Index Comparison\nImage: {target_image}", fontsize=22, fontweight='bold', y=1.05)
    
#     legend_elements = [
#         Line2D([0], [0], color='dodgerblue', lw=3, label='IC Utility (Want Low)'),
#         Line2D([0], [0], color='darkorange', lw=3, label='OC Utility (Want High)')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=14, frameon=False)
    
#     plt.tight_layout(pad=3.0)
#     plt.subplots_adjust(wspace=0.3, hspace=0.45, top=0.82)
#     plt.savefig("./Holistic_Comparison_Single.pdf", format="pdf", bbox_inches="tight")
#     print("Success! Saved comparison plot to ./Holistic_Comparison_Single.pdf")

# if __name__ == "__main__":
#     main()




# import os
# import json
# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D

# # ==========================================
# # CONFIGURATION
# # ==========================================
# MODELS = {
#     "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults_50",
#     "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults_50"
# }

# GAMMA = 1.0      # Decay weight for early-stage priority

# def load_json(file_path):
#     with open(file_path, "r") as file:
#         return json.load(file)

# def get_shared_images(models_dict):
#     """Finds images present in run_1 for all specified models."""
#     image_sets = []
#     for name, path in models_dict.items():
#         run_path = os.path.join(path, "run_1")
#         if not os.path.exists(run_path): return []
#         imgs = set([img for img in os.listdir(run_path) if os.path.exists(os.path.join(run_path, img, f"{img}_analysis.json"))])
#         image_sets.append(imgs)
#     return list(set.intersection(*image_sets))

# def get_max_shared_drop(models_dict, target_image):
#     """Finds the maximum shared SSIM drop across both models and all noises."""
#     min_ssims = []
#     for name, path in models_dict.items():
#         data = load_json(os.path.join(path, "run_1", target_image, f"{target_image}_analysis.json"))
#         for region in ["in_context_analysis", "out_of_context_analysis"]:
#             for noise, noise_data in data.get(region, {}).items():
#                 levels = noise_data.get("level_data", [])
#                 if levels: min_ssims.append(min([float(lvl.get("ssim", 1.0)) for lvl in levels]))
#     shared_min_ssim = max(min_ssims) if min_ssims else 1.0
#     return 1.0 - shared_min_ssim, shared_min_ssim

# def main():
#     print("Finding a shared image for the experiment...")
#     shared_images = get_shared_images(MODELS)
#     if not shared_images:
#         print("No shared images found.")
#         return
        
#     target_image = random.choice(shared_images)
#     print(f"Selected Image: {target_image}")
    
#     max_shared_drop, shared_min_ssim = get_max_shared_drop(MODELS, target_image)
#     if max_shared_drop <= 0: max_shared_drop = 1.0 # Fallback 
#     norm_x = np.linspace(0.0, 1.0, 100)
    
#     sns.set_theme(style="whitegrid", context="paper")
#     sample_data = load_json(os.path.join(MODELS["D-LKA"], "run_1", target_image, f"{target_image}_analysis.json"))
#     noise_types = sorted(list(sample_data.get("in_context_analysis", {}).keys()))
    
#     fig, axes = plt.subplots(len(MODELS), len(noise_types), figsize=(5.5 * len(noise_types), 6 * len(MODELS)))
#     if len(MODELS) == 1: axes = np.expand_dims(axes, axis=0)
#     if len(noise_types) == 1: axes = np.expand_dims(axes, axis=1)

#     for row_idx, (model_name, model_path) in enumerate(MODELS.items()):
#         data = load_json(os.path.join(model_path, "run_1", target_image, f"{target_image}_analysis.json"))
        
#         for col_idx, noise in enumerate(noise_types):
#             ax = axes[row_idx, col_idx]
#             ic_levels = data.get("in_context_analysis", {}).get(noise, {}).get("level_data", [])
#             oc_levels = data.get("out_of_context_analysis", {}).get(noise, {}).get("level_data", [])
            
#             if not ic_levels or not oc_levels:
#                 ax.set_visible(False)
#                 continue
                
#             # --- 1. EXTRACT RAW UTILITY ---
#             ic_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in ic_levels]
#             oc_utils = [float(lvl.get("discounted_utility", 0.0)) / 100.0 for lvl in oc_levels]
            
#             max_steps = max(len(ic_utils), len(oc_utils))
#             while len(ic_utils) < max_steps: ic_utils.append(ic_utils[-1])
#             while len(oc_utils) < max_steps: oc_utils.append(oc_utils[-1])

#             ic_0, oc_0 = max(1e-6, ic_utils[0]), max(1e-6, oc_utils[0])

#             # --- 2. CALCULATE TRENDS (G and S) ---
#             W, weighted_sep, weighted_drift = 0.0, 0.0, 0.0
#             for k in range(1, max_steps):
#                 eta_k = k / (max_steps - 1) if max_steps > 1 else 0
#                 w_k = math.exp(-GAMMA * eta_k)
#                 ic_hat = ic_utils[k] / ic_0
#                 oc_hat = oc_utils[k] / oc_0
                
#                 weighted_sep += w_k * max(0.0, oc_hat - ic_hat)
#                 weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
#                 W += w_k
                
#             G_raw = weighted_sep / W if W > 0 else 0.0
#             E_drift = math.sqrt(weighted_drift / W) if W > 0 else 0.0
#             S_raw = max(0.0, 1.0 - E_drift)

#             # --- 3. CALCULATE VOLUME (V) ---
#             ssims_ic = [float(lvl.get("ssim", 1.0)) for lvl in ic_levels]
#             ssims_oc = [float(lvl.get("ssim", 1.0)) for lvl in oc_levels]
            
#             emp_drops_ic = [(1.0 - s) / max_shared_drop for s in ssims_ic]
#             emp_drops_oc = [(1.0 - s) / max_shared_drop for s in ssims_oc]
            
#             for i in range(1, len(emp_drops_ic)): 
#                 if emp_drops_ic[i] <= emp_drops_ic[i-1]: emp_drops_ic[i] = emp_drops_ic[i-1] + 1e-6
#             for i in range(1, len(emp_drops_oc)): 
#                 if emp_drops_oc[i] <= emp_drops_oc[i-1]: emp_drops_oc[i] = emp_drops_oc[i-1] + 1e-6
                    
#             interp_ic = np.interp(norm_x, emp_drops_ic, ic_utils[:len(emp_drops_ic)])
#             interp_oc = np.interp(norm_x, emp_drops_oc, oc_utils[:len(emp_drops_oc)])
            
#             auc_ic = np.trapz(interp_ic, norm_x)
#             auc_oc = np.trapz(interp_oc, norm_x)
            
#             V_raw = (auc_oc + (1.0 - auc_ic)) / 2.0

#             # Normalize and clamp the 3 variables safely between [1e-6, 1.0] to prevent div-by-zero
#             V = max(1e-6, min(1.0, V_raw))
#             G = max(1e-6, min(1.0, G_raw))
#             S = max(1e-6, min(1.0, S_raw))

#             # --- 4. MULTI-CRITERIA DECISION FUSION ---
#             # A: Harmonic Mean
#             harmonic_score = (3.0 / ((1.0/V) + (1.0/G) + (1.0/S))) * 100.0
            
#             # B: TOPSIS
#             d_plus = math.sqrt((1.0 - V)**2 + (1.0 - G)**2 + (1.0 - S)**2)
#             d_minus = math.sqrt(V**2 + G**2 + S**2)
#             topsis_score = (d_minus / (d_plus + d_minus)) * 100.0 if (d_plus + d_minus) > 0 else 0.0

#             # --- PLOTTING ---
#             ax.plot(norm_x, interp_ic, color='dodgerblue', lw=3)
#             ax.plot(norm_x, interp_oc, color='darkorange', lw=3)
#             ax.fill_between(norm_x, interp_ic, 1.0, color='dodgerblue', alpha=0.1)
#             ax.fill_between(norm_x, 0, interp_oc, color='darkorange', alpha=0.1)
#             ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
            
#             title_text = (f"{model_name} | {noise.capitalize()}\n"
#                           f"Vol ($V$): {V:.2f} | Gap ($G$): {G:.2f} | Stab ($S$): {S:.2f}\n"
#                           f"Harmonic Mean: {harmonic_score:.1f}% | TOPSIS: {topsis_score:.1f}%")
#             ax.set_title(title_text, fontsize=13, fontweight='bold')
#             ax.tick_params(labelsize=11)

#     fig.supxlabel("Normalized SSIM Drop", fontsize=18, fontweight='bold')
#     fig.supylabel("Normalized Expected Utility ($E_k$)", fontsize=18, fontweight='bold')
#     plt.suptitle(f"Multi-Criteria Optimization Comparison\nImage: {target_image}", fontsize=22, fontweight='bold', y=1.05)
    
#     legend_elements = [
#         Line2D([0], [0], color='dodgerblue', lw=3, label='IC Utility (Want Low)'),
#         Line2D([0], [0], color='darkorange', lw=3, label='OC Utility (Want High)')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=14, frameon=False)
    
#     plt.tight_layout(pad=3.0)
#     plt.subplots_adjust(wspace=0.3, hspace=0.45, top=0.82)
#     plt.savefig("./Holistic_Comparison_Single.pdf", format="pdf", bbox_inches="tight")
#     print("Success! Saved comparison plot to ./Holistic_Comparison_Single.pdf")

# if __name__ == "__main__":
#     main()
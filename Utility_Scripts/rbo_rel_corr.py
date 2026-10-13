# import os
# import json
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import argparse

# def load_json(file_path):
#     with open(file_path, "r") as file:
#         return json.load(file)

# def extract_correlation_data(directories):
#     """
#     Crawls through all model directories, runs, and images to extract
#     paired observations of RBO and Reliability.
#     """
#     data_points = []
    
#     for model_name, base_dir in directories.items():
#         if not os.path.exists(base_dir):
#             print(f"Warning: Directory not found for {model_name}: {base_dir}")
#             continue
            
#         # Iterate through run_1 to run_N
#         for run_folder in os.listdir(base_dir):
#             run_path = os.path.join(base_dir, run_folder)
#             if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
#                 continue
                
#             # Iterate through images inside the run
#             for img_folder in os.listdir(run_path):
#                 img_path = os.path.join(run_path, img_folder)
#                 if not os.path.isdir(img_path):
#                     continue
                    
#                 json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
#                 if not os.path.exists(json_file):
#                     continue
                    
#                 try:
#                     data = load_json(json_file)
#                 except Exception as e:
#                     continue
                
#                 # Extract data for both In-Context and Out-of-Context
#                 for context_type in ["in_context_analysis", "out_of_context_analysis"]:
#                     if context_type not in data:
#                         continue
                        
#                     for noise_type, noise_data in data[context_type].items():
#                         rbo = noise_data.get("average_rank_biased_overlap", np.nan)
#                         reliability_list = noise_data.get("reliability_analysis", [])
                        
#                         if not np.isnan(rbo) and reliability_list:
#                             # Map to final reliability score or weighted IoU fallback
#                             rel_scores = [sp.get("final_reliability_score", sp.get("weighted_iou", 0) / 100.0) for sp in reliability_list]
#                             avg_reliability = np.mean(rel_scores)
                            
#                             data_points.append({
#                                 "Model": model_name,
#                                 "Run": run_folder,
#                                 "Context": "In-Context" if "in" in context_type else "Out-of-Context",
#                                 "Noise Type": noise_type,
#                                 "RBO": rbo,
#                                 "Reliability": avg_reliability
#                             })
                            
#     return pd.DataFrame(data_points)

# def main():
#     parser = argparse.ArgumentParser(description="Correlate RBO with Reliability.")
#     # Use the same directory structure as your LaTeX script
#     parser.add_argument('--directories', type=json.loads, 
#                         default={
#                             "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/new_expresults", 
#                             "SAMv2": "/bhupender/2025/Mouse_other/new_expresults", 
#                             "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/new_expresults", 
#                             "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/new_expresults", 
#                             "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults", 
#                             "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults", 
#                             "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/new_expresults", 
#                             "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/new_expresults", 
#                             "SSVS": "/bhupender/Others/SSVS/SSVS/new_expresults"
#                         },
#                         help="JSON string of directories")
#     args = parser.parse_args()

#     # Create an output directory for the correlation plots in the current working directory
#     output_dir = "./correlation_plots"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Extracting data from all models and runs...")
#     df = extract_correlation_data(args.directories)
    
#     if df.empty:
#         print("No valid data found. Check your directory structure.")
#         return
        
#     print(f"Extracted {len(df)} paired observations across all models.\n")

#     # Drop NaNs
#     df_clean = df.dropna(subset=['RBO', 'Reliability'])
    
#     # ---------------------------------------------------------
#     # 1. Statistical Analysis (Global & Per-Model)
#     # ---------------------------------------------------------
#     print("=== PEARSON CORRELATION (RBO vs. Reliability) ===")
    
#     # Global Correlation
#     r_global, p_global = pearsonr(df_clean['RBO'], df_clean['Reliability'])
#     print(f"GLOBAL (All Models): rho = {r_global:.4f} | p-value = {p_global:.4e}")
    
#     # Per-Model Correlation
#     print("\n--- Per Model Breakdown ---")
#     for model in df_clean['Model'].unique():
#         df_model = df_clean[df_clean['Model'] == model]
#         if len(df_model) > 2:
#             r_val, p_val = pearsonr(df_model['RBO'], df_model['Reliability'])
#             sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
#             print(f"{model:<15}: rho = {r_val:>7.4f} | p-value = {p_val:>10.4e} {sig}")

#     # ---------------------------------------------------------
#     # 2. Visual Analysis (Combined Scatter & Regression)
#     # ---------------------------------------------------------
#     print("\nGenerating correlation scatter plot...")
    
#     sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
#     plt.figure(figsize=(12, 8))
    
#     # We use a single regression line for the global trend, but color the dots by Model
#     ax = sns.scatterplot(
#         data=df_clean, 
#         x='RBO', 
#         y='Reliability', 
#         hue='Model',
#         alpha=0.6, 
#         s=40,
#         palette='tab10',
#         edgecolor=None
#     )
    
#     # Overlay a global regression line (black, dashed)
#     sns.regplot(
#         data=df_clean, 
#         x='RBO', 
#         y='Reliability', 
#         scatter=False, 
#         color='black', 
#         line_kws={'linewidth': 3, 'linestyle': '--'},
#         ax=ax
#     )
    
#     # Annotate the plot with the Global statistical findings
#     textstr = f"Global Pearson $\\rho$ = {r_global:.3f}\n$p < 0.001$" if p_global < 0.001 else f"Global Pearson $\\rho$ = {r_global:.3f}\n$p = {p_global:.3f}$"
#     props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
#     ax.text(0.3, 0.18, textstr, transform=ax.transAxes, fontsize=22,
#             verticalalignment='top', bbox=props)

#     plt.title("RBO vs. Predictive Reliability", fontsize=24, pad=15, fontweight='bold')
#     plt.xlabel("Rank-Biased Overlap (RBO)", fontsize=24, labelpad=10)
#     plt.ylabel("Superpixel Reliability Score", fontsize=24, labelpad=10)
    
#     # Ensure axes are scaled 0 to 1
#     plt.xlim(-0.05, 1.05)
#     plt.ylim(-0.05, 1.05)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     # Clean up the legend
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Models", fontsize=20, title_fontsize=24)
#     plt.tight_layout()

#     # Save outputs
#     save_path_pdf = os.path.join(output_dir, "Global_RBO_vs_Reliability.pdf")
#     save_path_svg = os.path.join(output_dir, "Global_RBO_vs_Reliability.svg")
    
#     plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
#     plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
#     plt.close()
    
#     print(f"Plots saved to {output_dir}")



#     # ---------------------------------------------------------
#     # 3. Supplementary Material: Model-Wise Facet Grid
#     # ---------------------------------------------------------
#     print("\nGenerating model-wise facet grid for supplementary material...")
    
#     # Use lmplot to automatically create a grid of scatter plots with regression lines
#     g = sns.lmplot(
#         data=df_clean, 
#         x='RBO', 
#         y='Reliability', 
#         col='Model',      # Create a new subplot for each model
#         col_wrap=3,       # Wrap into a 3x3 grid
#         hue='Model',      # Keep the colors consistent
#         palette='tab10',
#         scatter_kws={'alpha': 0.5, 's': 20},
#         line_kws={'linewidth': 2, 'color': 'black'}, # Black trendlines for contrast
#         height=4,         # Height of each subplot
#         aspect=1.2
#     )

#     # Clean up the grid aesthetics
#     g.set_axis_labels("Rank-Biased Overlap", "Reliability Score", fontsize=18)
#     g.set_titles(col_template="{col_name}", size=18, weight='bold')
#     g.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
#     for ax in g.axes.flatten():
#         ax.tick_params(axis='both', which='major', labelsize=18)
    
#     plt.suptitle("Model-Wise Correlation: RBO vs. Reliability Score", fontsize=20, y=1.05, weight='bold')
#     g.fig.subplots_adjust(wspace=0.3, hspace=0.5)
#     # Save the supplementary grid
#     grid_save_pdf = os.path.join(output_dir, "Supplementary_ModelWise_Correlation.pdf")
#     grid_save_svg = os.path.join(output_dir, "Supplementary_ModelWise_Correlation.svg")
    
#     plt.savefig(grid_save_pdf, format="pdf", bbox_inches="tight")
#     plt.savefig(grid_save_svg, format="svg", bbox_inches="tight")
#     plt.close()
    
#     print(f"Supplementary grid saved to {grid_save_pdf}")
# if __name__ == "__main__":
#     main()



import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.stats as stats
import argparse

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def extract_correlation_data(directories):
    """
    Crawls through all model directories, runs, and images to extract
    paired observations of RBO and Reliability.
    """
    data_points = []
    
    for model_name, base_dir in directories.items():
        if not os.path.exists(base_dir):
            continue
            
        for run_folder in os.listdir(base_dir):
            run_path = os.path.join(base_dir, run_folder)
            if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
                continue
                
            for img_folder in os.listdir(run_path):
                img_path = os.path.join(run_path, img_folder)
                if not os.path.isdir(img_path):
                    continue
                    
                json_file = os.path.join(img_path, f"{img_folder}_analysis.json")
                if not os.path.exists(json_file):
                    continue
                    
                try:
                    data = load_json(json_file)
                except Exception:
                    continue
                
                for context_type in ["in_context_analysis", "out_of_context_analysis"]:
                    if context_type not in data:
                        continue
                        
                    for noise_type, noise_data in data[context_type].items():
                        rbo = noise_data.get("average_rank_biased_overlap", np.nan)
                        reliability_list = noise_data.get("reliability_analysis", [])
                        
                        if not np.isnan(rbo) and reliability_list:
                            rel_scores = [sp.get("final_reliability_score", sp.get("weighted_iou", 0) / 100.0) for sp in reliability_list]
                            avg_reliability = np.mean(rel_scores)
                            
                            data_points.append({
                                "Model": model_name,
                                "Context": "In-Context" if "in" in context_type else "Out-of-Context",
                                "Noise Type": noise_type,
                                "RBO": rbo,
                                "Reliability": avg_reliability
                            })
                            
    return pd.DataFrame(data_points)

def filter_by_prediction_interval(df, x_col='RBO', y_col='Reliability', alpha=0.05):
    """
    Calculates the strict 95% Prediction Interval (alpha = 0.05) for the linear regression.
    Keeps only the points that fall inside this mathematical boundary.
    """
    x = df[x_col].values
    y = df[y_col].values
    n = len(x)
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    
    # Calculate Standard Error of the Estimate (Residual Standard Error)
    residuals = y - y_pred
    s_e = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Calculate the t-statistic for the given alpha level
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Calculate Margin of Error for the Prediction Interval for each x
    mean_x = np.mean(x)
    ssx = np.sum((x - mean_x)**2)
    margin_of_error = t_val * s_e * np.sqrt(1 + (1/n) + ((x - mean_x)**2 / ssx))
    
    # Define bounds
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error
    
    # Create mask for points strictly within the interval
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    df_filtered = df[mask].copy()
    
    print(f"PI Filter (alpha={alpha}): Retained {len(df_filtered)} points, excluded {len(df) - len(df_filtered)} points outside the interval.")
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description="Correlate RBO with Reliability.")
    parser.add_argument('--directories', type=json.loads, 
                        default={
                            "SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/new_expresults", 
                            "SAMv2": "/bhupender/2025/Mouse_other/new_expresults", 
                            "MSDUNet": "/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/new_expresults", 
                            "TransFuse": "/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/new_expresults", 
                            "D-LKA": "/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/new_expresults", 
                            "TransDeeplab": "/bhupender/2025/Mouse_other/Synapse/transdeeplab/new_expresults", 
                            "CDARL": "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/new_expresults", 
                            "DARL": "/bhupender/2025/cdarl_mouse/DARL/DARL/new_expresults", 
                            "SSVS": "/bhupender/Others/SSVS/SSVS/new_expresults"
                        },
                        help="JSON string of directories")
    args = parser.parse_args()

    output_dir = "./correlation_plots"
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting data from all models and runs...")
    df = extract_correlation_data(args.directories)
    
    if df.empty:
        print("No valid data found. Check your directory structure.")
        return
        
    # Drop basic NaNs
    df = df.dropna(subset=['RBO', 'Reliability'])
    
    # --- APPLY 0.05 INTERVAL FILTERING ---
    df_clean = filter_by_prediction_interval(df, alpha=0.05)
    
    # ---------------------------------------------------------
    # 1. Statistical Analysis (Global & Per-Model)
    # ---------------------------------------------------------
    print("\n=== PEARSON CORRELATION (Interval Bounded) ===")
    
    r_global, p_global = pearsonr(df_clean['RBO'], df_clean['Reliability'])
    print(f"GLOBAL (Bounded): rho = {r_global:.4f} | p-value = {p_global:.4e}")
    
    # ---------------------------------------------------------
    # 2. Visual Analysis (Combined Scatter & Regression)
    # ---------------------------------------------------------
    print("\nGenerating correlation scatter plot...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    
    ax = sns.scatterplot(
        data=df_clean, 
        x='RBO', 
        y='Reliability', 
        hue='Model',
        alpha=0.6, 
        s=40,
        palette='tab10',
        edgecolor=None
    )
    
    # By default, Seaborn plots the 95% Confidence Interval for the regression mean (ci=95)
    sns.regplot(
        data=df_clean, 
        x='RBO', 
        y='Reliability', 
        scatter=False, 
        color='black', 
        line_kws={'linewidth': 3, 'linestyle': '--'},
        ax=ax,
        ci=95 
    )
    
    textstr = f"Global Pearson $\\rho$ = {r_global:.3f}\n$p < 0.05$" if p_global < 0.001 else f"Global Pearson $\\rho$ = {r_global:.3f}\n$p = {p_global:.3f}$"

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.3, 0.18, textstr, transform=ax.transAxes, fontsize=22, verticalalignment='top', bbox=props)

    plt.title("RBO vs. Predictive Reliability (95% PI Bounded)", fontsize=24, pad=15, fontweight='bold')
    plt.xlabel("Rank-Biased Overlap (RBO)", fontsize=24, labelpad=10)
    plt.ylabel("Superpixel Reliability Score", fontsize=24, labelpad=10)
    
    plt.xlim(0.19, 1)
    plt.ylim(0.5, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Models", fontsize=20, title_fontsize=24)
    plt.tight_layout()

    save_path_pdf = os.path.join(output_dir, "Global_RBO_vs_Reliability.pdf")
    save_path_svg = os.path.join(output_dir, "Global_RBO_vs_Reliability.svg")
    plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
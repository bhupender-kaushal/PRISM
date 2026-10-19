import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def calculate_averages(data, context_type):
    """Calculate averages for Final Adjusted Score (weighted_iou), Initial Score, and Rank Perplexity."""
    averages = []
    
    if context_type not in data:
        return pd.DataFrame()
        
    for noise_type, noise_data in data[context_type].items():
        reliability_data = noise_data.get("reliability_analysis", [])
        if reliability_data:
            final_iou_avg = np.mean([sp.get("weighted_iou", sp.get("final_reliability_score", 0) * 100) for sp in reliability_data])
            initial_iou_avg = np.mean([sp["initial_iou"] for sp in reliability_data])
            rank_perplexity_avg = np.mean([sp["rank_perplexity"] for sp in reliability_data])
            
            averages.append({
                "Noise Type": noise_type,
                "Final IoU": final_iou_avg,
                "Rank Perplexity": rank_perplexity_avg,
                "Initial IoU": initial_iou_avg,
            })
    return pd.DataFrame(averages)

def process_directory(base_dir):
    """Aggregate data from all runs and subdirectories in the base directory."""
    in_context_data = []
    out_of_context_data = []

    for run_folder in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_folder)
        
        if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
            continue
            
        for folder_name in os.listdir(run_path):
            folder_path = os.path.join(run_path, folder_name)
            if os.path.isdir(folder_path):
                json_file = os.path.join(folder_path, f"{folder_name}_analysis.json")
                if os.path.exists(json_file):
                    print(f"Processing: {json_file}")
                    data = load_json(json_file)

                    in_context_averages = calculate_averages(data, "in_context_analysis")
                    out_of_context_averages = calculate_averages(data, "out_of_context_analysis")

                    if not in_context_averages.empty:
                        in_context_data.append(in_context_averages)
                    if not out_of_context_averages.empty:
                        out_of_context_data.append(out_of_context_averages)

    in_context_df = pd.concat(in_context_data, ignore_index=True) if in_context_data else pd.DataFrame()
    out_of_context_df = pd.concat(out_of_context_data, ignore_index=True) if out_of_context_data else pd.DataFrame()

    return in_context_df, out_of_context_df

def plot_averages_seaborn(averages_df, context_type, output_file):
    """Plot Initial and Final bars side-by-side, with Rank Perplexity nested inside the Initial bar."""
    if averages_df.empty:
        print(f"No data to plot for {context_type}. Skipping.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 15))
    
    x = np.arange(len(averages_df))  
    main_width = 0.25   # Width for the Initial and Final bars
    nest_width = 0.15   # Thinner width for the nested RP bar

    # 1. Plot Initial Score (Left Main Bar)
    plt.bar(x - main_width/2, averages_df["Initial IoU"], main_width, label="Initial Score", color="dodgerblue", alpha=0.9)

    # 2. Plot Rank Perplexity (Nested INSIDE the Left Main Bar)
    plt.bar(x - main_width/2, averages_df["Rank Perplexity"], nest_width, label="Rank Perplexity", color="orange", alpha=1.0)

    # 3. Plot Final Adjusted Score (Right Main Bar)
    plt.bar(x + main_width/2, averages_df["Final IoU"], main_width, label="Final Adjusted Score", color="steelblue", alpha=0.9)

    # Determine dynamic text offset based on max values
    max_score = averages_df[["Initial IoU", "Final IoU"]].max().max()
    offset = max_score * 0.015

    # Add numeric values on top of the bars
    for i in range(len(averages_df)):
        init_val = averages_df["Initial IoU"].iloc[i]
        rp_val = averages_df["Rank Perplexity"].iloc[i]
        fin_val = averages_df["Final IoU"].iloc[i]

        # Text on top of Initial Bar
        plt.text(x[i] - main_width/2, init_val + offset, f"{init_val:.1f}", 
                 ha='center', va='bottom', fontsize=20, color="dodgerblue", fontweight='bold')
        
        # Text on top of Final Bar
        plt.text(x[i] + main_width/2, fin_val + offset, f"{fin_val:.1f}", 
                 ha='center', va='bottom', fontsize=20, color="steelblue", fontweight='heavy')
        
        # Text strictly above the tiny nested RP bar (with a small background box for readability)
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none")
        plt.text(x[i] - main_width/2, rp_val + (offset*0.5), f"{rp_val:.2f}", 
                 ha='center', va='bottom', fontsize=20, color="black", fontweight='bold', bbox=bbox_props)

    # Customize the plot aesthetics
    plt.xticks(x, averages_df["Noise Type"], rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("Noise Types", fontsize=26, labelpad=10)
    plt.ylabel("IoU Score", fontsize=26, labelpad=10)
    plt.title(f"{context_type.replace('_', ' ').title()}", fontsize=30, pad=10)
    plt.legend(fontsize=24, loc='upper right')

    # Extend Y-axis slightly to fit the top annotations
    plt.ylim(0, max_score * 1.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, format="pdf", bbox_inches="tight")  
    svg_output_file = output_file.replace(".pdf", ".svg")  
    plt.savefig(svg_output_file, format="svg", bbox_inches="tight")  
    print(f"Plot saved as {output_file} (and .svg)")
    plt.close()

def main():
    # Base directory containing the Two-Phase run subdirectories
    base_dir = "./new_expresults"  
    
    # Specific output directory for these bar plots
    output_dir = os.path.join(base_dir, "plots/across_images")

    os.makedirs(output_dir, exist_ok=True)

    in_context_df, out_of_context_df = process_directory(base_dir)

    if not in_context_df.empty:
        in_context_avg = in_context_df.groupby("Noise Type").mean().reset_index()
        plot_averages_seaborn(in_context_avg, "in_context_analysis", os.path.join(output_dir, "in_context_summary.pdf"))

    if not out_of_context_df.empty:
        out_of_context_avg = out_of_context_df.groupby("Noise Type").mean().reset_index()
        plot_averages_seaborn(out_of_context_avg, "out_of_context_analysis", os.path.join(output_dir, "out_of_context_summary.pdf"))

if __name__ == "__main__":
    main()
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# filepath: /path/to/plot_noise_analysis.py
# Replace with the path to your JSON file
json_file_path = "/bhupender/Others/SSVS/SSVS/expresults/00018_33/00018_33_analysis.json"

def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def calculate_averages(data, context_type):
    """Calculate averages for pIoU, initial_iou, avg_iou, and rank_perplexity."""
    averages = []
    for noise_type, noise_data in data[context_type].items():
        reliability_data = noise_data["reliability_analysis"]
        if reliability_data:
            piou_avg = np.mean([sp["pIoU"] for sp in reliability_data])
            initial_iou_avg = np.mean([sp["initial_iou"] for sp in reliability_data])
            avg_iou_avg = np.mean([sp["avg_iou"] for sp in reliability_data])
            rank_perplexity_avg = np.mean([sp["rank_perplexity"] for sp in reliability_data])
            averages.append({
                "Noise Type": noise_type,
                "pIoU": piou_avg,
                "Avg IoU": avg_iou_avg,
                "Rank Perplexity": rank_perplexity_avg,
                "Initial IoU": initial_iou_avg,
            })
    return pd.DataFrame(averages)

def plot_averages_seaborn(averages_df, context_type):
    """Plot the averages for each noise type using Seaborn."""
    # Set the Seaborn style
    sns.set(style="whitegrid")

    # Create the bar plot
    plt.figure(figsize=(12,7))
    x = np.arange(len(averages_df))  # X-axis positions
    width = 0.3  # Bar width

    # Plot pIoU
    plt.bar(x - width, averages_df["pIoU"], width, label="pIoU", color="steelblue")

    # Plot Avg IoU
    plt.bar(x, averages_df["Avg IoU"], width, label="avgIoU", color="lightblue")

    # Plot Rank Perplexity as a sub-bar inside Avg IoU
    plt.bar(x, averages_df["Rank Perplexity"], width * 0.5, label="Rank Perplexity", 
            color="orange", alpha=0.7)

    # Add numeric value of Rank Perplexity on top of the bar
    for i, rp in enumerate(averages_df["Rank Perplexity"]):
        plt.text(x[i], rp + 0.02, f"{rp:.2f}", ha='center', va='bottom', fontsize=10, color="black")

    # Plot Initial IoU
    plt.bar(x + width, averages_df["Initial IoU"], width, label="iniIoU", color="dodgerblue")

    # Customize the plot
    plt.xticks(x, averages_df["Noise Type"], rotation=45)
    plt.xlabel("Noise Types")
    plt.ylabel("IoU Score")
    plt.title(f"{context_type.replace('_', ' ').title()}")
    plt.legend()

    # Adjust the y-axis scale to make the bars look better

    plt.ylim(0, max(averages_df[[ "Initial IoU"]].sum(axis=1)) * 1.4)

    plt.tight_layout()

    # Save the plot as a PDF file
    output_file = f"{context_type}_averages_plot_seaborn_stacked.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")  # Ensure the plot fits within the PDF
    print(f"Plot saved as {output_file}")

    # Show the plot
    plt.show()

def main():
    data = load_json(json_file_path)

    # Process in-context data
    in_context_averages = calculate_averages(data, "in_context_analysis")
    plot_averages_seaborn(in_context_averages, "in_context_analysis")

    # Process out-of-context data
    out_of_context_averages = calculate_averages(data, "out_of_context_analysis")
    plot_averages_seaborn(out_of_context_averages, "out_of_context_analysis")

if __name__ == "__main__":
    main()
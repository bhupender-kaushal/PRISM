import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging


def load_reliability_data(directory, method_name):
    """Load and aggregate reliability data from JSON files in the given directory."""
    reliability_data = {'in_context': {}, 'out_of_context': {}}

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            # Look for the JSON file inside the folder
            json_file = os.path.join(folder_path, f"{folder}_analysis.json")
            if os.path.exists(json_file):
                logging.info(f"Processing image '{folder}' for method '{method_name}'...")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Process in_context_analysis and out_of_context_analysis
                    for analysis_key in ['in_context_analysis', 'out_of_context_analysis']:
                        context = 'in_context' if analysis_key == 'in_context_analysis' else 'out_of_context'
                        if analysis_key in data:
                            for noise_type, noise_data in data[analysis_key].items():
                                if noise_type not in reliability_data[context]:
                                    reliability_data[context][noise_type] = []
                                # Extract final_reliability_score and compute the average for the image
                                if 'reliability_analysis' in noise_data:
                                    scores = [
                                        sp['final_reliability_score']
                                        for sp in noise_data['reliability_analysis']
                                    ]
                                    if scores:
                                        avg_score = np.mean(scores)
                                        reliability_data[context][noise_type].append(avg_score)
            else:
                logging.warning(f"No JSON file found for image '{folder}' in method '{method_name}'.")

    # Compute the overall average across images for each noise type
    for context in reliability_data:
        for noise_type in reliability_data[context]:
            reliability_data[context][noise_type] = np.mean(reliability_data[context][noise_type])

    return reliability_data


def plot_reliability_comparison(reliability_data_dict, output_dir):
    """Generate comparison plots for reliability scores across multiple methods."""
    os.makedirs(output_dir, exist_ok=True)
    methods = list(reliability_data_dict.keys())
    for analysis_type in reliability_data_dict[methods[0]].keys():
        plot_data = []
        for method, reliability_data in reliability_data_dict.items():
            if analysis_type in reliability_data:
                for noise_type, scores in reliability_data[analysis_type].items():
                    avg_score = np.mean(scores) if scores else 0.0
                    plot_data.append({
                        'Noise Type': noise_type,
                        'Method': method,
                        'Average Reliability Score': avg_score
                    })
        if not plot_data:
            logging.warning(f"No data available for analysis type '{analysis_type}'. Skipping plot generation.")
            continue
        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 9))
        sns.barplot(data=df_plot, x='Noise Type', y='Average Reliability Score', hue='Method', palette='mako')
        plt.ylabel("Reliability Score", fontsize=16)  # Update the y-axis label here
        plt.xlabel("Noise Type", fontsize=16)  # Update the x-axis label here
        plt.title(f"Reliability Comparison ({analysis_type.replace('_', ' ').title()})", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{analysis_type}_reliability_comparison.pdf")  # Save as PDF
        plt.savefig(plot_path, format='pdf', dpi=1000)  # Specify higher DPI (600)
        logging.info(f"Saved plot for analysis type '{analysis_type}' to {plot_path}")
        plt.close()




def main():
    parser = argparse.ArgumentParser(description="Compare reliability of multiple methods.")
    parser.add_argument('--directories', type=json.loads, required=True,
                        help="JSON string of directories in the format: '{\"Method 1\": \"/path/to/dir1\", \"Method 2\": \"/path/to/dir2\"}'")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for comparison plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('comparison')

    reliability_data_dict = {}
    for method, directory in args.directories.items():
        logger.info(f"Loading reliability data for {method}...")
        reliability_data_dict[method] = load_reliability_data(directory, method)
        if not reliability_data_dict[method]:
            logger.warning(f"No reliability data found for {method} in directory {directory}.")

    logger.info("Generating comparison plots...")
    plot_reliability_comparison(reliability_data_dict, args.output_dir)

    logger.info(f"Comparison plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()



    # python model_comparison_Plots.py --directories '{"SAMv1": "/bhupender/2025/Mouse_other/SAMv1/segment-anything/expresults", "SAMv2": "/bhupender/2025/Mouse_other/expresults"}' --output_dir ./comparison
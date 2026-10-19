import os
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def fetch_rbo_scores_from_json(folder_path):
    """
    Fetches and averages RBO scores from all JSON files in the specified folder.
    """
    rbo_plot_data = []

    # Traverse the folder to find all JSON files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_analysis.json"):
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    analysis_data = json.load(f)

                # Extract and average RBO scores for in-context and out-of-context analyses
                for analysis_type in ['in_context_analysis', 'out_of_context_analysis']:
                    if analysis_type in analysis_data:
                        for noise_type, noise_data in analysis_data[analysis_type].items():
                            rbo_scores = noise_data.get('rbo_at_each_level', [])
                            if rbo_scores:
                                # Average RBO scores for this noise type
                                avg_rbo_score = np.mean(rbo_scores)
                                rbo_plot_data.append({
                                    'Analysis Type': analysis_type.replace('_', ' ').title(),
                                    'Noise Type': noise_type,
                                    'RBO Score': avg_rbo_score
                                })
    return pd.DataFrame(rbo_plot_data)

def main():

    parser = argparse.ArgumentParser(description="Generate a combined RBO Violin Plot from multiple directories.")
    
    # parser.add_argument('--input_dirs', type=str, nargs='+', default=['/bhupender/2025/Mouse_other/SAMv1/segment-anything/expresults','/bhupender/2025/Mouse_other/expresults',  '/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/expresults', '/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/expresults', '/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/expresults', '/bhupender/2025/Mouse_other/Synapse/transdeeplab/expresults', '/bhupender/2025/cdarl_mouse/MEDIA_CDARL/expresults/mouse_ranked', '/bhupender/2025/cdarl_mouse/DARL/DARL/expresults/mouse_ranked'], required=False, help='Paths to the folders containing analysis JSON files.')

    parser.add_argument('--input_dirs', type=str, nargs='+', default=['/bhupender/2025/Mouse_other/SAMv1/segment-anything/expresults_quickshift','/bhupender/2025/Mouse_other/expresults_quickshift',  '/bhupender/2025/Mouse_other/Skin_Lesion/MSDUNet/expresults_quickshift', '/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/expresults_quickshift', '/bhupender/2025/Mouse_other/Synapse/deformableLKA/2D/expresults_quickshift', '/bhupender/2025/Mouse_other/Synapse/transdeeplab/expresults_quickshift', '/bhupender/2025/cdarl_mouse/MEDIA_CDARL/expresults/mouse_ranked_quickshift', '/bhupender/2025/cdarl_mouse/DARL/DARL/expresults/mouse_ranked_quickshift'], required=False, help='Paths to the folders containing analysis JSON files.')

    parser.add_argument('--titles', type=str, nargs='+', default=['SAMv1','SAMv2',  'MSDUNet', 'TransFuse', 'DLKA', 'TransDeepLab', 'CDARL', 'DARL'], required=False, help='Titles for each directory plot (must match the number of input directories).')
    parser.add_argument('--output_path', type=str, default='./RBO_Distributions_Columnar_quickshift.png', help='Path to save the combined violin plot.')
    parser.add_argument('--normalize', action='store_true', help='Normalize all RBO scores collectively to [0, 1].')
    args = parser.parse_args()

    if len(args.input_dirs) != len(args.titles):
        print("Error: The number of input directories must match the number of titles.")
        return

    # --- Plotting Logic ---
    num_titles = len(args.titles)
    rows_per_col = 4
    
    # Grid will have 4 rows. It will have 2 columns for models, and each model has 2 plots.
    # So, we need 4 columns for the actual plots (axes).
    grid_rows = rows_per_col
    grid_cols = 4 
    
    fig, axes = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(30, 5 * grid_rows))

    # Handle case where subplots returns a 1D array
    if grid_rows == 1:
        axes = np.array([axes])

    
    # Global axis labels
    fig.supxlabel('Noise Type', fontsize=26, y=0.06)
    fig.supylabel('RBO Score', fontsize=26, x=0.08)


    # Loop through each directory to create the plots
    for i, directory in enumerate(args.input_dirs):
        # Determine the plot's position in the grid
        if i < rows_per_col:
            # This plot goes in the first model column
            row, col_start = i, 0
        else:
            # This plot goes in the second model column
            row, col_start = i - rows_per_col, 2
        
        print(f"Processing '{args.titles[i]}' for grid position ({row}, {col_start})...")
        
        df = fetch_rbo_scores_from_json(directory)

        if df.empty:
            print(f"Warning: No data found in {directory}. Skipping plot for '{args.titles[i]}'.")
            axes[row, col_start].set_visible(False)
            axes[row, col_start + 1].set_visible(False)
            continue

        ax1, ax2 = axes[row, col_start], axes[row, col_start + 1]

        # --- Subplot 1: In Context Analysis ---
        df_in_context = df[df['Analysis Type'] == 'In Context Analysis']
        sns.violinplot(data=df_in_context, x='Noise Type', y='RBO Score', palette='viridis', ax=ax1)
        ax1.set_xlabel(None); ax1.set_ylabel(None)


        # --- Subplot 2: Out Of Context Analysis ---
        df_out_context = df[df['Analysis Type'] == 'Out Of Context Analysis']
        sns.violinplot(data=df_out_context, x='Noise Type', y='RBO Score', palette='viridis', ax=ax2)
        ax2.set_xlabel(None); ax2.set_ylabel(None)
        # ax2.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

    # --- Hide any unused subplots at the end of the grid ---
    for i in range(num_titles, rows_per_col * 2):
        if i < rows_per_col:
            row, col_start = i, 0
        else:
            row, col_start = i - rows_per_col, 2
        
        if row < grid_rows: # Boundary check
            axes[row, col_start].set_visible(False)
            axes[row, col_start + 1].set_visible(False)


    # --- Add Centered Row Titles (AFTER layout adjustment) ---
    for i, title in enumerate(args.titles):
        if i < rows_per_col:
            row, col_start = i, 0
        else:
            row, col_start = i - rows_per_col, 2
        
        if not axes[row, col_start].get_visible():
            continue
            
        x0 = axes[row, col_start].get_position().x0
        x1 = axes[row, col_start + 1].get_position().x1
        y_pos = axes[row, col_start].get_position().y1
        
        # Add the title text, centered above the row's pair of plots
        fig.text((x0 + x1) / 2, y_pos+0.005, title, ha='center', va='bottom', fontsize=22, weight='bold')



        # --- Add a legend for Noise Type colors ---
    unique_noise_types = df['Noise Type'].str.replace('_', ' ').str.title().unique()
    unique_noise_types = [noise.replace('Salt Pepper', 'Salt and Pepper') for noise in unique_noise_types]
    color_palette = sns.color_palette('viridis', len(unique_noise_types))
    noise_color_mapping = dict(zip(unique_noise_types, color_palette))

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=noise_color_mapping[noise], markersize=30, label=noise)
        for noise in unique_noise_types
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(unique_noise_types), fontsize=16, title=' ', title_fontsize=18, frameon=False)


        
    # Add "In Context" and "Out of Context" labels above each column
    fig.text(0.20, 0.90, "In Context", ha='center', va='bottom', fontsize=24)
    fig.text(0.40, 0.90, "Out of Context", ha='center', va='bottom', fontsize=24)
    fig.text(0.60, 0.90, "In Context", ha='center', va='bottom', fontsize=24)
    fig.text(0.80, 0.90, "Out of Context", ha='center', va='bottom', fontsize=24)
    plt.savefig(args.output_path, bbox_inches='tight', dpi=300)
    # Also save as PDF
    pdf_output_path = os.path.splitext(args.output_path)[0] + '.pdf'
    plt.savefig(pdf_output_path, bbox_inches='tight', dpi=300, format='pdf')
    plt.close(fig)
    print(f"Combined violin plot saved to {args.output_path}")

if __name__ == "__main__":
    main()

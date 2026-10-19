import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import exp

def calculate_rbo(list1, list2, p=0.9):
    """Calculates Rank-Biased Overlap between two lists."""
    if not list1 or not list2: return 0.0
    s_len = min(len(list1), len(list2))
    agreement = 0.0
    for d in range(1, s_len + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        agreement += len(set1.intersection(set2)) / d * (p ** (d - 1))
    return (1 - p) * agreement

def simulate_rank_chaos(num_superpixels=500, num_steps=10, lambda_param=1.0, num_simulations=100):
    """
    Simulates trajectories of ranked lists with varying degrees of chaos.
    IoU is assumed to be a constant 100% to isolate the effect of Rank Perplexity.
    """
    results = []
    base_list = list(range(1, num_superpixels + 1))
    
    # We simulate 100 different runs, from 0% chaos to 100% chaos
    for chaos_factor in np.linspace(0.0, 1.0, num_simulations):
        
        # Track history across perturbation steps
        history = [base_list.copy()]
        current_list = base_list.copy()
        rbo_scores = []
        
        # Simulate 'num_steps' of increasing noise
        for step in range(num_steps):
            # The higher the chaos_factor, the more swaps occur
            num_swaps = int((num_superpixels / 2) * chaos_factor * ((step + 1) / num_steps))
            
            for _ in range(num_swaps):
                idx1, idx2 = np.random.choice(num_superpixels, 2, replace=False)
                current_list[idx1], current_list[idx2] = current_list[idx2], current_list[idx1]
            
            history.append(current_list.copy())
            rbo_scores.append(calculate_rbo(base_list, current_list, p=0.9))
            
        avg_run_rbo = np.mean(rbo_scores)
        
        # Calculate Rank Perplexity using your exact methodology
        total_rp = 0
        for sp_id in base_list:
            # Extract the trajectory of this specific superpixel across the steps
            sp_trajectory = [step_list.index(sp_id) + 1 for step_list in history]
            
            # Calculate jumps
            num_changes = 0
            sum_of_jumps = 0
            for i in range(1, len(sp_trajectory)):
                if sp_trajectory[i] != sp_trajectory[i-1]:
                    num_changes += 1
                    sum_of_jumps += abs(sp_trajectory[i] - sp_trajectory[i-1])
            
            # Normalize Perplexity
            sp_rp = (sum_of_jumps / num_changes) if num_changes > 0 else 0.0
            sp_rp = sp_rp / num_superpixels
            total_rp += sp_rp
            
        avg_run_rp = total_rp / num_superpixels
        
        # Calculate RP-Based Reliability (Assuming constant 100 IoU)
        # Final Score = IoU * exp(-lambda * RP)
        constant_iou = 100.0
        confidence_weight = exp(-lambda_param * avg_run_rp * 10) # *10 scales the visual drop
        final_reliability = constant_iou * confidence_weight
        
        results.append({
            'Chaos Factor': chaos_factor,
            'Average RBO': avg_run_rbo,
            'Rank Perplexity': avg_run_rp,
            'Reliability Score': final_reliability
        })
        
    return pd.DataFrame(results)

def plot_contrasts(df):
    """Generates a 1x2 plot contrasting RBO against RP and Reliability."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- PLOT 1: RBO vs Rank Perplexity (Inverse Relationship) ---
    sns.scatterplot(data=df, x='Average RBO', y='Rank Perplexity', c=df['Chaos Factor'], cmap='coolwarm', s=80, ax=axes[0], edgecolor='k')
    axes[0].set_title("Rank-Biased Overlap vs. Rank Perplexity", fontsize=18, fontweight='bold', pad=15)
    axes[0].set_xlabel("Rank-Biased Overlap (Similarity to Original)", fontsize=16)
    axes[0].set_ylabel("Rank Perplexity (Trajectory Volatility)", fontsize=16)
    
    # --- PLOT 2: RBO vs Reliability (Direct Relationship) ---
    sns.scatterplot(data=df, x='Average RBO', y='Reliability Score', c=df['Chaos Factor'], cmap='coolwarm', s=80, ax=axes[1], edgecolor='k')
    axes[1].set_title("RBO vs. Final Reliability (Constant 100% IoU)", fontsize=18, fontweight='bold', pad=15)
    axes[1].set_xlabel("Rank-Biased Overlap (Similarity to Original)", fontsize=16)
    axes[1].set_ylabel("RP-Discounted Reliability Score", fontsize=16)
    
    # Add an explanatory text box to Plot 2
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    textstr = "Notice that even with a perfect\npredictive output (100% IoU),\nthe Reliability Score drops\nproportionally with RBO due to\nhigh internal Rank Perplexity."
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    
    # Formatting
    for ax in axes:
        ax.tick_params(labelsize=14)
        ax.invert_xaxis() # Invert X-axis so it reads left-to-right from High Chaos to Low Chaos
    
    plt.tight_layout()
    plt.savefig("Metric_Contrast_Simulation.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("Metric_Contrast_Simulation.png", format="png", bbox_inches="tight", dpi=300)
    print("Plots saved as Metric_Contrast_Simulation.pdf and .png")

if __name__ == "__main__":
    print("Simulating Rank Chaos (Assuming Constant IoU)...")
    df_sim = simulate_rank_chaos(num_superpixels=500, num_steps=10, lambda_param=1.0, num_simulations=150)
    print("Generating Contrast Plots...")
    plot_contrasts(df_sim)
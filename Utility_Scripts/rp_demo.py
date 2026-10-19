import numpy as np
import matplotlib.pyplot as plt

def plot_rp_penalization():
    """
    Plots the penalization effect of Rank Perplexity (RP) on the Final Reliability Score,
    formatted strictly according to the provided plotting structure.
    """
    # 1. Define the Parameters
    # Using np.arange to match the discrete integer steps and marker logic
    rp_values = np.arange(0, 21, 1)  
    base_iou = 100.0
    
    # Three lambda scenarios to match the three-line layout
    lambdas = [0.05, 0.15, 0.30]
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(17, 12))

    # 2. Calculate and Plot the Curves
    for i, lam in enumerate(lambdas):
        reliability_scores = base_iou * np.exp(-lam * rp_values)
        
        plt.plot(
            rp_values, 
            reliability_scores, 
            label=fr'Penalty Factor ($\lambda$): {lam:.2f}', 
            linewidth=3, 
            marker='o', 
            color=colors[i], 
            linestyle='-'
        )

        # Annotate the points with the reliability values on every second rank perplexity
        for j, (rp, rel) in enumerate(zip(rp_values, reliability_scores)):
            if j > 0 and j % 2 == 0 and j < 20:
                # Dynamic vertical offset logic so annotations do not overlap the lines
                if i == 0:
                    va_align = 'baseline'
                    offset = 1.0
                elif i == 1:
                    va_align = 'top'
                    offset = -1.0
                else:
                    va_align = 'top'
                    offset = -1.5

                plt.text(
                    rp, 
                    rel, 
                    f'{rel:.1f}', 
                    fontsize=20, 
                    ha='left', 
                    va=va_align, 
                    color='black', 
                    position=(rp, rel + offset)
                )

    # 3. Aesthetics and Formatting (Sourced from provided template)
    plt.xlabel('Rank Perplexity (RP)', fontsize=25)
    plt.ylabel('Final Reliability Score', fontsize=25)
    plt.title('Effect of Rank Perplexity on Reliability Score', fontsize=30, fontweight='bold')
    
    plt.ylim(0, 105) 
    plt.xlim(0, 20)  
    
    # Restricting x-ticks to min and max boundaries as specified
    plt.xticks([0, 20], fontsize=20)  
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20) 
    plt.tight_layout(pad=1.0)
    
    # 4. Save Outputs
    plt.savefig("Reliability_vs_RP_Line.pdf", dpi=1000)
    plt.savefig("Reliability_vs_RP_Line.svg", format="svg")
    print("Line plot saved to Reliability_vs_RP_Line.pdf and Reliability_vs_RP_Line.svg")

if __name__ == "__main__":
    plot_rp_penalization()
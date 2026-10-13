import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def generate_propositions_plot(output_path="Theoretical_Propositions_1_and_2.pdf"):
    # Define theoretical noise grid [0, 1]
    noise_intensity = np.linspace(0, 1.0, 200)
    E_0 = 95.0  # Base utility ceiling

    # ---------------------------------------------------------
    # Theoretical Trajectories
    # ---------------------------------------------------------
    # Prop 1 (IC): Sigmoid-like structural collapse due to disrupted semantics
    # E_ic drops as local features inside M_target are destroyed
    curve_ic = E_0 * (1.0 - (1.0 / (1.0 + np.exp(-12 * (noise_intensity - 0.45)))))
    
    # Prop 2 (OC): Perfect stability due to ERF containment / ViT attention down-weighting
    curve_oc = np.full_like(noise_intensity, E_0)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # Shade the Separation Gap
    ax.fill_between(noise_intensity, curve_ic, curve_oc, color='lightgreen', alpha=0.15, label="Ideal Separation ($G -> 1.0$)")

    # Plot the curves
    line_oc, = ax.plot(noise_intensity, curve_oc, color='darkorange', lw=4, label="Out-of-Context (OC) Perturbation Curve   ")
    line_ic, = ax.plot(noise_intensity, curve_ic, color='dodgerblue', lw=4, label="In-Context (IC) Perturbation Curve")

    # ---------------------------------------------------------
    # Annotations for Proposition 1
    # ---------------------------------------------------------
    ax.annotate(
        "Proposition 1 (Foreground Robustness):\n"
        "Semantic features within $GT$ are perturbed.\n"
        "Dense spatial alignment fails to maintain regional predictive capability.",
        xy=(0.6, 15), xytext=(0.05, 25),
        arrowprops=dict(facecolor='dodgerblue', shrink=0.05, width=2, headwidth=8),
        fontsize=14, fontweight='bold', color='black',
        bbox=dict(boxstyle="round,pad=0.4", fc="aliceblue", ec="dodgerblue", lw=1.5)
    )

    # ---------------------------------------------------------
    # Annotations for Proposition 2
    # ---------------------------------------------------------
    ax.annotate(
        "Proposition 2 (Background Invariance):\n"
        "Local ERF limits (CNNs) & attention down-weighting (ViTs)\n"
        "prevent background noise from propagating globally.",
        xy=(0.6, E_0), xytext=(0.35, E_0 - 35),
        arrowprops=dict(facecolor='darkorange', shrink=0.05, width=2, headwidth=8),
        fontsize=14, fontweight='bold', color='black',
        bbox=dict(boxstyle="round,pad=0.4", fc="papayawhip", ec="darkorange", lw=1.5)
    )

    # ---------------------------------------------------------
    # Axis & Formatting
    # ---------------------------------------------------------
    ax.set_ylim(-5, 110)
    ax.set_xlim(-0.02, 1.02)
    ax.axhline(y=E_0, color="gray", linestyle="--", lw=2, alpha=0.6, zorder=0)
    
    ax.set_xlabel("Perturbation Intensity ($\delta$)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Utility ($E$)%", fontsize=20, fontweight='bold')
    ax.set_title("Theoretical Support for Proposition 1 and 2 ", fontsize=24, fontweight='bold', pad=20)
    
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle=":", alpha=0.7)
    
    # Legend
    ax.legend(loc='lower left', fontsize=14, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated Theoretical Propositions Plot: {output_path}")

if __name__ == "__main__":
    generate_propositions_plot()
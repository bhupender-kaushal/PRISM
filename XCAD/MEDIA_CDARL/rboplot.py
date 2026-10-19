import json
import matplotlib.pyplot as plt

json_path = "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/expresults/mouse_ranked/00018_33/00018_33_analysis.json"

with open(json_path) as f:
    data = json.load(f)

noise_types = {
    "poisson": "noise_intensity_scale",
    "gaussian": "noise_std_dev",
    "speckle": "noise_std_dev",
    "salt_pepper": "noise_density",
    "gaussian_blur": "blur_sigma",
    "rician": "noise_sigma"
}

plt.style.use("seaborn-v0_8-dark-palette")
fig, ax = plt.subplots(figsize=(10, 6))

for noise, level_key in noise_types.items():
    if noise not in data["in_context_analysis"]:
        continue
    analysis = data["in_context_analysis"][noise]
    levels = [entry.get(level_key) for entry in analysis["level_data"] if level_key in entry]
    rbo = analysis["rbo_at_each_level"]
    n = min(len(levels), len(rbo))
    if n == 0:
        continue
    ax.plot(levels[:n], rbo[:n], marker='o', label=noise.replace("_", " ").title())

ax.set_xlabel("Noise Intensity")
ax.set_ylabel("Rank Biased Overlap (RBO)")
ax.set_title("RBO vs Noise Intensity (Averaged Over All Images)")
ax.legend(title="Noise Type")
plt.tight_layout()
plt.savefig("rbo_vs_noise_intensity.png", dpi=300)
plt.close()
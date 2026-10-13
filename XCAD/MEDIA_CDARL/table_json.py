import os
import json
import random

# Directory containing your JSON files
json_dir = "/bhupender/2025/cdarl_mouse/MEDIA_CDARL/expresults/mouse_ranked"
noise_types = ["poisson", "gaussian", "speckle", "salt_pepper", "gaussian_blur", "rician"]
num_images = 10

# Find all analysis.json files
json_files = []
for root, dirs, files in os.walk(json_dir):
    for file in files:
        if file.endswith("_analysis.json"):
            json_files.append(os.path.join(root, file))

# Pick 20 random images
random.seed(42)
selected_files = random.sample(json_files, min(num_images, len(json_files)))

# For each image, extract reliability data for each noise type
table_data = {nt: [] for nt in noise_types}
image_names = []

for f in selected_files:
    with open(f, "r") as jf:
        data = json.load(jf)
    image_name = os.path.basename(f).split("_analysis.json")[0]
    image_names.append(image_name)
    for nt in noise_types:
        rels = data.get("out_of_context_analysis", {}).get(nt, {}).get("reliability_analysis", [])
        if rels:
            # Take average over all superpixels
            avg_rank_jump = sum(r.get("avg_rank_jump", 0) for r in rels) / len(rels)
            avg_iou = sum(r.get("avg_iou", 0) for r in rels) / len(rels)
            avg_reliability = sum(r.get("reliability_score", 0) for r in rels) / len(rels)
            table_data[nt].append((avg_rank_jump, avg_iou, avg_reliability))
        else:
            table_data[nt].append((None, None, None))

# Build LaTeX table
header = "Noise Type"
for name in image_names:
    header += f" & \\multicolumn{{2}}{{c}}{{{name}}}"
header += " \\\\\n"
header += " & " + " & ".join(["RankJump & IoU" for _ in image_names]) + " \\\\\n"
header += "\\hline\n"

rows = []
for nt in noise_types:
    # First sub-row: avg_rank_jump & avg_iou
    row1 = [nt.replace("_", "\\_")]
    for tup in table_data[nt]:
        if tup[0] is not None:
            row1.append(f"{tup[0]:.2f} & {tup[1]:.2f}")
        else:
            row1.append("-- & --")
    rows.append(" & ".join(row1) + " \\\\")
    # Second sub-row: reliability score
    row2 = [""]
    for tup in table_data[nt]:
        if tup[2] is not None:
            row2.append(f"\\multicolumn{{2}}{{c}}{{{tup[2]:.2f}}}")
        else:
            row2.append("\\multicolumn{2}{c}{--}")
    rows.append(" & ".join(row2) + " \\\\ \\hline")

table = "\\begin{tabular}{l" + "cc" * num_images + "}\n"
table += header
table += "\n".join(rows)
table += "\n\\end{tabular}"

print(table)
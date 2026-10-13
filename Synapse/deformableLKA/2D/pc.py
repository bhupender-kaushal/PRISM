import os
import json
from scipy.stats import pearsonr

# Directory containing all *_analysis.json files
json_dir = "/bhupender/2025/Mouse_other/MRI/deformableLKA/2D/results_full_reliability"

noise_types = ["poisson", "gaussian", "speckle", "s&p", "gaussian_blur", "rician"]
contexts = ["in_context_analysis", "out_of_context_analysis"]

results = {}

for context in contexts:
    results[context] = {}
    for noise in noise_types:
        iou_before = []
        iou_after = []
        # Loop over all images
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if not file.endswith("_analysis.json"):
                    continue
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                noise_data = data.get(context, {}).get(noise, {})
                level_data = noise_data.get("level_data", [])
                # For each pair of consecutive noise levels, collect IoU for all superpixels
                for i in range(len(level_data) - 1):
                    # Only keep superpixels with both 'sp_id' and 'iou'
                    sp_before = {sp['sp_id']: sp['iou'] for sp in level_data[i].get('superpixels', []) if 'sp_id' in sp and 'iou' in sp}
                    sp_after = {sp['sp_id']: sp['iou'] for sp in level_data[i+1].get('superpixels', []) if 'sp_id' in sp and 'iou' in sp}
                    # Only consider superpixels present in both levels
                    for sp_id in sp_before:
                        if sp_id in sp_after:
                            iou_before.append(sp_before[sp_id])
                            iou_after.append(sp_after[sp_id])
        if len(iou_before) > 1 and len(iou_after) > 1:
            r, p = pearsonr(iou_before, iou_after)
            results[context][noise] = {"pearson_r": r, "p_value": p, "n_pairs": len(iou_before)}
        else:
            results[context][noise] = {"pearson_r": None, "p_value": None, "n_pairs": 0}

# Save results
with open("iou_pearson_correlation_overall.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to iou_pearson_correlation_overall.json")
import os
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from new_assessments import calculate_holistic_s_factor, trapezoid

# Set seaborn style for publication-ready plots
sns.set_theme(style="whitegrid")
COMMON_X = np.linspace(0.0, 1.0, 101)

# ==========================================
# CONFIGURATION & DIRECTORIES
# ==========================================
# Result folders per model; replace these to plot runs stored elsewhere.
RESULTS_ROOT = "/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM/Pascal_VOC/results"

PAIRINGS = {
    "Pascal-VOC": ("SAMv1", "SAMv2"),
    # "ISIC-2016": ("MSDUNet", "TransFuse"),
    # "Synapse": ("D-LKA", "TransDeeplab"),
    # "XCAD": ("CDARL", "DARL"),
}
ICOC_DIRS = {     # prism.py -> run_1/<img>/<img>_analysis.json with in/out_of_context_analysis
    "SAMv1": f"{RESULTS_ROOT}/sam1/prism_out_icoc",
    "SAMv2": f"{RESULTS_ROOT}/sam2/prism_out_icoc",
}
GLOBAL_DIRS = {   # prism_global.py -> run_1/<img>/<img>_analysis.json with global_sota_analysis
    "SAMv1": f"{RESULTS_ROOT}/sam1/prism_out_global",
    "SAMv2": f"{RESULTS_ROOT}/sam2/prism_out_global",
}
ID_DIRS = {       # prism_sota.py -> <img>/<img>_analysis.json with prob_guided_analysis
    "SAMv1": f"{RESULTS_ROOT}/sam1/prism_out_insertion_deletion",
    "SAMv2": f"{RESULTS_ROOT}/sam2/prism_out_insertion_deletion",
}
RUN_SUBDIR = "run_1"

OUT_DIR = f"{RESULTS_ROOT}/plots"

NOISES = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
ID_BASELINES = ['black', 'mean']

# Continuous x-axis = noise_level / NOISE_MAX, the last level of the schedule in prism.py /
# prism_global.py (np.arange(0.05, 3.1, 0.05)). IC/OC curves that stop early (SSIM < 0.70)
# end where they stopped instead of being stretched to the full axis.
NOISE_MAX = 3.05

# Rank-stability measure between the step-0 superpixel ranking and the step-k ranking:
#   'footrule'  1 - sum|rank_k - rank_0| / floor(N^2/2)  (same as rank_perplexity in the JSONs)
#   'rbo'       extrapolated rank-biased overlap (Webber et al. 2010), top-weighted by RBO_P
RANK_METRIC = 'rbo'
RBO_P = 0.9

# ==========================================
# EXTRACTION & MATH FUNCTIONS
# ==========================================
def clean_noise_name(raw_name):
    if raw_name == 'salt_pepper': return 'Salt & Pepper'
    return str(raw_name).strip().replace('_', ' ').title()

def get_interpolated(x_raw, y_raw):
    """Resample a curve onto COMMON_X; NaN outside the range the curve actually covers."""
    if len(x_raw) < 2: return np.full_like(COMMON_X, np.nan)
    x_raw, indices = np.unique(np.asarray(x_raw, dtype=float), return_index=True)
    y_raw = np.asarray(y_raw, dtype=float)[indices]
    y = np.interp(COMMON_X, x_raw, y_raw)
    y[(COMMON_X < x_raw[0] - 1e-9) | (COMMON_X > x_raw[-1] + 1e-9)] = np.nan
    return y

def curve_auc(x, y):
    """Area under the curve over the range it covers, divided by that range (= mean height)."""
    if len(x) < 2 or x[-1] <= x[0]: return np.nan
    return trapezoid(y, x) / (x[-1] - x[0])

def prob_ranking(superpixels):
    """Superpixel ids ordered by probability, ties by area (same ordering as the PRISM scripts)."""
    ordered = sorted(superpixels, key=lambda sp: (sp['mean_prob'], sp['area']), reverse=True)
    return [sp['sp_id'] for sp in ordered]

def footrule_stability(list_0, list_k):
    N = len(list_0)
    max_jumps = math.floor(N ** 2 / 2.0)
    pos_k = {sp_id: r for r, sp_id in enumerate(list_k)}
    jumps = sum(abs(pos_k.get(sp_id, r) - r) for r, sp_id in enumerate(list_0))
    return 1.0 - jumps / max_jumps if max_jumps > 0 else 1.0

def rbo_ext(list_0, list_k, p=RBO_P):
    """Extrapolated RBO for two rankings of the same depth."""
    k = min(len(list_0), len(list_k))
    if k == 0: return np.nan
    seen_0, seen_k, overlap, weighted = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        a, b = list_0[d - 1], list_k[d - 1]
        if a == b: overlap += 1
        else:
            overlap += (a in seen_k) + (b in seen_0)
        seen_0.add(a); seen_k.add(b)
        weighted += (overlap / d) * p ** d
    return (overlap / k) * p ** k + (1 - p) / p * weighted

def stability_trajectory(levels):
    """Rank stability of every step against step 0 (1.0 = ranking unchanged)."""
    if len(levels) < 2 or not levels[0].get('superpixels'): return []
    list_0 = prob_ranking(levels[0]['superpixels'])
    if len(list_0) < 2: return []
    measure = rbo_ext if RANK_METRIC == 'rbo' else footrule_stability
    return [measure(list_0, prob_ranking(lvl['superpixels'])) for lvl in levels]

def load_jsons(root, pattern, skipped, tag):
    root = Path(root)
    if not root.exists():
        skipped[f"{tag}: folder missing ({root})"] += 1
        return
    for p in sorted(root.glob(pattern)):
        try:
            with open(p, 'r') as f: data = json.load(f)
        except Exception as e:
            skipped[f"{tag}: unreadable JSON ({type(e).__name__})"] += 1
            continue
        yield data.get('image_info', {}).get('name', p.stem.replace('_analysis', '')), data

# ==========================================
# DATA COLLECTION
# ==========================================
def collect_model(model, curves, aucs, skipped):
    """Fill curves[(condition, series)][model] with resampled curves and aucs with per-image AUC rows."""
    def add(family, condition, image_id, x, perf, stab):
        curves[(condition, f'{family}_Perf')][model].append(get_interpolated(x, perf))
        aucs.append({'model': model, 'image_id': image_id, 'family': family, 'condition': condition,
                     'measure': 'IoU AUC', 'auc': curve_auc(x, perf)})
        if len(stab) == len(x):
            curves[(condition, f'{family}_Stab')][model].append(get_interpolated(x, stab))
            aucs.append({'model': model, 'image_id': image_id, 'family': family, 'condition': condition,
                         'measure': f'Stability AUC ({RANK_METRIC})', 'auc': curve_auc(x, stab)})

    def continuous(levels):
        x = [float(lvl['noise_level']) / NOISE_MAX for lvl in levels]
        return x, [float(lvl['global_iou']) / 100.0 for lvl in levels], stability_trajectory(levels)

    # --- I/D (fraction perturbed / revealed) ---
    for image_id, data in load_jsons(ID_DIRS.get(model, ''), "*/*_analysis.json", skipped, f"{model} I/D"):
        trajectories = data.get('prob_guided_analysis', {}).get('trajectories', {})
        for baseline in ID_BASELINES:
            traj = trajectories.get(baseline, {}).get('prob_ranked')
            if traj is None:
                skipped[f"{model} I/D: missing baseline '{baseline}'"] += 1
                continue
            for family, key in [('Deletion', 'deletion'), ('Insertion', 'insertion')]:
                lvls = traj.get(key, [])
                if len(lvls) < 2: continue
                x = [float(s['fraction_perturbed']) for s in lvls]
                add(family, baseline, image_id, x, [float(s['global_iou']) / 100.0 for s in lvls], stability_trajectory(lvls))

    # --- IC / OC (noise severity) + S-Factor ---
    for image_id, data in load_jsons(ICOC_DIRS.get(model, ''), f"{RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} IC/OC"):
        ic = data.get('in_context_analysis', {}).get('level_data_per_noise', {})
        oc = data.get('out_of_context_analysis', {}).get('level_data_per_noise', {})
        for noise in NOISES:
            if noise not in ic or noise not in oc:
                skipped[f"{model} IC/OC: missing noise '{noise}'"] += 1
                continue
            ic_lvls, oc_lvls = ic[noise]['level_data'], oc[noise]['level_data']
            for family, lvls in [('IC', ic_lvls), ('OC', oc_lvls)]:
                if len(lvls) > 1: add(family, noise, image_id, *continuous(lvls))
            n = min(len(ic_lvls), len(oc_lvls))
            aucs.append({'model': model, 'image_id': image_id, 'family': 'S-Factor', 'condition': noise,
                         'measure': 'S-Factor', 'auc': calculate_holistic_s_factor(ic_lvls[:n], oc_lvls[:n])})

    # --- Global (noise severity) ---
    for image_id, data in load_jsons(GLOBAL_DIRS.get(model, ''), f"{RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} Global"):
        gl = data.get('global_sota_analysis', {}).get('level_data_per_noise', {})
        for noise in NOISES:
            if noise not in gl:
                skipped[f"{model} Global: missing noise '{noise}'"] += 1
                continue
            lvls = gl[noise]['level_data']
            if len(lvls) > 1: add('Global', noise, image_id, *continuous(lvls))

# ==========================================
# PLOTTING ENGINE
# ==========================================
def plot_panel(ax, title, xlabel, ylabel, curves, condition, models, series, colors, styles):
    plotted = False
    for s in series:
        for model in models:
            matrix = np.array([arr for arr in curves[(condition, s)][model] if not np.isnan(arr).all()])
            if len(matrix) == 0: continue
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # columns past every curve's end are all-NaN
                mean_curve = np.nanmean(matrix, axis=0)
            ax.plot(COMMON_X, mean_curve, label=f'{model} ({s.split("_")[0]}, n={len(matrix)})',
                    color=colors[s], linestyle=styles[model], linewidth=2.5)
            plotted = True
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    if plotted: ax.legend(loc='best', fontsize=9, ncol=2)

def generate_curve_report(dataset_name, models, curves, pdf):
    styles = dict(zip(models, ['-', '--', ':', '-.']))
    stab_label = f"Rank Stability ({RANK_METRIC.upper()}, 1.0 = unchanged)"
    total_rows = len(ID_BASELINES) + len(NOISES)
    fig, axes = plt.subplots(total_rows, 2, figsize=(16, 5 * total_rows), squeeze=False)
    fig.suptitle(f'{dataset_name} Curves: {" vs ".join(models)}', fontsize=20, fontweight='bold',
                 y=1.0 - (0.05 / total_rows))

    id_colors = {'Insertion_Perf': '#2ca02c', 'Deletion_Perf': '#d62728', 'Insertion_Stab': '#2ca02c', 'Deletion_Stab': '#d62728'}
    for row, baseline in enumerate(ID_BASELINES):
        name = baseline.capitalize()
        plot_panel(axes[row, 0], f'Standard I/D Performance ({name} Canvas)', 'Fraction Perturbed / Revealed', 'Normalized IoU',
                   curves, baseline, models, ['Insertion_Perf', 'Deletion_Perf'], id_colors, styles)
        plot_panel(axes[row, 1], f'Standard I/D Rank Stability ({name} Canvas)', 'Fraction Perturbed / Revealed', stab_label,
                   curves, baseline, models, ['Insertion_Stab', 'Deletion_Stab'], id_colors, styles)

    cont_colors = {'Global_Perf': '#2ca02c', 'IC_Perf': '#1f77b4', 'OC_Perf': '#9467bd',
                   'Global_Stab': '#2ca02c', 'IC_Stab': '#1f77b4', 'OC_Stab': '#9467bd'}
    xlabel = f'Noise Level / {NOISE_MAX}'
    for row, noise in enumerate(NOISES, start=len(ID_BASELINES)):
        name = clean_noise_name(noise)
        plot_panel(axes[row, 0], f'Continuous Performance ({name})', xlabel, 'Normalized IoU',
                   curves, noise, models, ['Global_Perf', 'IC_Perf', 'OC_Perf'], cont_colors, styles)
        plot_panel(axes[row, 1], f'Continuous Rank Stability ({name})', xlabel, stab_label,
                   curves, noise, models, ['Global_Stab', 'IC_Stab', 'OC_Stab'], cont_colors, styles)

    plt.tight_layout(rect=[0, 0, 1, 1.0 - (0.03 / total_rows)])
    pdf.savefig(fig)
    plt.close(fig)

def generate_auc_report(dataset_name, df, pdf):
    """Per-image AUC distributions: one page per measure, one panel per family, hue = model."""
    family_order = ['Deletion', 'Insertion', 'Global', 'IC', 'OC', 'S-Factor']
    for measure in df['measure'].unique():
        m_df = df[df['measure'] == measure].copy()
        m_df['condition'] = m_df['condition'].map(lambda c: c if c in ID_BASELINES else clean_noise_name(c))
        families = [f for f in family_order if f in m_df['family'].unique()]
        g = sns.catplot(data=m_df, x='condition', y='auc', hue='model', col='family', col_order=families,
                        kind='box', sharex=False, sharey=(measure != 'S-Factor'), col_wrap=min(3, len(families)),
                        height=4.5, aspect=1.2, showfliers=False)
        for ax, fam in zip(g.axes.flat, families):
            sns.stripplot(data=m_df[m_df['family'] == fam], x='condition', y='auc', hue='model', dodge=True,
                          size=2.5, alpha=0.5, palette='dark:black', legend=False, ax=ax)
            ax.tick_params(axis='x', rotation=25)
            ax.set_xlabel('')
        g.set_titles('{col_name}')
        g.set_ylabels(measure)
        g.figure.suptitle(f'{dataset_name}: Per-Image {measure}', fontsize=16, fontweight='bold', y=1.03)
        pdf.savefig(g.figure, bbox_inches='tight')
        plt.close(g.figure)

# ==========================================
# MASTER EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating reports for {len(PAIRINGS)} dataset(s) | rank metric: {RANK_METRIC}\n")
    for dataset, models in PAIRINGS.items():
        print(f"Processing Dataset: {dataset} ({' vs '.join(models)})")
        curves, aucs, skipped = defaultdict(lambda: defaultdict(list)), [], Counter()
        for model in models:
            collect_model(model, curves, aucs, skipped)
        for reason, count in sorted(skipped.items()): print(f"  skipped {count:5d} x {reason}")

        df = pd.DataFrame(aucs).dropna(subset=['auc'])
        tag = dataset.replace(' ', '_')
        df.to_csv(os.path.join(OUT_DIR, f"{tag}_Per_Image_AUC.csv"), index=False)
        summary = df.groupby(['family', 'condition', 'measure', 'model'])['auc'].agg(['mean', 'std', 'count']).round(4)
        summary.to_csv(os.path.join(OUT_DIR, f"{tag}_AUC_Summary.csv"))

        pdf_path = os.path.join(OUT_DIR, f"{tag}_Report.pdf")
        with PdfPages(pdf_path) as pdf:
            generate_curve_report(dataset, list(models), curves, pdf)
            if not df.empty: generate_auc_report(dataset, df, pdf)
        print(f"  -> Saved: {pdf_path} (+ per-image and summary AUC CSVs)")
    print(f"\nAll reports saved to '{OUT_DIR}'.")

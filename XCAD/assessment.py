import os
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata, gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
# Result folders per model; edit PRISM_ROOT (or override the dicts below) to evaluate runs stored elsewhere.
PRISM_ROOT = "/cluster/home/bhupenk/bhupender/current_projects/2026/PRISM"

# model -> (dataset folder, results subfolder, dataset name), as written by the PRISM scripts / slurm files
MODELS = {
    "DARL": ("XCAD", "darl", "XCAD"),
    "CDARL": ("XCAD", "cdarl", "XCAD"),
}
ICOC_DIRS = {m: f"{PRISM_ROOT}/{ds}/results/{sub}/prism_out_icoc" for m, (ds, sub, _) in MODELS.items()}                  # prism.py
GLOBAL_DIRS = {m: f"{PRISM_ROOT}/{ds}/results/{sub}/prism_out_global" for m, (ds, sub, _) in MODELS.items()}              # prism_global.py
ID_DIRS = {m: f"{PRISM_ROOT}/{ds}/results/{sub}/prism_out_insertion_deletion" for m, (ds, sub, _) in MODELS.items()}      # prism_sota.py
RUN_SUBDIR = "run_1"   # dataset-run folder written by prism.py / prism_global.py

OUT_DIR = f"{PRISM_ROOT}/XCAD/results/meta_evaluation"

# Score families (paradigms) to evaluate
PARADIGMS = {
    'S-Factor (IC/OC)': True,
    'Global Continuous AUC': True,
    'Standard I/D AUC (Deletion)': True,
    'Standard I/D AUC (Insertion)': True,
}

# Raters: noise perturbations for IC/OC + global, perturbation baselines for I/D
NOISES = ['poisson', 'gaussian', 'speckle', 'salt_pepper', 'gaussian_blur', 'rician']
ID_BASELINES = ['mean', 'black']

# Noise-level ablation: each fraction truncates a curve to its first part
# (noise steps for IC/OC + global, fraction of superpixels perturbed for I/D).
# 1.0 = full curve.
LEVEL_FRACTIONS = [0.3, 0.5, 1.0]

# Rater schemes (each enabled scheme produces its own rows in the results table):
#   'perturbation'           raters = perturbations; one result per level fraction
#   'perturbation_x_level'   raters = every (perturbation, level fraction) pair; one result
#   'level_within_perturbation'  raters = level fractions; one result per perturbation
RATER_SCHEMES = ['perturbation', 'perturbation_x_level', 'level_within_perturbation']

PRINT_RANK_LISTS = False   # print the per-image rank matrix of every evaluation

# ==========================================
# 2. MATH ENGINES
# ==========================================
GAMMA = 1.0
LAMBDA_P = 1.0

def calculate_holistic_s_factor(ic_levels, oc_levels):
    IC_vals = [float(lvl.get("discounted_utility", lvl.get("global_iou", 0.0))) for lvl in ic_levels]
    OC_vals = [float(lvl.get("discounted_utility", lvl.get("global_iou", 0.0))) for lvl in oc_levels]

    N_min = min(len(IC_vals), len(OC_vals))
    if N_min < 2: return np.nan

    IC_0, OC_0 = max(1e-6, IC_vals[0]), max(1e-6, OC_vals[0])
    W_ic, A_ic = 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1)
        w_k = math.exp(-GAMMA * eta_k)
        A_ic += w_k * (IC_vals[k] / IC_0)
        W_ic += w_k
    norm_A_ic = min(1.0, A_ic / W_ic if W_ic > 0 else 0.0)

    W_oc, A_oc, weighted_drift = 0.0, 0.0, 0.0
    for k in range(1, N_min):
        eta_k = k / (N_min - 1)
        w_k = math.exp(-GAMMA * eta_k)
        oc_hat = OC_vals[k] / OC_0
        A_oc += w_k * oc_hat
        weighted_drift += w_k * ((oc_hat - 1.0) ** 2)
        W_oc += w_k
    norm_A_oc = min(1.0, A_oc / W_oc if W_oc > 0 else 0.0)

    G_gap = max(0.0, norm_A_oc - norm_A_ic)
    E_drift = math.sqrt(weighted_drift / W_oc) if W_oc > 0 else 0.0

    return math.exp(-LAMBDA_P * (max(0.0, 1.0 - G_gap) + E_drift)) * 100.0

def trapezoid(y, x):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

def truncate_levels(levels, frac):
    """First part of a noise curve: steps 0..idx, idx = frac of the last step (at least 1)."""
    idx = max(1, int(round(frac * (len(levels) - 1))))
    return levels[:idx + 1]

def global_auc(levels, frac):
    """Area under global IoU vs normalised noise step, truncated at frac."""
    if len(levels) < 2: return np.nan
    sub = truncate_levels(levels, frac)
    x = [k / (len(levels) - 1) for k in range(len(sub))]
    y = [float(step['global_iou']) / 100.0 for step in sub]
    return trapezoid(y, x)

def id_auc(traj, frac, saved_auc):
    """Area under global IoU vs fraction perturbed, truncated at fraction_perturbed <= frac."""
    if frac >= 1.0 and saved_auc is not None: return float(saved_auc)
    sub = [step for step in traj if float(step['fraction_perturbed']) <= frac + 1e-9]
    if len(sub) < 2: return np.nan
    return trapezoid([float(s['global_iou']) / 100.0 for s in sub], [float(s['fraction_perturbed']) for s in sub])

# ==========================================
# 3. DATA PARSER
# ==========================================
def frac_label(frac):
    return f"{int(round(frac * 100))}%"

def labelled(items, fmt=str):
    """Pair each configured rater with a unique label; repeated entries become separate raters (#1, #2, ...)."""
    base = [fmt(item) for item in items]
    counts, seen, out = Counter(base), Counter(), []
    for item, label in zip(items, base):
        if counts[label] > 1:
            seen[label] += 1
            label = f"{label}#{seen[label]}"
        out.append((item, label))
    return out

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
        image_id = data.get('image_info', {}).get('name', p.stem.replace('_analysis', ''))
        yield image_id, data

def extract_all_scores():
    records, skipped = [], Counter()
    noises, baselines = labelled(NOISES), labelled(ID_BASELINES)
    levels = labelled(LEVEL_FRACTIONS, frac_label)
    def add(model, image_id, paradigm, perturbation, level, score):
        if score is not None and not np.isnan(score):
            records.append({'model': model, 'image_id': image_id, 'paradigm': paradigm,
                            'perturbation': perturbation, 'level': level, 'score': score})

    # --- IC/OC S-FACTOR ---
    if PARADIGMS['S-Factor (IC/OC)']:
        for model, root in ICOC_DIRS.items():
            for image_id, data in load_jsons(root, f"{RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} IC/OC"):
                try:
                    ic = data['in_context_analysis']['level_data_per_noise']
                    oc = data['out_of_context_analysis']['level_data_per_noise']
                except KeyError as e:
                    skipped[f"{model} IC/OC: missing key {e}"] += 1
                    continue
                for noise, noise_lbl in noises:
                    if noise not in ic or noise not in oc:
                        skipped[f"{model} IC/OC: missing noise '{noise}'"] += 1
                        continue
                    ic_lvls, oc_lvls = ic[noise]['level_data'], oc[noise]['level_data']
                    n = min(len(ic_lvls), len(oc_lvls))
                    for frac, lvl_lbl in levels:
                        add(model, image_id, 'S-Factor (IC/OC)', noise_lbl, lvl_lbl,
                            calculate_holistic_s_factor(truncate_levels(ic_lvls[:n], frac), truncate_levels(oc_lvls[:n], frac)))

    # --- GLOBAL CONTINUOUS AUC ---
    if PARADIGMS['Global Continuous AUC']:
        for model, root in GLOBAL_DIRS.items():
            for image_id, data in load_jsons(root, f"{RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} Global"):
                try:
                    gl = data['global_sota_analysis']['level_data_per_noise']
                except KeyError as e:
                    skipped[f"{model} Global: missing key {e}"] += 1
                    continue
                for noise, noise_lbl in noises:
                    if noise not in gl:
                        skipped[f"{model} Global: missing noise '{noise}'"] += 1
                        continue
                    for frac, lvl_lbl in levels:
                        add(model, image_id, 'Global Continuous AUC', noise_lbl, lvl_lbl, global_auc(gl[noise]['level_data'], frac))

    # --- STANDARD I/D AUC ---
    want_del, want_ins = PARADIGMS['Standard I/D AUC (Deletion)'], PARADIGMS['Standard I/D AUC (Insertion)']
    if want_del or want_ins:
        for model, root in ID_DIRS.items():
            for image_id, data in load_jsons(root, "*/*_analysis.json", skipped, f"{model} I/D"):
                try:
                    trajectories = data['prob_guided_analysis']['trajectories']
                except KeyError as e:
                    skipped[f"{model} I/D: missing key {e}"] += 1
                    continue
                for baseline, base_lbl in baselines:
                    traj = trajectories.get(baseline, {}).get('prob_ranked')
                    if traj is None:
                        skipped[f"{model} I/D: missing baseline '{baseline}'"] += 1
                        continue
                    for frac, lvl_lbl in levels:
                        if want_del:
                            add(model, image_id, 'Standard I/D AUC (Deletion)', base_lbl, lvl_lbl, id_auc(traj['deletion'], frac, traj.get('auc_del')))
                        if want_ins:
                            add(model, image_id, 'Standard I/D AUC (Insertion)', base_lbl, lvl_lbl, id_auc(traj['insertion'], frac, traj.get('auc_ins')))

    if skipped:
        print("\n>>> SKIPPED DURING EXTRACTION")
        for reason, count in sorted(skipped.items()): print(f"  {count:5d} x {reason}")
    return pd.DataFrame(records)

# ==========================================
# 4. KENDALL'S W & SPEARMAN'S RHO ENGINES
# ==========================================
def compute_kendalls_w(rank_matrix):
    """Kendall's W with tie correction; rows = images (subjects), columns = raters."""
    n, m = rank_matrix.shape
    if n < 2 or m < 2: return np.nan
    R_i = np.sum(rank_matrix, axis=1)
    S = np.sum((R_i - np.mean(R_i)) ** 2)
    ties = sum(np.sum(c ** 3 - c) for c in (np.unique(rank_matrix[:, j], return_counts=True)[1] for j in range(m)))
    denom = m ** 2 * (n ** 3 - n) - m * ties
    return (12 * S) / denom if denom > 0 else np.nan

def compute_mean_spearman(df_pivot):
    if df_pivot.shape[1] < 2: return np.nan
    corr_matrix = df_pivot.corr(method='spearman').values
    idx = np.triu_indices_from(corr_matrix, k=1)
    return np.nanmean(corr_matrix[idx])

def rater_groups(p_df):
    """Yield (scheme, group, rater-labelled frame) for every enabled rater scheme."""
    if 'perturbation' in RATER_SCHEMES:
        for _, lvl_lbl in labelled(LEVEL_FRACTIONS, frac_label):
            sub = p_df[p_df['level'] == lvl_lbl]
            yield 'perturbation', f"levels {lvl_lbl}", sub.assign(rater=sub['perturbation'])
    if 'perturbation_x_level' in RATER_SCHEMES:
        yield 'perturbation_x_level', 'all', p_df.assign(rater=p_df['perturbation'] + '@' + p_df['level'])
    if 'level_within_perturbation' in RATER_SCHEMES:
        for pert in p_df['perturbation'].unique():
            sub = p_df[p_df['perturbation'] == pert]
            yield 'level_within_perturbation', pert, sub.assign(rater=sub['level'])

def run_image_ranking_consistency(df_all):
    results = []
    print("\n>>> DIAGNOSTIC: Image counts surviving the rater grid dropna")
    for model in df_all['model'].unique():
        m_df = df_all[df_all['model'] == model]
        for paradigm in m_df['paradigm'].unique():
            p_df = m_df[m_df['paradigm'] == paradigm]
            for scheme, group, r_df in rater_groups(p_df):
                pivot = r_df.pivot_table(index='image_id', columns='rater', values='score')
                pivot_clean = pivot.dropna(axis=0, how='any').sort_index()
                tag = f"[{model}] {paradigm} | {scheme} | {group}"
                print(f"{tag}: {len(pivot)} initial -> {len(pivot_clean)} overlapping images, {pivot_clean.shape[1]} raters")
                if len(pivot_clean) < 2 or pivot_clean.shape[1] < 2:
                    print(f"   -> SKIPPING (not enough overlap)")
                    continue

                # Rank images within each rater (highest score = rank 1)
                rank_matrix = np.column_stack([rankdata(-pivot_clean[c].values) for c in pivot_clean.columns])
                if PRINT_RANK_LISTS:
                    print(f"\n=== RAW RANK LISTS FOR {tag} ===")
                    print(pd.DataFrame(rank_matrix, index=pivot_clean.index, columns=pivot_clean.columns).to_string())

                results.append({
                    'Dataset': MODELS[model][2],
                    'Model': model,
                    'Paradigm': paradigm,
                    'Scheme': scheme,
                    'Group': group,
                    'Kendalls_W': compute_kendalls_w(rank_matrix),
                    'Spearmans_Rho': compute_mean_spearman(pivot_clean),
                    'Num_Images': len(pivot_clean),
                    'Num_Raters': pivot_clean.shape[1],
                    'Raters': ', '.join(map(str, pivot_clean.columns)),
                })
    return pd.DataFrame(results).round(3)

# ==========================================
# 5. PLOTTING
# ==========================================
PARADIGM_PALETTE = {
    'Standard I/D AUC (Deletion)': '#d62728',
    'Standard I/D AUC (Insertion)': '#ff7f0e',
    'Global Continuous AUC': '#2ca02c',
    'S-Factor (IC/OC)': '#1f77b4'
}

def plot_model_scheme(model, scheme, s_df):
    """Side-by-side W and rho bars; x = paradigm, hue = group when a scheme has several groups."""
    multi_group = s_df['Group'].nunique() > 1
    fig, axes = plt.subplots(1, 2, figsize=(max(15, 2.5 * s_df['Paradigm'].nunique() * (1 + multi_group)), 7))
    fig.suptitle(f"Image Ranking Consistency: {model} ({scheme})", fontsize=18, fontweight='bold', y=1.02)
    for ax, metric, title, ylabel in [
        (axes[0], 'Kendalls_W', "Kendall's W", "Score (1.0 = Perfect Rank Agreement)"),
        (axes[1], 'Spearmans_Rho', "Spearman's Rho (Mean Pairwise)", "Correlation (1.0 = Perfect Agreement)"),
    ]:
        if multi_group:
            sns.barplot(data=s_df, x='Paradigm', y=metric, hue='Group', edgecolor='black', linewidth=1.2, ax=ax)
            ax.legend(title='Group', fontsize=9)
        else:
            sns.barplot(data=s_df, x='Paradigm', y=metric, hue='Paradigm', palette=PARADIGM_PALETTE,
                        edgecolor='black', linewidth=1.2, dodge=False, ax=ax)
            if ax.legend_: ax.legend_.remove()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8 if multi_group else 11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylim(min(0, s_df[metric].min() - 0.05), 1.15)
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, f"Fig_{model}_{scheme}_Ranking_Consistency.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {pdf_path}")

SCORE_LEVEL = 1.0   # level fraction shown in the per-model score distributions (1.0 = full curve)

def draw_half_violin(ax, x, values, side, color, width=0.4):
    """Half violin at x; side = -1 draws to the left, +1 to the right (both sides = full candle)."""
    if values.size == 0: return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    lo, hi = values.min(), values.max()
    tick = lambda: ax.plot([x, x + side * width * 0.6], [med, med], color=color, lw=2.5, solid_capstyle="butt")
    if values.size < 2 or hi - lo < 1e-9: return tick()
    try:
        kde = gaussian_kde(values)
    except np.linalg.LinAlgError:          # (near-)singular: treat as constant
        return tick()
    ys = np.linspace(lo, hi, 256)
    dens = kde(ys)
    scale = width / dens.max()
    ax.fill_betweenx(ys, x, x + side * dens * scale, facecolor=color, alpha=0.55, edgecolor=color, lw=1.0)
    for y, ls in ((q1, "--"), (med, "-"), (q3, "--")):
        ax.plot([x, x + side * float(kde(y)[0]) * scale], [y, y], color="#222222", ls=ls, lw=1.4 if ls == "-" else 0.9)

def perturbation_name(label):
    base, _, rep = str(label).partition('#')
    name = 'Salt & Pepper' if base == 'salt_pepper' else base.replace('_', ' ').title()
    return f"{name}#{rep}" if rep else name

def plot_model_scores(model, m_df):
    """Per-image score distributions at SCORE_LEVEL: one panel per paradigm, x = perturbation, one candle per rater."""
    lvl = frac_label(SCORE_LEVEL)
    m_df = m_df[m_df['level'] == lvl]
    paradigms = [p for p in PARADIGMS if p in set(m_df['paradigm'])]
    if not paradigms:
        print(f"  -> No scores at level {lvl} for {model}; skipping score distributions.")
        return
    perts = {p: list(dict.fromkeys(m_df.loc[m_df['paradigm'] == p, 'perturbation'])) for p in paradigms}
    widths = [len(perts[p]) for p in paradigms]
    fig, axes = plt.subplots(1, len(paradigms), figsize=(0.95 * sum(widths) + 3, 5), squeeze=False,
                             gridspec_kw={'width_ratios': widths})
    for ax, paradigm in zip(axes[0], paradigms):
        p_df, color = m_df[m_df['paradigm'] == paradigm], PARADIGM_PALETTE.get(paradigm, '#555555')
        for x, pert in enumerate(perts[paradigm]):
            v = p_df.loc[p_df['perturbation'] == pert, 'score'].dropna().values
            draw_half_violin(ax, x, v, -1, color)
            draw_half_violin(ax, x, v, +1, color)
        ax.set_xticks(range(len(perts[paradigm])))
        ax.set_xticklabels([perturbation_name(p) for p in perts[paradigm]], rotation=30, ha='right')
        ax.set_xlim(-0.6, len(perts[paradigm]) - 0.4)
        ax.set_title(paradigm.replace(' (', '\n(') if len(perts[paradigm]) < 4 else paradigm, fontsize=12, fontweight='bold')
        ax.set_ylabel("Score" if paradigm.startswith('S-Factor') else "AUC", fontsize=11)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    n_img = m_df['image_id'].nunique()
    fig.suptitle(f"Per-Image Score Distributions: {model} (levels {lvl}, {n_img} images)", fontsize=16, fontweight='bold', y=1.0)
    fig.legend(handles=[Line2D([], [], color="#222222", lw=1.4, label="median"),
                        Line2D([], [], color="#222222", lw=0.9, ls="--", label="quartiles")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf_path = os.path.join(OUT_DIR, f"Fig_{model}_Score_Distributions.pdf")
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {pdf_path}")

# ==========================================
# 6. MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f">>> Extracting Data | Noises: {NOISES} | I/D baselines: {ID_BASELINES} | "
          f"Level fractions: {LEVEL_FRACTIONS} | Schemes: {RATER_SCHEMES}")
    df_all = extract_all_scores()

    if df_all.empty:
        print("Error: No data extracted. Check paths and JSON structure.")
    else:
        df_all.to_csv(os.path.join(OUT_DIR, "Per_Image_Scores.csv"), index=False)

        print(f"\n>>> Generating per-model score distributions (levels {frac_label(SCORE_LEVEL)})...")
        for model, m_df in df_all.groupby('model', sort=False):
            plot_model_scores(model, m_df)

        print("\n>>> Calculating Kendall's W and Spearman's Rho...")
        w_results = run_image_ranking_consistency(df_all)

        if w_results.empty:
            print("Error: DataFrame returned empty after dropna checks. See diagnostic printouts above.")
        else:
            w_results.to_csv(os.path.join(OUT_DIR, "Table3_Image_Ranking_Consistency.csv"), index=False)
            print("\n=== INTERNAL IMAGE RANKING CONSISTENCY ===")
            print(w_results.drop(columns='Raters').to_string(index=False))

            print("\n>>> Generating Model Plots (one per model and rater scheme)...")
            for (model, scheme), s_df in w_results.groupby(['Model', 'Scheme'], sort=False):
                plot_model_scheme(model, scheme, s_df)

            print(f"\n[DONE] Saved CSVs and plots to: {OUT_DIR}")

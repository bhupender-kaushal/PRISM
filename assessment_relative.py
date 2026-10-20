"""
Relative-measure assessment: every method scored on RELATIVE change w.r.t. the clean image
==========================================================================================
Builds on assessment_global.py (models, result folders, level fractions, rater schemes,
Kendall's W / Spearman engines and figures) and runs it twice:

  raw/       the methods as assessment_global.py scores them
             (S-Factor, Global AUC on IoU/100, I/D AUC on IoU/100)
  relative/  every method measured as relative change:
             S-Factor (IC/OC)           unchanged (already relative to step 0)
             S-Factor new (utility)     100 * D_ic * I_oc on discounted_utility
             S-Factor new (IoU)         100 * D_ic * I_oc on global_iou
             Global AUC (relative)      area under min(1, IoU_k / IoU_0)
             I/D Deletion (relative)    area under min(1, IoU / clean IoU)   (clean = deletion step 0)
             I/D Insertion (relative)   area under min(1, IoU / clean IoU)

New score, per image and noise (w_k = exp(-k/(N-1)), k = 1..N-1, as in the S-factor):
    D_ic = 1 - sum_k w_k * IC_k/IC_0 / sum_k w_k          drop of the IC prediction under object noise (rewarded)
    I_oc = 1 - sum_k w_k * |OC_k/OC_0 - 1| / sum_k w_k    invariance of the IC prediction to background noise
    S    = 100 * clip(D_ic, 0, 1) * clip(I_oc, 0, 1)

Outputs in OUT_DIR:
    raw/, relative/                  Per_Image_Scores.csv, Table3_Image_Ranking_Consistency.csv,
                                     Model_Separation.csv and all figures (PDF + PNG), incl.
                                     Fig_All_Datasets_Model_Comparison_<raw|relative>.pdf: rows = datasets,
                                     columns = methods, split candle = the dataset's two models
    Summary_Kendalls_W.csv           W per paradigm x model x level (scheme 'perturbation'), raw and relative
    Summary_Model_Separation.csv     paired rank-biserial r per paradigm x dataset x level, raw and relative
    Summary_Model_MannWhitney.csv    unpaired Mann-Whitney U per model pair x paradigm x level x perturbation
                                     ('all' = per-image mean over perturbations), effect size, Holm-adjusted p
    Fig_Summary_MannWhitney_Overview.pdf          heatmap per level: methods x datasets, winner + effect
    Fig_Summary_MannWhitney_Per_Perturbation.pdf  heatmap per dataset (100%): methods x perturbations
"""
import io
import os
import sys
import contextlib
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, wilcoxon, mannwhitneyu

import assessment_global as A

# ==========================================
# 1. CONFIGURATION
# ==========================================
OUT_DIR = f"{A.PRISM_ROOT}/Results/relative_assessment"
SAVE_PNG = True     # also save a PNG next to every PDF figure

REL_PARADIGMS = ['S-Factor (IC/OC)', 'S-Factor new (utility)', 'S-Factor new (IoU)',
                 'Global AUC (relative)', 'I/D Deletion (relative)', 'I/D Insertion (relative)']
REL_PALETTE = {'S-Factor (IC/OC)': '#1f77b4', 'S-Factor new (utility)': '#9467bd', 'S-Factor new (IoU)': '#17becf',
               'Global AUC (relative)': '#2ca02c', 'I/D Deletion (relative)': '#d62728', 'I/D Insertion (relative)': '#ff7f0e'}

# Row order of the summary tables: each raw method next to its relative version
SUMMARY_ORDER = ['S-Factor (IC/OC)', 'S-Factor new (utility)', 'S-Factor new (IoU)',
                 'Global Continuous AUC', 'Global AUC (relative)',
                 'Standard I/D AUC (Deletion)', 'I/D Deletion (relative)',
                 'Standard I/D AUC (Insertion)', 'I/D Insertion (relative)']

if SAVE_PNG:
    _savefig = Figure.savefig
    def _savefig_with_png(self, path, *args, **kwargs):
        _savefig(self, path, *args, **kwargs)
        if str(path).endswith(".pdf"):
            kwargs.pop("format", None)
            _savefig(self, str(path)[:-4] + ".png", *args, **{**kwargs, "dpi": 120})
    Figure.savefig = _savefig_with_png

# ==========================================
# 2. RELATIVE SCORES
# ==========================================
def new_score(ic_levels, oc_levels, field):
    I = np.array([float(l[field]) for l in ic_levels])
    O = np.array([float(l[field]) for l in oc_levels])
    N = min(len(I), len(O))
    if N < 2: return np.nan
    w = np.exp(-np.arange(1, N) / (N - 1))
    d_ic = 1 - np.sum(w * I[1:N] / max(1e-6, I[0])) / w.sum()
    i_oc = 1 - np.sum(w * np.abs(O[1:N] / max(1e-6, O[0]) - 1)) / w.sum()
    return 100 * float(np.clip(d_ic, 0, 1) * np.clip(i_oc, 0, 1))

def global_auc_rel(levels, frac):
    """A.global_auc with IoU divided by the clean (step 0) IoU instead of 100;
    each step's relative IoU is capped at 1, so a rise above the clean IoU is not rewarded."""
    if len(levels) < 2 or float(levels[0]['global_iou']) <= 0: return np.nan
    sub = A.truncate_levels(levels, frac)
    x = [k / (len(levels) - 1) for k in range(len(sub))]
    return A.trapezoid([min(1.0, float(s['global_iou']) / float(levels[0]['global_iou'])) for s in sub], x)

def id_auc_rel(traj, frac, clean_iou):
    """A.id_auc with IoU divided by the clean IoU instead of 100;
    each step's relative IoU is capped at 1, as for the Global AUC."""
    sub = [s for s in traj if float(s['fraction_perturbed']) <= frac + 1e-9]
    if len(sub) < 2 or clean_iou <= 0: return np.nan
    return A.trapezoid([min(1.0, float(s['global_iou']) / clean_iou) for s in sub], [float(s['fraction_perturbed']) for s in sub])

def extract_relative_scores():
    records, skipped = [], Counter()
    noises, baselines = A.labelled(A.NOISES), A.labelled(A.ID_BASELINES)
    levels = A.labelled(A.LEVEL_FRACTIONS, A.frac_label)
    def add(model, image_id, paradigm, perturbation, level, score):
        if score is not None and not np.isnan(score):
            records.append({'model': model, 'image_id': image_id, 'paradigm': paradigm,
                            'perturbation': perturbation, 'level': level, 'score': score})

    for model in A.MODELS:
        for image_id, data in A.load_jsons(A.ICOC_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} IC/OC"):
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
                    ti, to = A.truncate_levels(ic_lvls[:n], frac), A.truncate_levels(oc_lvls[:n], frac)
                    add(model, image_id, 'S-Factor (IC/OC)', noise_lbl, lvl_lbl, A.calculate_holistic_s_factor(ti, to))
                    add(model, image_id, 'S-Factor new (utility)', noise_lbl, lvl_lbl, new_score(ti, to, 'discounted_utility'))
                    add(model, image_id, 'S-Factor new (IoU)', noise_lbl, lvl_lbl, new_score(ti, to, 'global_iou'))

        for image_id, data in A.load_jsons(A.GLOBAL_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} Global"):
            gl = data.get('global_sota_analysis', {}).get('level_data_per_noise', {})
            for noise, noise_lbl in noises:
                if noise not in gl:
                    skipped[f"{model} Global: missing noise '{noise}'"] += 1
                    continue
                for frac, lvl_lbl in levels:
                    add(model, image_id, 'Global AUC (relative)', noise_lbl, lvl_lbl, global_auc_rel(gl[noise]['level_data'], frac))

        for image_id, data in A.load_jsons(A.ID_DIRS[model], "*/*_analysis.json", skipped, f"{model} I/D"):
            trajectories = data.get('prob_guided_analysis', {}).get('trajectories', {})
            for baseline, base_lbl in baselines:
                traj = trajectories.get(baseline, {}).get('prob_ranked')
                if traj is None:
                    skipped[f"{model} I/D: missing baseline '{baseline}'"] += 1
                    continue
                clean_iou = float(traj['deletion'][0]['global_iou'])      # deletion step 0 = unperturbed image
                for frac, lvl_lbl in levels:
                    add(model, image_id, 'I/D Deletion (relative)', base_lbl, lvl_lbl, id_auc_rel(traj['deletion'], frac, clean_iou))
                    add(model, image_id, 'I/D Insertion (relative)', base_lbl, lvl_lbl, id_auc_rel(traj['insertion'], frac, clean_iou))

    if skipped:
        print("\n>>> SKIPPED DURING RELATIVE EXTRACTION")
        for reason, count in sorted(skipped.items()): print(f"  {count:5d} x {reason}")
    return pd.DataFrame(records)

# ==========================================
# 3. MODEL SEPARATION (paired Wilcoxon per dataset)
# ==========================================
def model_separation(df):
    """Paired comparison of each dataset's two models: per-image mean over raters, Wilcoxon signed-rank
    (zero_method='pratt') and matched-pairs rank-biserial r (+1 = first model always higher)."""
    rows = []
    for dataset in dict.fromkeys(ds for _, _, ds in A.MODELS.values()):
        models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset and m in set(df['model'])]
        if len(models) != 2: continue
        a, b = models
        for (paradigm, level), g in df[df['model'].isin(models)].groupby(['paradigm', 'level']):
            pivot = g.groupby(['image_id', 'model'])['score'].mean().unstack()
            if a not in pivot or b not in pivot: continue
            pivot = pivot.dropna()
            d = (pivot[a] - pivot[b]).values
            nz = d[d != 0]
            if len(nz) == 0:
                r, p = 0.0, 1.0
            else:
                rk = rankdata(np.abs(nz))
                w_plus, w_minus = rk[nz > 0].sum(), rk[nz < 0].sum()
                r, p = (w_plus - w_minus) / (w_plus + w_minus), wilcoxon(d, zero_method='pratt').pvalue
            rows.append({'Dataset': dataset, 'Model_A': a, 'Model_B': b, 'Paradigm': paradigm, 'Level': level,
                         'Num_Images': len(d), 'A_Wins': float(np.mean(d > 0)), 'Median_Diff': float(np.median(d)),
                         'Rank_Biserial_r': float(r), 'Wilcoxon_p': float(p)})
    return pd.DataFrame(rows)

def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

def model_mann_whitney(df):
    """Unpaired Mann-Whitney U between each dataset's two models, per paradigm x level, overall
    (per-image mean over raters) and per perturbation. Effect size = rank-biserial 2U/(n1*n2) - 1
    (+1: every score of the first model above every score of the second). Holm correction over all rows."""
    rows = []
    for dataset in dict.fromkeys(ds for _, _, ds in A.MODELS.values()):
        models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset and m in set(df['model'])]
        if len(models) != 2: continue
        a, b = models
        for (paradigm, level), g in df[df['model'].isin(models)].groupby(['paradigm', 'level']):
            groups = [('all', g.groupby(['model', 'image_id'])['score'].mean().reset_index())]
            groups += [(pert, gp) for pert, gp in g.groupby('perturbation', sort=False)]
            for pert, gg in groups:
                xa = gg.loc[gg['model'] == a, 'score'].dropna().values
                xb = gg.loc[gg['model'] == b, 'score'].dropna().values
                if len(xa) < 2 or len(xb) < 2: continue
                if np.ptp(np.concatenate([xa, xb])) == 0:
                    u, p = len(xa) * len(xb) / 2.0, 1.0           # all values identical (e.g. all-zero curves)
                else:
                    u, p = mannwhitneyu(xa, xb, alternative='two-sided')
                eff = 2 * u / (len(xa) * len(xb)) - 1
                rows.append({'Dataset': dataset, 'Model_A': a, 'Model_B': b, 'Paradigm': paradigm, 'Level': level,
                             'Perturbation': pert, 'N_A': len(xa), 'N_B': len(xb),
                             'Median_A': float(np.median(xa)), 'Median_B': float(np.median(xb)),
                             'U': float(u), 'p': float(p), 'Rank_Biserial': float(eff),
                             'Higher_Model': a if eff > 0 else (b if eff < 0 else 'tie')})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['p_Holm'] = holm(out['p'].values)
    return out

def plot_mega_comparison(df, name, paradigms):
    """All datasets in one figure at A.SCORE_LEVEL: rows = datasets, columns = paradigms.
    Split candle per perturbation: left half = the dataset's first model, right half = its second model."""
    lvl = A.frac_label(A.SCORE_LEVEL)
    df = df[df['level'] == lvl]
    pairs = {}
    for dataset in dict.fromkeys(ds for _, _, ds in A.MODELS.values()):
        models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset and m in set(df['model'])]
        if len(models) == 2: pairs[dataset] = models
    paradigms = [p for p in paradigms if p in set(df['paradigm'])]
    if not pairs or not paradigms: return
    perts = {p: list(dict.fromkeys(df.loc[df['paradigm'] == p, 'perturbation'])) for p in paradigms}
    widths = [len(perts[p]) for p in paradigms]
    fig, axes = plt.subplots(len(pairs), len(paradigms), figsize=(0.95 * sum(widths) + 3, 4.2 * len(pairs)),
                             squeeze=False, gridspec_kw={'width_ratios': widths})
    for i, (dataset, models) in enumerate(pairs.items()):
        d_df = df[df['model'].isin(models)]
        for j, paradigm in enumerate(paradigms):
            ax, p_df = axes[i, j], d_df[d_df['paradigm'] == paradigm]
            for x, pert in enumerate(perts[paradigm]):
                for model, side, color in zip(models, (-1, +1), A.MODEL_COLORS):
                    v = p_df.loc[(p_df['perturbation'] == pert) & (p_df['model'] == model), 'score'].dropna().values
                    A.draw_half_violin(ax, x, v, side, color)
            A.format_score_axis(ax, paradigm, perts[paradigm])
            if i > 0: ax.set_title("")
            if i < len(pairs) - 1: ax.set_xticklabels([])
        axes[i, 0].set_ylabel(f"{dataset}\n{axes[i, 0].get_ylabel()}", fontsize=12, fontweight='bold')
        axes[i, 0].legend(handles=[Patch(facecolor=c, alpha=0.55, label=f"{m} ({'left' if k == 0 else 'right'})")
                                   for k, (m, c) in enumerate(zip(models, A.MODEL_COLORS))],
                          loc='upper left', fontsize=8, frameon=True)
    fig.suptitle(f"Per-Image Score Distributions, model vs model ({name}, levels {lvl})", fontsize=18, fontweight='bold', y=1.0)
    fig.legend(handles=[Line2D([], [], color="#222222", lw=1.4, label="median"),
                        Line2D([], [], color="#222222", lw=0.9, ls="--", label="quartiles")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    pdf_path = os.path.join(OUT_DIR, name, f"Fig_All_Datasets_Model_Comparison_{name}.pdf")
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

# Mann-Whitney heatmaps: +1 (first model always higher) = first-model colour, -1 = second-model colour
MWU_CMAP = LinearSegmentedColormap.from_list("mwu", [A.MODEL_COLORS[1], "#ffffff", A.MODEL_COLORS[0]])

def _mwu_cell(ax, x, y, row, short=False):
    """Colour one heatmap cell by the signed effect and write winner / |effect| / significance."""
    eff = float(row['Rank_Biserial'])
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=MWU_CMAP((eff + 1) / 2)))
    star = "*" if row['p_Holm'] < 0.05 else ""
    if row['Higher_Model'] == 'tie':
        text = "tie"
    elif short:
        text = f"{abs(eff):.2f}{star}"
    else:
        text = f"{row['Higher_Model']}\n{abs(eff):.2f}{star}"
    ax.text(x, y, text, ha='center', va='center', fontsize=8 if short else 9,
            fontweight='bold' if star else 'normal', color="#111111")

def _mwu_axes(ax, ncols, nrows, xlabels, ylabels):
    ax.set_xlim(-0.5, ncols - 0.5); ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(range(ncols)); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticks(range(nrows)); ax.set_yticklabels(ylabels, fontsize=10)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    for k in range(1, nrows): ax.axhline(k - 0.5, color="white", lw=2)
    for k in range(1, ncols): ax.axvline(k - 0.5, color="white", lw=2)

def plot_mwu_overview(mwu_all, order, path):
    """One panel per level: rows = methods, columns = datasets, cell = Mann-Whitney result on the
    per-image mean over perturbations (winner, |rank-biserial|, * Holm p < 0.05)."""
    t = mwu_all[mwu_all['Perturbation'] == 'all']
    datasets = list(dict.fromkeys(t['Dataset']))
    pairs = t.drop_duplicates('Dataset').set_index('Dataset')[['Model_A', 'Model_B']]
    methods = [m for m in order if m in set(t['Paradigm'])]
    levels = [l for l in [A.frac_label(f) for f in A.LEVEL_FRACTIONS] if l in set(t['Level'])]
    fig, axes = plt.subplots(1, len(levels), figsize=(4.3 * len(datasets) * len(levels) / 1.6 + 3, 0.62 * len(methods) + 3),
                             squeeze=False, sharey=True)
    xlabels = [f"{ds}\n{pairs.loc[ds, 'Model_A']} vs\n{pairs.loc[ds, 'Model_B']}" for ds in datasets]
    for ax, level in zip(axes[0], levels):
        lt = t[t['Level'] == level].set_index(['Paradigm', 'Dataset'])
        for y, method in enumerate(methods):
            for x, ds in enumerate(datasets):
                if (method, ds) in lt.index:
                    _mwu_cell(ax, x, y, lt.loc[(method, ds)])
                else:
                    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#eeeeee"))
        _mwu_axes(ax, len(datasets), len(methods), xlabels, methods)
        ax.set_title(f"levels {level}", fontsize=13, fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=MWU_CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=axes[0].tolist(), orientation='horizontal', fraction=0.05, pad=0.18, aspect=50)
    cb.set_label("Mann-Whitney rank-biserial effect:  +1 = first model's scores always higher,  "
                 "-1 = second model's always higher,  0 = no difference", fontsize=10)
    fig.suptitle("Which model scores higher? Mann-Whitney U per model pair "
                 "(per-image mean over perturbations; bold * = Holm-adjusted p < 0.05)", fontsize=14, fontweight='bold')
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mwu_per_perturbation(mwu_all, order, level, path):
    """One panel per dataset at `level`: rows = methods, columns = perturbations (noises, then I/D canvases)."""
    t = mwu_all[(mwu_all['Level'] == level) & (mwu_all['Perturbation'] != 'all')]
    datasets = list(dict.fromkeys(t['Dataset']))
    methods = [m for m in order if m in set(t['Paradigm'])]
    perts = list(dict.fromkeys(list(t['Perturbation'])))
    perts = [p for p in perts if p not in A.ID_BASELINES] + [p for p in perts if p in A.ID_BASELINES]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(1.25 * len(perts) + 4, (0.5 * len(methods) + 1.2) * len(datasets)),
                             squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        dt = t[t['Dataset'] == ds].set_index(['Paradigm', 'Perturbation'])
        a, b = dt['Model_A'].iloc[0], dt['Model_B'].iloc[0]
        for y, method in enumerate(methods):
            for x, pert in enumerate(perts):
                if (method, pert) in dt.index:
                    _mwu_cell(ax, x, y, dt.loc[(method, pert)], short=True)
                else:
                    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#f2f2f2"))
        _mwu_axes(ax, len(perts), len(methods), [A.perturbation_name(p) for p in perts], methods)
        ax.set_title(f"{ds}:  {a} (blue = higher)  vs  {b} (orange = higher)", fontsize=12, fontweight='bold', loc='left')
    fig.suptitle(f"Mann-Whitney U per perturbation (levels {level}); cell = |rank-biserial effect|, "
                 "bold * = Holm-adjusted p < 0.05, grey = not applicable", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 4. ONE VERSION (raw or relative) THROUGH assessment_global.py
# ==========================================
def run_version(name, df, paradigms, palette):
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    A.OUT_DIR = out                          # assessment_global.py's plotting functions save into A.OUT_DIR
    A.PARADIGMS = {p: True for p in paradigms}
    A.PARADIGM_PALETTE.update(palette)
    df.to_csv(os.path.join(out, "Per_Image_Scores.csv"), index=False)

    print(f"\n>>> [{name}] Kendall's W / Spearman's rho and figures -> {out}")
    with contextlib.redirect_stdout(io.StringIO()):          # silence the per-group diagnostics
        w = A.run_image_ranking_consistency(df)
        for (model, scheme), s_df in w.groupby(['Model', 'Scheme'], sort=False):
            A.plot_model_scheme(model, scheme, s_df)
        for model, m_df in df.groupby('model', sort=False):
            A.plot_model_scores(model, m_df)
        for dataset in dict.fromkeys(ds for _, _, ds in A.MODELS.values()):
            models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset and m in set(df['model'])]
            if len(models) == 2:
                A.plot_dataset_comparison(dataset, models, df[df['model'].isin(models)])
        plot_mega_comparison(df, name, paradigms)
    w.to_csv(os.path.join(out, "Table3_Image_Ranking_Consistency.csv"), index=False)
    sep = model_separation(df)
    sep.round(4).to_csv(os.path.join(out, "Model_Separation.csv"), index=False)
    mwu = model_mann_whitney(df)
    mwu.round(6).to_csv(os.path.join(out, "Model_MannWhitney.csv"), index=False)
    return w, sep, mwu

# ==========================================
# 5. MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    pd.set_option('display.width', 250)

    print(f">>> Extracting raw scores | Noises: {A.NOISES} | Level fractions: {A.LEVEL_FRACTIONS}")
    df_raw = A.extract_all_scores()
    w_raw, sep_raw, mwu_raw = run_version("raw", df_raw, list(A.PARADIGMS), {})

    print("\n>>> Extracting relative scores")
    df_rel = extract_relative_scores()
    w_rel, sep_rel, mwu_rel = run_version("relative", df_rel, REL_PARADIGMS, REL_PALETTE)

    # Side-by-side summaries (the original S-Factor appears once: it is identical in both versions)
    w_all = pd.concat([w_raw.assign(Version='raw'),
                       w_rel[w_rel['Paradigm'] != 'S-Factor (IC/OC)'].assign(Version='relative')])
    w_all = w_all[w_all['Scheme'] == 'perturbation']
    w_all.drop(columns='Raters').to_csv(os.path.join(OUT_DIR, "Summary_Kendalls_W.csv"), index=False)
    sep_all = pd.concat([sep_raw.assign(Version='raw'),
                         sep_rel[sep_rel['Paradigm'] != 'S-Factor (IC/OC)'].assign(Version='relative')])
    sep_all.round(4).to_csv(os.path.join(OUT_DIR, "Summary_Model_Separation.csv"), index=False)
    mwu_all = pd.concat([mwu_raw.assign(Version='raw'),
                         mwu_rel[mwu_rel['Paradigm'] != 'S-Factor (IC/OC)'].assign(Version='relative')])
    mwu_all.round(6).to_csv(os.path.join(OUT_DIR, "Summary_Model_MannWhitney.csv"), index=False)
    plot_mwu_overview(mwu_all, SUMMARY_ORDER, os.path.join(OUT_DIR, "Fig_Summary_MannWhitney_Overview.pdf"))
    plot_mwu_per_perturbation(mwu_all, SUMMARY_ORDER, A.frac_label(A.SCORE_LEVEL),
                              os.path.join(OUT_DIR, "Fig_Summary_MannWhitney_Per_Perturbation.pdf"))

    print("\n=== Kendall's W, scheme 'perturbation' (mean over models) ===")
    print(w_all.pivot_table(index='Paradigm', columns='Group', values='Kendalls_W', aggfunc='mean')
          .reindex(SUMMARY_ORDER).round(3).to_string())
    print("\n=== Model separation: mean paired |r| over datasets ===")
    print(sep_all.assign(abs_r=sep_all['Rank_Biserial_r'].abs())
          .pivot_table(index='Paradigm', columns='Level', values='abs_r', aggfunc='mean')
          .reindex(SUMMARY_ORDER).round(2).to_string())
    print("\n=== Mann-Whitney U per model pair (levels 100%, per-image mean over perturbations) ===")
    print("    cell = higher-scoring model, rank-biserial effect, * Holm-adjusted p < 0.05")
    t = mwu_all[(mwu_all['Level'] == '100%') & (mwu_all['Perturbation'] == 'all')].copy()
    t['cell'] = (t['Higher_Model'] + ' ' + t['Rank_Biserial'].abs().round(2).astype(str)
                 + np.where(t['p_Holm'] < 0.05, '*', ' '))
    print(t.pivot_table(index='Paradigm', columns='Dataset', values='cell', aggfunc='first')
          .reindex(SUMMARY_ORDER).to_string())
    print(f"\n[DONE] Saved CSVs and figures to: {OUT_DIR}")

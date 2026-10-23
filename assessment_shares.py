"""
PRISM share assessment: retention + target/context shares, with validity controls
================================================================================
Builds on assessment_global.py (models, result folders, loaders, level fractions, Kendall's W)
and assessment_relative.py (paired model separation, Mann-Whitney, Holm). Reads the stored IC/OC
(prism.py) and Global (prism_global.py) trajectories; nothing is re-run. Only 'global_iou' is used,
so rank perplexity and spillover play no part here (they are handled separately later).

Per model, noise and level fraction (ratio of sums over images, as in Guarded Retention):
    kept_X(i) = mean_k min(IoU_i^X(eta_k), IoU_i(0))            X in {G (global), IC, OC}, k >= 1
    GR_X      = sum_i kept_X(i) / sum_i IoU_i(0)                 retention, in [0, 1]
    D_X       = 1 - GR_X                                          share of clean IoU lost
    target share  = D_IC / D_G      context share = D_OC / D_G    (undefined if D_G < TAU)
    share score   = harmonic mean of clip(target share) and 1 - clip(context share)
Model-level values come with 95% case-bootstrap intervals (images resampled, all quantities
recomputed from the resampled sums).

Per-image scores for the paired tests and Kendall's W are the linearised contributions of each
ratio (delta method):
    v_i = R + (a_i - R * b_i) / mean(b)      for R = sum(a) / sum(b)
Their mean is exactly the model-level ratio, and images are weighted by their clean IoU
(failed images do not count as perfectly robust).

Controls (a check only counts if both fail it):
    CLEAN (control)   score = clean IoU, ignores the corruption entirely
    RANDOM (control)  seeded uniform noise per (model, image, noise, level)

Checks written to OUT_DIR:
    Model_Level_Summary.csv     GR_G, GR_OC, GR_IC, D_*, shares, share score, 95% CIs
    Per_Image_Scores.csv        linearised per-image scores of every paradigm (+ controls)
    Table3_Image_Ranking_Consistency.csv   Kendall's W / Spearman (assessment_global engine)
    Model_Separation.csv        paired Wilcoxon + rank-biserial r per dataset pair (+ Holm)
    Model_MannWhitney.csv       unpaired Mann-Whitney U (assessment_relative engine)
    Discriminability.csv        between/within-model variance ratio (one-way F) per dataset pair
    Accuracy_Twin_Check.csv     score change for a twin that keeps 80% of every IoU
                                (IoU of a twin dropping 20% of GT pixels from its prediction is
                                 multiplied by 0.8 in expectation, so it is simulated without reruns)
    Clean_IoU_Consistency.csv   clean IoU agreement between the IC/OC and Global runs
    Fig_Share_Plane.pdf/.png    target share vs context share per dataset, model and noise
"""
import io
import os
import zlib
import contextlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import assessment_global as A
import assessment_relative as R

# ==========================================
# 1. CONFIGURATION
# ==========================================
OUT_DIR = f"{A.PRISM_ROOT}/Results/share_assessment"
TAU = 0.05          # minimum global drop D_G for a share to be defined ("not vulnerable" below it)
TWIN_KAPPA = 0.8    # accuracy twin keeps this fraction of every IoU
N_BOOT = 2000       # case-bootstrap resamples for the model-level intervals
SEED = 0            # bootstrap seed (fixed: the results are reproducible)
SAVE_PNG = True

PARADIGMS = ['Share score', 'Target share', 'Context share', 'Dependence IC', 'GR OC', 'GR Global',
             'Global AUC (raw)', 'Global AUC (relative)', 'CLEAN (control)', 'RANDOM (control)']
RATIO_PARADIGMS = ['Target share', 'Context share', 'Dependence IC', 'GR OC', 'GR Global']


# ==========================================
# 2. PER-IMAGE PARTS
# ==========================================
def iou_curve(levels):
    return np.array([float(l['global_iou']) / 100.0 for l in levels])

def kept(levels):
    """(mean_k min(IoU_k, IoU_0) over k >= 1, IoU_0) for one truncated trajectory."""
    v = iou_curve(levels)
    return float(np.mean(np.minimum(v[1:], v[0]))), float(v[0])

def random_score(model, image_id, noise, level):
    return np.random.default_rng(zlib.crc32(f"{model}|{image_id}|{noise}|{level}".encode())).uniform()

def extract_parts():
    """One row per (model, image, noise, level): clean IoU and kept IoU for G, IC and OC,
    plus the raw / relative Global AUC and the control scores."""
    rows, skipped = [], Counter()
    noises, levels = A.labelled(A.NOISES), A.labelled(A.LEVEL_FRACTIONS, A.frac_label)
    for model in A.MODELS:
        glob = {img: d for img, d in A.load_jsons(A.GLOBAL_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json",
                                                    skipped, f"{model} Global")}
        for image_id, data in A.load_jsons(A.ICOC_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json",
                                           skipped, f"{model} IC/OC"):
            if image_id not in glob:
                skipped[f"{model}: image in IC/OC but not in Global"] += 1
                continue
            try:
                ic = data['in_context_analysis']['level_data_per_noise']
                oc = data['out_of_context_analysis']['level_data_per_noise']
                gl = glob[image_id]['global_sota_analysis']['level_data_per_noise']
            except KeyError as e:
                skipped[f"{model}: missing key {e}"] += 1
                continue
            for noise, noise_lbl in noises:
                if noise not in ic or noise not in oc or noise not in gl:
                    skipped[f"{model}: missing noise '{noise}'"] += 1
                    continue
                li, lo, lg = ic[noise]['level_data'], oc[noise]['level_data'], gl[noise]['level_data']
                n = min(len(li), len(lo), len(lg))       # same noise grid for all three paradigms
                if n < 2:
                    skipped[f"{model}: trajectory shorter than 2 levels"] += 1
                    continue
                for frac, lvl_lbl in levels:
                    ti, to, tg = (A.truncate_levels(x[:n], frac) for x in (li, lo, lg))
                    (k_ic, q_ic), (k_oc, q_oc), (k_g, q_g) = kept(ti), kept(to), kept(tg)
                    rows.append({'model': model, 'image_id': image_id, 'perturbation': noise_lbl, 'level': lvl_lbl,
                                 'q0_G': q_g, 'q0_IC': q_ic, 'q0_OC': q_oc,
                                 'kept_G': k_g, 'kept_IC': k_ic, 'kept_OC': k_oc,
                                 'global_auc_raw': A.global_auc(lg[:n], frac),
                                 'global_auc_rel': R.global_auc_rel(lg[:n], frac),
                                 'random': random_score(model, image_id, noise_lbl, lvl_lbl)})
    if skipped:
        print("\n>>> SKIPPED DURING EXTRACTION")
        for reason, count in sorted(skipped.items()):
            print(f"  {count:5d} x {reason}")
    return pd.DataFrame(rows)


# ==========================================
# 3. RATIOS (model level, bootstrap) AND LINEARISED PER-IMAGE SCORES
# ==========================================
def numerators_denominators(p):
    """(a, b) arrays per ratio paradigm, so that the ratio is sum(a) / sum(b)."""
    lost = lambda X: p[f'q0_{X}'].values - p[f'kept_{X}'].values      # clean IoU lost, per image
    return {
        'GR Global':     (p['kept_G'].values, p['q0_G'].values),
        'GR OC':         (p['kept_OC'].values, p['q0_OC'].values),
        'Dependence IC': (lost('IC'), p['q0_IC'].values),
        'Target share':  (lost('IC'), lost('G')),
        'Context share': (lost('OC'), lost('G')),
    }

def ratio_values(ab, d_global):
    """Model-level ratios from summed parts; shares are undefined when D_G < TAU."""
    out = {}
    for name, (a, b) in ab.items():
        sb = b.sum()
        out[name] = a.sum() / sb if sb > 0 else np.nan
        if name in ('Target share', 'Context share') and not d_global >= TAU:
            out[name] = np.nan
    ts, cs = out['Target share'], out['Context share']
    if np.isnan(ts) or np.isnan(cs):
        out['Share score'] = np.nan
    else:
        x, y = np.clip(ts, 0, 1), 1 - np.clip(cs, 0, 1)
        out['Share score'] = 2 * x * y / (x + y) if x + y > 0 else 0.0
    return out

def model_level(parts):
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, noise, level), p in parts.groupby(['model', 'perturbation', 'level'], sort=False):
        ab = numerators_denominators(p)
        d_g = 1 - p['kept_G'].sum() / p['q0_G'].sum()
        point = ratio_values(ab, d_g)
        n = len(p)
        boot = {k: [] for k in point}
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            ab_b = {k: (a[idx], b[idx]) for k, (a, b) in ab.items()}
            d_g_b = 1 - p['kept_G'].values[idx].sum() / p['q0_G'].values[idx].sum()
            for k, v in ratio_values(ab_b, d_g_b).items():
                boot[k].append(v)
        row = {'Dataset': A.MODELS[model][2], 'Model': model, 'Perturbation': noise, 'Level': level,
               'Num_Images': n, 'D_G': d_g, 'Vulnerable (D_G >= TAU)': bool(d_g >= TAU),
               'Clean_IoU_mean': p['q0_G'].mean()}
        for k, v in point.items():
            b = np.array(boot[k], float)
            b = b[~np.isnan(b)]
            row[k] = v
            row[f'{k} CI_low'] = np.percentile(b, 2.5) if len(b) > 0.9 * N_BOOT else np.nan
            row[f'{k} CI_high'] = np.percentile(b, 97.5) if len(b) > 0.9 * N_BOOT else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def per_image_scores(parts):
    """Long table in assessment_global's record format: model, image_id, paradigm, perturbation, level, score."""
    recs = []
    for (model, noise, level), p in parts.groupby(['model', 'perturbation', 'level'], sort=False):
        ab = numerators_denominators(p)
        d_g = 1 - p['kept_G'].sum() / p['q0_G'].sum()
        ratios = ratio_values(ab, d_g)
        for name in RATIO_PARADIGMS:
            a, b = ab[name]
            Rv = ratios[name]
            if np.isnan(Rv) or b.mean() <= 0:
                continue
            v = Rv + (a - Rv * b) / b.mean()           # linearised contribution; mean(v) == Rv
            recs += [{'model': model, 'image_id': i, 'paradigm': name, 'perturbation': noise,
                      'level': level, 'score': float(s)} for i, s in zip(p['image_id'], v)]
        for name, col in (('Global AUC (raw)', 'global_auc_raw'), ('Global AUC (relative)', 'global_auc_rel'),
                          ('CLEAN (control)', 'q0_G'), ('RANDOM (control)', 'random')):
            recs += [{'model': model, 'image_id': i, 'paradigm': name, 'perturbation': noise,
                      'level': level, 'score': float(s)} for i, s in zip(p['image_id'], p[col]) if np.isfinite(s)]
    return pd.DataFrame(recs)


# ==========================================
# 4. VALIDITY CHECKS
# ==========================================
def accuracy_twin(parts):
    """Twin keeps TWIN_KAPPA of every IoU (clean and perturbed). Gap = original - twin, model level,
    full curve; also divided by the between-model SD of the original metric (|gap/SD| <= 0.1 passes)."""
    full = parts[parts['level'] == A.frac_label(1.0)]
    twin = full.copy()
    for c in ['q0_G', 'q0_IC', 'q0_OC', 'kept_G', 'kept_IC', 'kept_OC', 'global_auc_raw']:
        twin[c] = twin[c] * TWIN_KAPPA                  # relative AUC and ratios are scale-free
    rows = []
    for (model, noise), p in full.groupby(['model', 'perturbation'], sort=False):
        t = twin[(twin['model'] == model) & (twin['perturbation'] == noise)]
        vals = {}
        for tag, df in (('orig', p), ('twin', t)):
            ab = numerators_denominators(df)
            d_g = 1 - df['kept_G'].sum() / df['q0_G'].sum()
            r = ratio_values(ab, d_g)
            r['Global AUC (raw)'] = df['global_auc_raw'].mean()
            r['Global AUC (relative)'] = df['global_auc_rel'].mean()
            r['CLEAN (control)'] = df['q0_G'].mean()
            vals[tag] = r
        for k in vals['orig']:
            rows.append({'Model': model, 'Perturbation': noise, 'Paradigm': k,
                         'Original': vals['orig'][k], 'Twin': vals['twin'][k],
                         'Gap': vals['orig'][k] - vals['twin'][k]})
    out = pd.DataFrame(rows)
    sd = out.groupby(['Perturbation', 'Paradigm'])['Original'].transform('std')
    out['Gap_over_between_model_SD'] = out['Gap'] / sd.replace(0, np.nan)
    return out

def discriminability(df):
    """One-way F (between-model / within-model mean square) per dataset pair, paradigm and level,
    on per-image scores averaged over perturbations (as in the paired separation test)."""
    rows = []
    for dataset in dict.fromkeys(ds for _, _, ds in A.MODELS.values()):
        models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset and m in set(df['model'])]
        if len(models) != 2:
            continue
        for (paradigm, level), g in df[df['model'].isin(models)].groupby(['paradigm', 'level']):
            x = g.groupby(['model', 'image_id'])['score'].mean().reset_index()
            groups = [x.loc[x['model'] == m, 'score'].values for m in models]
            if min(len(v) for v in groups) < 2:
                continue
            grand = np.concatenate(groups).mean()
            ms_b = sum(len(v) * (v.mean() - grand) ** 2 for v in groups) / (len(groups) - 1)
            ms_w = sum(((v - v.mean()) ** 2).sum() for v in groups) / (sum(len(v) for v in groups) - len(groups))
            rows.append({'Dataset': dataset, 'Paradigm': paradigm, 'Level': level,
                         'Between_Within_Ratio': ms_b / ms_w if ms_w > 0 else np.nan})
    return pd.DataFrame(rows)

def clean_consistency(parts):
    """Clean IoU should be identical in the IC, OC and Global runs (same prompt, same image)."""
    full = parts[parts['level'] == A.frac_label(1.0)]
    return (full.assign(diff_IC=(full['q0_IC'] - full['q0_G']).abs(), diff_OC=(full['q0_OC'] - full['q0_G']).abs())
            .groupby('model')[['diff_IC', 'diff_OC']].max().reset_index()
            .rename(columns={'diff_IC': 'max |q0_IC - q0_G|', 'diff_OC': 'max |q0_OC - q0_G|'}))


# ==========================================
# 5. FIGURE: SHARE PLANE
# ==========================================
NOISE_MARKERS = {'poisson': 'o', 'gaussian': 's', 'speckle': '^', 'salt_pepper': 'D', 'gaussian_blur': 'v', 'rician': 'P'}

def plot_share_plane(ml, path):
    """Target share (x) vs context share (y) per dataset at the full curve; ideal corner = (1, 0).
    Colours follow assessment_global's model order; markers = noise; bars = 95% bootstrap CI."""
    full = ml[ml['Level'] == A.frac_label(1.0)]
    datasets = list(dict.fromkeys(ds for _, _, ds in A.MODELS.values()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.4), squeeze=False)
    handles = []
    for ax, dataset in zip(axes[0], datasets):
        models = [m for m, (_, _, ds) in A.MODELS.items() if ds == dataset]
        for m, color in zip(models, A.MODEL_COLORS):
            sub = full[full['Model'] == m]
            for _, r in sub.iterrows():
                if np.isnan(r['Target share']) or np.isnan(r['Context share']):
                    continue
                ax.errorbar(r['Target share'], r['Context share'],
                            xerr=[[r['Target share'] - r['Target share CI_low']], [r['Target share CI_high'] - r['Target share']]],
                            yerr=[[r['Context share'] - r['Context share CI_low']], [r['Context share CI_high'] - r['Context share']]],
                            fmt=NOISE_MARKERS.get(r['Perturbation'], 'o'), color=color, ms=7, lw=1, capsize=2,
                            markeredgecolor='white', markeredgewidth=1)
            handles.append(Line2D([], [], color=color, marker='o', lw=0, ms=7, label=m))
        ax.plot([1], [0], marker='*', ms=14, color='#444444', lw=0)
        ax.annotate('ideal', (1, 0), xytext=(9, -3), textcoords='offset points', fontsize=8, color='#444444')
        ax.set_xlim(-0.05, 1.25); ax.set_ylim(-0.05, 1.05)
        ax.set_title(dataset, fontsize=11, loc='left', fontweight='bold')
        ax.set_xlabel('target share  D_IC / D_G  (higher = better)', fontsize=9)
        ax.grid(color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel('context share  D_OC / D_G  (lower = better)', fontsize=9)
    handles += [Line2D([], [], color='#666666', marker=mk, lw=0, ms=7, label=n) for n, mk in NOISE_MARKERS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 7), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    if SAVE_PNG:
        fig.savefig(path.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)


# ==========================================
# 6. MASTER EXECUTION
# ==========================================
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option('display.width', 250)
    print(f">>> Share assessment | noises: {A.NOISES} | level fractions: {A.LEVEL_FRACTIONS} | TAU = {TAU}")
    parts = extract_parts()
    if parts.empty:
        raise SystemExit("No IC/OC + Global trajectory pairs found; check the result folders in assessment_global.py.")

    ml = model_level(parts)
    ml.round(4).to_csv(os.path.join(OUT_DIR, "Model_Level_Summary.csv"), index=False)
    df = per_image_scores(parts)
    df.to_csv(os.path.join(OUT_DIR, "Per_Image_Scores.csv"), index=False)
    clean_consistency(parts).round(5).to_csv(os.path.join(OUT_DIR, "Clean_IoU_Consistency.csv"), index=False)

    with contextlib.redirect_stdout(io.StringIO()):          # silence the per-group diagnostics
        w = A.run_image_ranking_consistency(df)
    w.to_csv(os.path.join(OUT_DIR, "Table3_Image_Ranking_Consistency.csv"), index=False)

    sep = R.model_separation(df)
    if not sep.empty:
        sep['p_Holm'] = R.holm(sep['Wilcoxon_p'].values)
    sep.round(4).to_csv(os.path.join(OUT_DIR, "Model_Separation.csv"), index=False)
    R.model_mann_whitney(df).round(6).to_csv(os.path.join(OUT_DIR, "Model_MannWhitney.csv"), index=False)
    disc = discriminability(df)
    disc.round(3).to_csv(os.path.join(OUT_DIR, "Discriminability.csv"), index=False)
    twin = accuracy_twin(parts)
    twin.round(5).to_csv(os.path.join(OUT_DIR, "Accuracy_Twin_Check.csv"), index=False)
    plot_share_plane(ml, os.path.join(OUT_DIR, "Fig_Share_Plane.pdf"))

    full = A.frac_label(1.0)
    print("\n=== Model level (full curve): retention, dependence and shares ===")
    cols = ['GR Global', 'GR OC', 'Dependence IC', 'Target share', 'Context share', 'Share score']
    print(ml[ml['Level'] == full].groupby('Model', sort=False)[cols].mean().round(3).to_string())
    print(f"\n    'not vulnerable' cells (D_G < {TAU}): {int((~ml['Vulnerable (D_G >= TAU)']).sum())} of {len(ml)}")
    print("\n=== Kendall's W, scheme 'perturbation' (mean over models), incl. controls ===")
    wp = w[w['Scheme'] == 'perturbation']
    print(wp.pivot_table(index='Paradigm', columns='Group', values='Kendalls_W', aggfunc='mean')
          .reindex([p for p in PARADIGMS if p in set(wp['Paradigm'])]).round(3).to_string())
    print("\n=== Model separation (full curve): paired rank-biserial r, * Holm p < 0.05 ===")
    s = sep[sep['Level'] == full].copy()
    s['cell'] = s['Rank_Biserial_r'].round(2).astype(str) + np.where(s['p_Holm'] < 0.05, '*', '')
    print(s.pivot_table(index='Paradigm', columns='Dataset', values='cell', aggfunc='first')
          .reindex([p for p in PARADIGMS if p in set(s['Paradigm'])]).to_string())
    print("\n=== Discriminability (full curve): between/within-model ratio ===")
    d = disc[disc['Level'] == full]
    print(d.pivot_table(index='Paradigm', columns='Dataset', values='Between_Within_Ratio')
          .reindex([p for p in PARADIGMS if p in set(d['Paradigm'])]).round(2).to_string())
    print("\n=== Accuracy twin: mean |gap / between-model SD| (pass <= 0.1) ===")
    print(twin.assign(a=twin['Gap_over_between_model_SD'].abs()).groupby('Paradigm')['a'].mean()
          .reindex([p for p in PARADIGMS if p in set(twin['Paradigm'])]).round(3).to_string())
    print(f"\n[DONE] Saved CSVs and figure to: {OUT_DIR}")


if __name__ == "__main__":
    run()
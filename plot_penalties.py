"""
Rank perplexity and CSR per model and perturbation
==================================================
For every model (assessment_global.MODELS), image and noise, from prism.py's IC/OC results:
    RP   mean rank_perplexity over the perturbed steps of the curve (k >= 1)
    CSR  conditional average spillover of the curve (spillover_metrics.conditional_average_spillover:
         mean instant_spillover_ratio over the steps with spillover > 0, 0 if none)
for both the IC curve (IC superpixels perturbed) and the OC curve (70% of OC superpixels perturbed).

Outputs in OUT_DIR:
    Fig_<model>_RP_CSR.pdf/.png         per model: RP and CSR panels, x = perturbation,
                                        split candle = IC (left) | OC (right) over the images
    Fig_All_Models_RP_CSR.pdf/.png      all models: rows = models, columns = RP, CSR
    Penalties_Per_Image.csv             model, image_id, noise, domain, RP, CSR
    Penalties_Summary.csv               median / IQR / mean per model x noise x domain
    Fig_<model>_RP_Spill_by_Level.pdf/.png   per model: rank perplexity (top) and step spillover (bottom)
                                        vs noise level, one column per noise; IC and OC median + IQR band
    Fig_All_Models_RP_by_Level.pdf/.png rank perplexity vs noise level: rows = models, columns = noises
    Penalties_By_Level.csv              median / q1 / q3 of RP and spillover per model x noise x domain x level

All methods (one value per image and perturbation, RP = mean over the perturbed steps):
    PRISM IC/OC   as above (IC superpixels ranked; IC | OC split)
    Global        prism_global.py: stored rank_perplexity (all superpixels ranked, whole image perturbed)
                  and stored conditional_average_spillover
    I/D Deletion / Insertion   prism_sota.py, black / mean canvas: RP recomputed from the stored per-step
                  superpixel probabilities (IC superpixels, ranked by (mean_prob, area) as in prism.py) against
                  the clean ranking (deletion step 0), / floor(Z^2/2); CSR = conditional average of the increase
                  in background leakage (spillover_ratio_k - spillover_ratio_clean, steps > 0)
    Fig_All_Methods_RP.pdf/.png         rank perplexity: rows = models, column groups = methods
    Fig_All_Methods_CSR.pdf/.png        CSR, same layout
    Penalties_All_Methods.csv           model, image_id, method, condition, domain, RP, CSR
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import assessment_global as A

# ==========================================
# CONFIGURATION
# ==========================================
OUT_DIR = f"{A.PRISM_ROOT}/Results/penalties"
DOMAIN_COLORS = {"IC": "#2a78d6", "OC": "#eb6834"}
MEASURES = [("RP", "Rank perplexity (mean over steps)"), ("CSR", "CSR (conditional avg. spillover)")]

# ==========================================
# EXTRACTION
# ==========================================
def curve_penalties(trajectory):
    levels = trajectory['level_data']
    rp = [float(l.get('rank_perplexity', 0.0)) for l in levels[1:]]
    csr = trajectory.get('spillover_metrics', {}).get('conditional_average_spillover')
    if csr is None:                                    # recompute if the summary is missing
        bleed = [float(l.get('instant_spillover_ratio', 0.0)) for l in levels[1:]]
        bleed = [s for s in bleed if s > 0]
        csr = sum(bleed) / len(bleed) if bleed else 0.0
    return (float(np.mean(rp)) if rp else np.nan), float(csr)

def extract():
    rows, skipped = [], Counter()
    for model in A.MODELS:
        for image_id, data in A.load_jsons(A.ICOC_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} IC/OC"):
            for domain, key in (("IC", "in_context_analysis"), ("OC", "out_of_context_analysis")):
                per_noise = data.get(key, {}).get('level_data_per_noise', {})
                for noise in A.NOISES:
                    if noise not in per_noise:
                        skipped[f"{model} {domain}: missing noise '{noise}'"] += 1
                        continue
                    rp, csr = curve_penalties(per_noise[noise])
                    rows.append({'model': model, 'image_id': image_id, 'noise': noise, 'domain': domain, 'RP': rp, 'CSR': csr})
    for reason, count in sorted(skipped.items()): print(f"  skipped {count:5d} x {reason}")
    return pd.DataFrame(rows)

def conditional_mean(values):
    pos = [v for v in values if v > 0]
    return sum(pos) / len(pos) if pos else 0.0

def ranks_of(superpixels):
    """IC superpixel ranks at one step, ranked by (mean_prob, area) descending as in prism.py."""
    ordered = sorted(superpixels, key=lambda sp: (sp['mean_prob'], sp['area']), reverse=True)
    return {sp['sp_id']: r for r, sp in enumerate(ordered, 1)}

def id_penalties(traj, clean_ranks, clean_spill):
    """Mean rank perplexity over the perturbed steps and CSR of one I/D trajectory."""
    Z = len(clean_ranks)
    if Z < 2 or len(traj) < 2: return np.nan, np.nan
    max_jumps = (Z * Z) // 2
    rps, spills = [], []
    for step in traj:
        if float(step.get('fraction_perturbed', 0.0)) <= 0: continue          # the unperturbed / empty-canvas start
        rk = ranks_of(step.get('superpixels', []))
        rps.append(sum(abs(rk[s] - clean_ranks[s]) for s in clean_ranks if s in rk) / max_jumps)
        spills.append(max(0.0, float(step.get('spillover_ratio', 0.0)) - clean_spill))
    return (float(np.mean(rps)) if rps else np.nan), conditional_mean(spills)

def extract_all_methods(df_icoc):
    """RP and CSR per image for every method; PRISM IC/OC rows come from extract()."""
    rows = [dict(model=r.model, image_id=r.image_id, method='PRISM IC/OC', condition=r.noise, domain=r.domain,
                 RP=r.RP, CSR=r.CSR) for r in df_icoc.itertuples()]
    skipped = Counter()
    for model in A.MODELS:
        for image_id, data in A.load_jsons(A.GLOBAL_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} Global"):
            per_noise = data.get('global_sota_analysis', {}).get('level_data_per_noise', {})
            for noise in A.NOISES:
                if noise not in per_noise: continue
                rp, csr = curve_penalties(per_noise[noise])
                rows.append(dict(model=model, image_id=image_id, method='Global', condition=noise, domain='all', RP=rp, CSR=csr))
        for image_id, data in A.load_jsons(A.ID_DIRS[model], "*/*_analysis.json", skipped, f"{model} I/D"):
            trajectories = data.get('prob_guided_analysis', {}).get('trajectories', {})
            for base in A.ID_BASELINES:
                t = trajectories.get(base, {}).get('prob_ranked')
                if not t or not t.get('deletion'): continue
                clean = t['deletion'][0]                                     # fraction 0 = unperturbed image
                clean_ranks, clean_spill = ranks_of(clean.get('superpixels', [])), float(clean.get('spillover_ratio', 0.0))
                for method, key in (('I/D Deletion', 'deletion'), ('I/D Insertion', 'insertion')):
                    rp, csr = id_penalties(t[key], clean_ranks, clean_spill)
                    rows.append(dict(model=model, image_id=image_id, method=method, condition=base, domain='all', RP=rp, CSR=csr))
    return pd.DataFrame(rows)

METHOD_COLUMNS = [('PRISM IC/OC', A.NOISES), ('Global', A.NOISES),
                  ('I/D Deletion', A.ID_BASELINES), ('I/D Insertion', A.ID_BASELINES)]
METHOD_COLORS = {'Global': A.PARADIGM_PALETTE['Global Continuous AUC'],
                 'I/D Deletion': A.PARADIGM_PALETTE['Standard I/D AUC (Deletion)'],
                 'I/D Insertion': A.PARADIGM_PALETTE['Standard I/D AUC (Insertion)']}

def plot_all_methods(da, measure, title, path):
    """Rows = models, column groups = methods; candle per perturbation over the images
    (PRISM IC/OC split IC | OC, the others a full candle)."""
    models = list(dict.fromkeys(da['model']))
    widths = [len(conds) for _, conds in METHOD_COLUMNS]
    fig, axes = plt.subplots(len(models), len(METHOD_COLUMNS), figsize=(1.05 * sum(widths) + 3, 2.6 * len(models)),
                             squeeze=False, gridspec_kw={'width_ratios': widths}, sharey='row')
    for i, model in enumerate(models):
        m_df = da[da['model'] == model]
        for j, (method, conds) in enumerate(METHOD_COLUMNS):
            ax = axes[i, j]
            g = m_df[m_df['method'] == method]
            for x, cond in enumerate(conds):
                c = g[g['condition'] == cond]
                if method == 'PRISM IC/OC':
                    A.draw_half_violin(ax, x, c.loc[c['domain'] == 'IC', measure].dropna().values, -1, DOMAIN_COLORS['IC'])
                    A.draw_half_violin(ax, x, c.loc[c['domain'] == 'OC', measure].dropna().values, +1, DOMAIN_COLORS['OC'])
                else:
                    v = c[measure].dropna().values
                    A.draw_half_violin(ax, x, v, -1, METHOD_COLORS[method])
                    A.draw_half_violin(ax, x, v, +1, METHOD_COLORS[method])
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([A.perturbation_name(c) for c in conds] if i == len(models) - 1 else [],
                               rotation=30, ha='right', fontsize=9)
            ax.set_xlim(-0.6, len(conds) - 0.4)
            if i == 0: ax.set_title(method, fontsize=12, fontweight='bold')
            if j == 0: ax.set_ylabel(f"{model}\n({A.MODELS[model][2]})\n{measure}", fontsize=9)
            ax.grid(True, color="#dddddd", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
    handles = [Patch(facecolor=DOMAIN_COLORS['IC'], alpha=0.55, label='PRISM IC curve'),
               Patch(facecolor=DOMAIN_COLORS['OC'], alpha=0.55, label='PRISM OC curve')]
    handles += [Patch(facecolor=c, alpha=0.55, label=m) for m, c in METHOD_COLORS.items()]
    handles += [Line2D([], [], color="#222222", lw=1.4, label="median"),
                Line2D([], [], color="#222222", lw=0.9, ls="--", label="quartiles")]
    fig.suptitle(title, fontsize=17, fontweight='bold', y=1.0)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path[:-4] + ".png", dpi=110, bbox_inches='tight')
    plt.close(fig)

def extract_by_level():
    """Per step: model, image_id, noise, domain, noise level, rank_perplexity, instant_spillover_ratio."""
    rows, skipped = [], Counter()
    for model in A.MODELS:
        for image_id, data in A.load_jsons(A.ICOC_DIRS[model], f"{A.RUN_SUBDIR}/*/*_analysis.json", skipped, f"{model} IC/OC"):
            for domain, key in (("IC", "in_context_analysis"), ("OC", "out_of_context_analysis")):
                per_noise = data.get(key, {}).get('level_data_per_noise', {})
                for noise in A.NOISES:
                    for l in per_noise.get(noise, {}).get('level_data', []):
                        rows.append((model, image_id, noise, domain, round(float(l.get('noise_level', 0.0)), 4),
                                     float(l.get('rank_perplexity', 0.0)), float(l.get('instant_spillover_ratio', 0.0))))
    return pd.DataFrame(rows, columns=['model', 'image_id', 'noise', 'domain', 'level', 'RP', 'spill'])

def level_summary(lv):
    g = lv.groupby(['model', 'noise', 'domain', 'level'], sort=False)
    out = g[['RP', 'spill']].median().add_suffix('_median')
    out = out.join(g[['RP', 'spill']].quantile(0.25).add_suffix('_q1')).join(g[['RP', 'spill']].quantile(0.75).add_suffix('_q3'))
    return out.join(g.size().rename('n_images')).reset_index()

def draw_level_panel(ax, s, model, noise, measure):
    for domain in ("IC", "OC"):
        d = s[(s['model'] == model) & (s['noise'] == noise) & (s['domain'] == domain)].sort_values('level')
        if d.empty: continue
        ax.fill_between(d['level'], d[f'{measure}_q1'], d[f'{measure}_q3'], color=DOMAIN_COLORS[domain], alpha=0.18, lw=0)
        ax.plot(d['level'], d[f'{measure}_median'], color=DOMAIN_COLORS[domain], lw=1.8)
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

LEVEL_LEGEND = [Line2D([], [], color=DOMAIN_COLORS["IC"], lw=1.8, label="IC curve: median over images"),
                Line2D([], [], color=DOMAIN_COLORS["OC"], lw=1.8, label="OC curve: median over images"),
                Patch(facecolor="#888888", alpha=0.25, label="25-75% of images")]

def plot_model_by_level(model, s):
    rows = [("RP", "Rank perplexity"), ("spill", "Spillover (per step)")]
    fig, axes = plt.subplots(len(rows), len(A.NOISES), figsize=(3.3 * len(A.NOISES), 3.0 * len(rows)),
                             squeeze=False, sharex=True, sharey='row')
    for i, (measure, label) in enumerate(rows):
        for j, noise in enumerate(A.NOISES):
            ax = axes[i, j]
            draw_level_panel(ax, s, model, noise, measure)
            if i == 0: ax.set_title(A.perturbation_name(noise), fontsize=12, fontweight='bold')
            if i == len(rows) - 1: ax.set_xlabel("noise level η")
            if j == 0: ax.set_ylabel(label, fontsize=11)
    fig.suptitle(f"{model}: rank perplexity and spillover at each noise level", fontsize=15, fontweight='bold', y=1.0)
    fig.legend(handles=LEVEL_LEGEND, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT_DIR, f"Fig_{model}_RP_Spill_by_Level.pdf")
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path[:-4] + ".png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_all_by_level(s):
    models = list(dict.fromkeys(s['model']))
    fig, axes = plt.subplots(len(models), len(A.NOISES), figsize=(3.3 * len(A.NOISES), 2.4 * len(models)),
                             squeeze=False, sharex=True, sharey=True)
    for i, model in enumerate(models):
        for j, noise in enumerate(A.NOISES):
            ax = axes[i, j]
            draw_level_panel(ax, s, model, noise, "RP")
            if i == 0: ax.set_title(A.perturbation_name(noise), fontsize=12, fontweight='bold')
            if i == len(models) - 1: ax.set_xlabel("noise level η")
            if j == 0: ax.set_ylabel(f"{model}\nrank perplexity", fontsize=10)
    fig.suptitle("Rank perplexity at each noise level, all models", fontsize=17, fontweight='bold', y=1.0)
    fig.legend(handles=LEVEL_LEGEND, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT_DIR, "Fig_All_Models_RP_by_Level.pdf")
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path[:-4] + ".png", dpi=120, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# PLOTTING
# ==========================================
def draw_panel(ax, m_df, measure, title, show_xlabels=True):
    for x, noise in enumerate(A.NOISES):
        for domain, side in (("IC", -1), ("OC", +1)):
            v = m_df.loc[(m_df['noise'] == noise) & (m_df['domain'] == domain), measure].dropna().values
            A.draw_half_violin(ax, x, v, side, DOMAIN_COLORS[domain])
    ax.set_xticks(range(len(A.NOISES)))
    ax.set_xticklabels([A.perturbation_name(n) for n in A.NOISES] if show_xlabels else [], rotation=30, ha='right')
    ax.set_xlim(-0.6, len(A.NOISES) - 0.4)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

LEGEND = [Patch(facecolor=DOMAIN_COLORS["IC"], alpha=0.55, label="IC curve (object perturbed)"),
          Patch(facecolor=DOMAIN_COLORS["OC"], alpha=0.55, label="OC curve (background perturbed)"),
          Line2D([], [], color="#222222", lw=1.4, label="median"),
          Line2D([], [], color="#222222", lw=0.9, ls="--", label="quartiles")]

def plot_model(model, m_df):
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(7.5 * len(MEASURES), 5), squeeze=False)
    for ax, (measure, title) in zip(axes[0], MEASURES):
        draw_panel(ax, m_df, measure, title)
        ax.set_ylabel(measure, fontsize=11)
    fig.suptitle(f"{model}: rank perplexity and CSR per perturbation ({m_df['image_id'].nunique()} images)",
                 fontsize=15, fontweight='bold', y=1.0)
    fig.legend(handles=LEGEND, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT_DIR, f"Fig_{model}_RP_CSR.pdf")
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path[:-4] + ".png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_all(df):
    models = list(dict.fromkeys(df['model']))
    fig, axes = plt.subplots(len(models), len(MEASURES), figsize=(7.5 * len(MEASURES), 3.2 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        m_df = df[df['model'] == model]
        for j, (measure, title) in enumerate(MEASURES):
            draw_panel(axes[i, j], m_df, measure, title if i == 0 else "", show_xlabels=(i == len(models) - 1))
            axes[i, j].set_ylabel(f"{model}\n({A.MODELS[model][2]})\n{measure}" if j == 0 else measure, fontsize=10)
    fig.suptitle("Rank perplexity and CSR per perturbation, all models", fontsize=17, fontweight='bold', y=1.0)
    fig.legend(handles=LEGEND, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = os.path.join(OUT_DIR, "Fig_All_Models_RP_CSR.pdf")
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path[:-4] + ".png", dpi=120, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = extract()
    df.to_csv(os.path.join(OUT_DIR, "Penalties_Per_Image.csv"), index=False)
    q = lambda p: (lambda s: s.quantile(p))
    summary = df.groupby(['model', 'noise', 'domain'], sort=False)[['RP', 'CSR']].agg(['median', q(0.25), q(0.75), 'mean'])
    summary.columns = [f"{m}_{s}" for m, s in zip(summary.columns.get_level_values(0),
                                                  ['median', 'q1', 'q3', 'mean'] * 2)]
    summary.round(5).to_csv(os.path.join(OUT_DIR, "Penalties_Summary.csv"))
    for model, m_df in df.groupby('model', sort=False):
        plot_model(model, m_df)
    plot_all(df)

    s = level_summary(extract_by_level())
    s.round(5).to_csv(os.path.join(OUT_DIR, "Penalties_By_Level.csv"), index=False)
    for model in dict.fromkeys(s['model']):
        plot_model_by_level(model, s)
    plot_all_by_level(s)

    da = extract_all_methods(df)
    da.to_csv(os.path.join(OUT_DIR, "Penalties_All_Methods.csv"), index=False)
    plot_all_methods(da, 'RP', "Rank perplexity per perturbation, all methods and models "
                     "(mean over the perturbed steps; one value per image)", os.path.join(OUT_DIR, "Fig_All_Methods_RP.pdf"))
    plot_all_methods(da, 'CSR', "CSR (conditional average spillover) per perturbation, all methods and models",
                     os.path.join(OUT_DIR, "Fig_All_Methods_CSR.pdf"))
    pd.set_option('display.width', 200)
    print(df.groupby(['model', 'domain'], sort=False)[['RP', 'CSR']].median().unstack('domain').round(4).to_string())
    print(f"\n[DONE] Saved figures and CSVs to: {OUT_DIR}")

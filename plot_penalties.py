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
    pd.set_option('display.width', 200)
    print(df.groupby(['model', 'domain'], sort=False)[['RP', 'CSR']].median().unstack('domain').round(4).to_string())
    print(f"\n[DONE] Saved figures and CSVs to: {OUT_DIR}")

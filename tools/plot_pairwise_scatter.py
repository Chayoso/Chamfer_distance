#!/usr/bin/env python3
"""Generate figs/pairwise_scatter.png from Table 4a (tab:pairwise) values.

Style: filled markers for improvements, hollow for regressions,
background shading, annotations on regressions + best improvement,
summary stats box.

Usage:
    python tools/plot_pairwise_scatter.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")

SHAPES = ['sphere', 'bunny', 'bob', 'spot', 'teapot']
SHAPE_LABELS = {'sphere': 'Sphere', 'bunny': 'Bunny', 'bob': 'Bob',
                'spot': 'Spot', 'teapot': 'Teapot'}
SHORT = {'sphere': 'Sph', 'bunny': 'Bu', 'bob': 'Bo', 'spot': 'Sp', 'teapot': 'Te'}

# (physics, ours) for each directed pair — from Table 4a
PAIRS = {
    ('sphere', 'bunny'):  (0.251, 0.226),
    ('sphere', 'bob'):    (0.282, 0.276),
    ('sphere', 'spot'):   (0.257, 0.236),
    ('sphere', 'teapot'): (0.262, 0.234),
    ('bunny', 'sphere'):  (0.295, 0.250),
    ('bunny', 'bob'):     (0.297, 0.288),
    ('bunny', 'spot'):    (0.256, 0.240),
    ('bunny', 'teapot'):  (0.286, 0.259),
    ('bob', 'sphere'):    (0.293, 0.296),
    ('bob', 'bunny'):     (0.283, 0.276),
    ('bob', 'spot'):      (0.280, 0.295),
    ('bob', 'teapot'):    (0.326, 0.309),
    ('spot', 'sphere'):   (0.291, 0.315),
    ('spot', 'bunny'):    (0.275, 0.252),
    ('spot', 'bob'):      (0.425, 0.284),
    ('spot', 'teapot'):   (0.318, 0.293),
    ('teapot', 'sphere'): (0.226, 0.213),
    ('teapot', 'bunny'):  (0.257, 0.252),
    ('teapot', 'bob'):    (0.355, 0.312),
    ('teapot', 'spot'):   (0.237, 0.259),
}

MARKERS = {
    'sphere': 'o',
    'bunny':  's',
    'bob':    '^',
    'spot':   'D',
    'teapot': 'v',
}

COLORS = {
    'sphere': '#4C72B0',
    'bunny':  '#E07070',
    'bob':    '#55A868',
    'spot':   '#CCB943',
    'teapot': '#9467BD',
}

MS = 100  # marker size


def plot():
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- Background shading ---
    lims = [0.19, 0.44]
    # Pink above diagonal (Physics better)
    ax.fill_between(lims, lims, [lims[1], lims[1]],
                    color='#E8B0B0', alpha=0.18, zorder=0)
    # Green below diagonal (Ours better)
    ax.fill_between(lims, [lims[0], lims[0]], lims,
                    color='#B0E8B0', alpha=0.12, zorder=0)

    # Diagonal line
    ax.plot(lims, lims, '-', color='gray', linewidth=1.2, alpha=0.7, zorder=1)

    # Region labels — placed in clear areas away from legend and summary box
    ax.text(0.52, 0.05, 'Ours better', fontsize=12, color='#2a7f2a', alpha=0.45,
            ha='center', style='italic', zorder=1, transform=ax.transAxes)
    ax.text(0.70, 0.92, 'Physics better', fontsize=12, color='#b03030', alpha=0.45,
            ha='center', style='italic', zorder=1, transform=ax.transAxes)

    # --- Classify improved vs regression ---
    improved = {}   # src -> [(phys, ours), ...]
    regressed = {}  # src -> [(phys, ours, tgt), ...]
    all_deltas = []

    for (src, tgt), (p, o) in PAIRS.items():
        delta = (o - p) / p
        all_deltas.append(delta)
        if o < p:
            improved.setdefault(src, []).append((p, o))
        else:
            regressed.setdefault(src, []).append((p, o, tgt))

    # --- Plot improved (filled) ---
    legend_done = set()
    for src in SHAPES:
        pts = improved.get(src, [])
        if not pts:
            continue
        px = [v[0] for v in pts]
        py = [v[1] for v in pts]
        ax.scatter(px, py, marker=MARKERS[src], s=MS,
                   c=COLORS[src], edgecolors='white', linewidths=0.8,
                   label=SHAPE_LABELS[src], zorder=3)
        legend_done.add(src)

    # --- Plot regressions (hollow) ---
    for src in SHAPES:
        pts = regressed.get(src, [])
        if not pts:
            continue
        px = [v[0] for v in pts]
        py = [v[1] for v in pts]
        label = SHAPE_LABELS[src] if src not in legend_done else None
        ax.scatter(px, py, marker=MARKERS[src], s=MS,
                   facecolors='none', edgecolors=COLORS[src], linewidths=2.0,
                   label=label, zorder=3)
        legend_done.add(src)

    # Regression legend marker (hollow circle)
    ax.scatter([], [], marker='o', s=MS, facecolors='none',
               edgecolors='gray', linewidths=2.0, label='Regression')

    # --- Annotate regressions ---
    for (src, tgt), (p, o) in PAIRS.items():
        if o >= p:  # regression
            lbl = f'{SHORT[src]}\u2192{SHORT[tgt]}'
            # offset direction: push label away from diagonal
            dx, dy = -0.008, 0.005
            ax.annotate(lbl, (p, o), xytext=(p + dx, o + dy),
                        fontsize=7.5, color='#555555', style='italic',
                        zorder=4)

    # --- Annotate best improvement (spot→bob) ---
    best_pair = None
    best_delta = 0
    for (src, tgt), (p, o) in PAIRS.items():
        delta = (o - p) / p
        if delta < best_delta:
            best_delta = delta
            best_pair = (src, tgt)

    if best_pair:
        src, tgt = best_pair
        p, o = PAIRS[best_pair]
        lbl = f'{SHORT[src]}\u2192{SHORT[tgt]}  {best_delta:+.0%}'
        ax.annotate(lbl, (p, o),
                    xytext=(p - 0.02, o + 0.015),
                    fontsize=10, fontweight='bold', color='#C44E52',
                    arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                    zorder=5)

    # --- Summary stats box ---
    n_improved = sum(1 for (_, _), (p, o) in PAIRS.items() if o < p)
    improved_deltas = [d for d in all_deltas if d < 0]
    avg_improved = np.mean(improved_deltas) * 100 if improved_deltas else 0
    stats_text = f'{n_improved}/20 improved\navg. {avg_improved:+.1f}%'
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85),
            zorder=5)

    # --- Axes ---
    ax.set_xlabel('Physics-only  (two-sided CD)', fontsize=13)
    ax.set_ylabel('Ours  (two-sided CD)', fontsize=13)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(title='Source shape', fontsize=9, title_fontsize=10,
              loc='upper left', framealpha=0.92)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = WORKSPACE / 'figs' / 'pairwise_scatter.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Improved: {n_improved}/20, avg improved: {avg_improved:+.1f}%")


if __name__ == '__main__':
    plot()

"""
Generate per-species dimensionality reduction plots (PCA, t-SNE, UMAP) of
intrinsic electrophysiological features.
Follows the preprocessing pipeline from synphys_culture_classifier.ipynb:
  RobustScaler → KNNImputer → {PCA / t-SNE / UMAP}
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

_HERE    = Path(__file__).parent
CSV_PATH = _HERE / 'intrinsic_features.csv'
OUT_DIR  = _HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Feature set ───────────────────────────────────────────────────────────────
# Full feature set matching notebook pipeline — subthreshold, suprathreshold
# firing, AP waveform, cross-sweep adaptation, and chirp/impedance features.
# Per-sweep metadata (stim_amp, v_baseline, sag, index) and rarely-populated
# features (adp_v <2%) are excluded.
CANDIDATE_FEATURES = [
    # Subthreshold / passive
    'sag', 'vm_for_sag', 'input_resistance', 'tau',
    # Threshold
    'rheobase_i', 'fi_fit_slope',
    # Rheobase sweep — firing
    'rheobase_avg_rate', 'rheobase_latency',
    'rheobase_first_isi', 'rheobase_mean_isi', 'rheobase_isi_cv',
    # Rheobase sweep — AP waveform
    'rheobase_threshold_v', 'rheobase_upstroke', 'rheobase_downstroke',
    'rheobase_upstroke_downstroke_ratio', 'rheobase_width',
    'rheobase_peak_deltav', 'rheobase_fast_trough_v', 'rheobase_fast_trough_deltav',
    # Hero sweep — firing
    'hero_avg_rate', 'hero_latency', 'hero_adapt',
    'hero_isi_cv', 'hero_mean_isi', 'hero_first_isi',
    # Hero sweep — AP waveform
    'hero_threshold_v', 'hero_upstroke', 'hero_downstroke',
    'hero_upstroke_downstroke_ratio', 'hero_width',
    'hero_peak_deltav', 'hero_fast_trough_v', 'hero_fast_trough_deltav',
    # Cross-sweep adaptation
    'adapt_mean', 'isi_cv_mean',
    'upstroke_adapt_ratio', 'downstroke_adapt_ratio',
    'upstroke_downstroke_ratio_adapt_ratio',
    'width_adapt_ratio', 'fast_trough_v_adapt_ratio',
    'peak_v_adapt_ratio', 'threshold_v_adapt_ratio',
    # Chirp / impedance
    'peak_ratio_chirp', 'peak_freq_chirp', 'z_max_chirp', 'z_mean_chirp',
    'phase_at_peak_chirp',
]

UMAP_PARAMS  = dict(n_neighbors=10, min_dist=0.25, random_state=42)
TSNE_PARAMS  = dict(n_components=2, perplexity=10, random_state=42,
                    max_iter=2000, init='pca', learning_rate='auto')
PCA_PARAMS   = dict(n_components=2, random_state=42)

SPECIES_MAP = {
    'mouse': 'Mouse (TEa)',
    'NHP':   'Non-Human Primate',
    'human': 'Human',
}

PALETTE = {'E': '#E87D3E', 'I': '#4A90D9'}   # orange / blue


def infer_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive species and E/I label from sheet name."""
    def _parse(s):
        sl = s.lower()
        if 'mouse' in sl:   species = 'mouse'
        elif 'nhp' in sl:   species = 'NHP'
        elif 'human' in sl: species = 'human'
        else:                species = 'unknown'
        cell_type = 'I' if ('e to i' in sl or ' i ' in sl or sl.endswith(' i')) else 'E'
        return species, cell_type
    df[['species', 'cell_type']] = df['sheet'].apply(lambda s: pd.Series(_parse(s)))
    return df


def qc_and_preprocess(species_df: pd.DataFrame, species: str):
    """Apply QC filters and return (X_scaled, df_filtered, features)."""
    avail = [f for f in CANDIDATE_FEATURES if f in species_df.columns]
    df = species_df[avail + ['cell_type']].copy()

    # ≥50% features per cell
    cell_mask = df[avail].notna().mean(axis=1) > 0.50
    df = df.loc[cell_mask]
    # ≥80% cells per feature
    feat_mask = df[avail].notna().mean(axis=0) > 0.80
    features = [f for f in avail if feat_mask[f]]

    n = len(df)
    print(f"  {n} cells, {len(features)} features after QC")

    X = df[features].values.astype(float)
    X[~np.isfinite(X)] = np.nan

    # RobustScaler → KNNImputer (shared pre-processing for all methods)
    pre = Pipeline([
        ('norm',   RobustScaler()),
        ('impute', KNNImputer(n_neighbors=min(5, n - 1))),
    ])
    X_scaled = pre.fit_transform(X)

    return X_scaled, df.reset_index(drop=True), features


def _scatter(ax, df, x_col, y_col, xlabel, ylabel):
    types_present = sorted(df['cell_type'].unique())
    for ct in types_present:
        mask = df['cell_type'] == ct
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   c=PALETTE[ct], s=40, alpha=0.85,
                   linewidths=0.3, edgecolors='white',
                   label=f'{ct} ({mask.sum()})', zorder=3)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def make_plots(species_df: pd.DataFrame, species: str) -> None:
    """Run PCA, t-SNE, and UMAP on one species and save a 1×3 figure."""
    title_base = SPECIES_MAP.get(species, species)

    X, df, features = qc_and_preprocess(species_df, species)
    n = len(df)
    if n < 5:
        print(f"  Too few cells — skipping.")
        return

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(**PCA_PARAMS)
    Y_pca = pca.fit_transform(X)
    df['pca0'], df['pca1'] = Y_pca[:, 0], Y_pca[:, 1]
    var = pca.explained_variance_ratio_
    print(f"  PCA  PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    perp = min(TSNE_PARAMS['perplexity'], n - 1)
    tsne = TSNE(perplexity=perp, n_components=2,
                random_state=TSNE_PARAMS['random_state'],
                max_iter=TSNE_PARAMS['max_iter'],
                init=TSNE_PARAMS['init'],
                learning_rate=TSNE_PARAMS['learning_rate'])
    Y_tsne = tsne.fit_transform(X)
    df['tsne0'], df['tsne1'] = Y_tsne[:, 0], Y_tsne[:, 1]

    # ── UMAP ─────────────────────────────────────────────────────────────────
    n_neighbors = min(UMAP_PARAMS['n_neighbors'], n - 1)
    umap = UMAP(n_neighbors=n_neighbors,
                min_dist=UMAP_PARAMS['min_dist'],
                random_state=UMAP_PARAMS['random_state'])
    Y_umap = umap.fit_transform(X)
    df['umap0'], df['umap1'] = Y_umap[:, 0], Y_umap[:, 1]

    # ── Figure: 1 row × 3 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title_base, fontsize=13, fontweight='bold', y=1.01)

    _scatter(axes[0], df, 'pca0',  'pca1',
             f'PC1 ({var[0]*100:.1f}%)', f'PC2 ({var[1]*100:.1f}%)')
    axes[0].set_title('PCA', fontsize=11)

    _scatter(axes[1], df, 'tsne0', 'tsne1', 't-SNE 1', 't-SNE 2')
    axes[1].set_title('t-SNE', fontsize=11)

    _scatter(axes[2], df, 'umap0', 'umap1', 'UMAP-1', 'UMAP-2')
    axes[2].set_title('UMAP', fontsize=11)

    # Shared legend on the right panel
    handles = [mpatches.Patch(color=PALETTE[ct], label=f'{ct} (n={( df["cell_type"]==ct).sum()})')
               for ct in sorted(df['cell_type'].unique())]
    axes[2].legend(handles=handles, title='Cell type',
                   frameon=True, fontsize=9, title_fontsize=9,
                   loc='best')

    fig.tight_layout()

    out_pdf = OUT_DIR / f'dimred_{species}.pdf'
    out_png = OUT_DIR / f'dimred_{species}.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_pdf}")

    # ── Also save individual panels ──────────────────────────────────────────
    for method, x_col, y_col, xl, yl in [
        ('pca',  'pca0',  'pca1',  f'PC1 ({var[0]*100:.1f}%)', f'PC2 ({var[1]*100:.1f}%)'),
        ('tsne', 'tsne0', 'tsne1', 't-SNE 1', 't-SNE 2'),
        ('umap', 'umap0', 'umap1', 'UMAP-1',  'UMAP-2'),
    ]:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        _scatter(ax2, df, x_col, y_col, xl, yl)
        ax2.set_title(f'{title_base}', fontsize=11, fontweight='bold')
        ax2.set_title(f'{title_base} — {method.upper()}', fontsize=11, fontweight='bold')
        ax2.legend(handles=handles, title='Cell type',
                   frameon=True, fontsize=9, title_fontsize=9)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / f'{method}_{species}.pdf', bbox_inches='tight')
        fig2.savefig(OUT_DIR / f'{method}_{species}.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)


def main():
    df = pd.read_csv(CSV_PATH)
    df = infer_labels(df)

    for species in ['mouse', 'NHP', 'human']:
        sub = df[df['species'] == species].reset_index(drop=True)
        print(f"\n{'='*55}")
        print(f"Species: {species}  n={len(sub)}  "
              f"types={sub['cell_type'].value_counts().to_dict()}")
        make_plots(sub, species)


if __name__ == '__main__':
    main()

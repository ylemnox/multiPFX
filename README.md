# multiPFX

Cross-species intrinsic electrophysiology feature extraction and dimensionality reduction from MIES multipatch NWB files.

Extracts 63 intrinsic ephys features per cell (subthreshold, suprathreshold firing, AP waveform, cross-sweep adaptation, and chirp/impedance) and visualizes them with PCA, t-SNE, and UMAP — one plot per species, colored by excitatory / inhibitory cell type.

---

## Project structure

```
multiPFX/
├── data/                        ← place your NWB files and metadata Excel here
│   └── README.md
├── aisynphys/                   ← trimmed aisynphys library (MIES NWB reader)
│   ├── setup.py
│   └── aisynphys/
│       ├── intrinsic_ephys.py   ← feature extraction (modified from Allen Institute original)
│       └── nwb_recordings.py
├── figures/                     ← example output figures (3 methods × 3 species)
├── extract_intrinsic.py         ← Step 1: NWB → intrinsic_features.csv
├── generate_umap.py             ← Step 2: CSV → PCA / t-SNE / UMAP figures
├── intrinsic_features.csv       ← example output (158 cells, 63 features)
├── synphys_culture_classifier.ipynb  ← reference notebook (Allen Institute)
├── feature_comparison_v2.html   ← feature coverage table (notebook vs. this pipeline)
└── pipeline_features.html       ← full list of extracted features with coverage
```

---

## Dependencies

Tested on macOS (Apple Silicon) with Python 3.10 via conda.

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays throughout |
| `pandas` | Feature DataFrames, Excel parsing |
| `h5py` | Low-level NWB (HDF5) access |
| `openpyxl` | Reading the metadata Excel workbook |
| `ipfx` | Allen Institute spike/sweep feature extractors |
| `neuroanalysis` | MIES NWB reader (`MiesNwb`, `MiesRecording`) |
| `umap-learn` | UMAP dimensionality reduction |
| `scikit-learn` | PCA, t-SNE, RobustScaler, KNNImputer |
| `matplotlib` | Figure rendering |

```bash
# 1. Create environment
conda create -n multiPFX python=3.10
conda activate multiPFX

# 2. Install all external dependencies
pip install -r requirements.txt

# 3. Install the bundled aisynphys library
cd aisynphys && pip install -e . && cd ..
```

> `ipfx` and `neuroanalysis` are Allen Institute open-source packages available on PyPI.

---

## Usage

### Step 1 — Extract features from NWB files

1. Copy your NWB files into `data/` (subdirectories are fine).
2. Copy your metadata Excel file into `data/` and rename it `meta_info.xlsx`
   (or edit `EXCEL_PATH` at the top of `extract_intrinsic.py`).

   The Excel workbook should have one sheet per recording group (e.g. `Mouse E`, `Mouse I`, `NHP E`). Each sheet lists NWB filenames and AD channel numbers. Sheet names are used to infer species and E/I cell type automatically.

3. Run:

```bash
python extract_intrinsic.py
```

Output: `intrinsic_features.csv` — one row per cell, 63 feature columns.

### Step 2 — Generate dimensionality reduction plots

```bash
python generate_umap.py
```

Output: PNGs and PDFs in `figures/` — one 1×3 panel figure (PCA / t-SNE / UMAP) per species, plus individual panels.

---

## Features extracted

| Category | Count | Examples |
|---|---|---|
| Subthreshold / passive | 4 | `sag`, `input_resistance`, `tau` |
| Threshold / F-I | 2 | `rheobase_i`, `fi_fit_slope` |
| Rheobase sweep — firing | 6 | `rheobase_avg_rate`, `rheobase_latency`, `rheobase_isi_cv` |
| Rheobase sweep — AP waveform | 8 | `rheobase_upstroke`, `rheobase_width`, `rheobase_threshold_v` |
| Hero sweep — firing | 6 | `hero_avg_rate`, `hero_adapt`, `hero_isi_cv` |
| Hero sweep — AP waveform | 8 | `hero_upstroke_downstroke_ratio`, `hero_fast_trough_v` |
| Cross-sweep adaptation ratios | 9 | `upstroke_adapt_ratio`, `width_adapt_ratio` |
| Chirp / impedance | 5 | `peak_freq_chirp`, `z_max_chirp`, `phase_at_peak_chirp` |

Full feature list with per-species coverage: see [`pipeline_features.html`](pipeline_features.html).
Comparison against the Allen Institute reference notebook: see [`feature_comparison_v2.html`](feature_comparison_v2.html).

---

## NWB compatibility

The extraction pipeline handles both NWB 1.x and NWB 2.x MIES files via monkey-patching of `neuroanalysis.MiesNwb`. It also supports oodDAQ (optimized-overlap dDAQ) acquisition, where sweeps from different channels are interleaved and contain inactive zero-voltage periods.

---

## Credits

Feature extraction builds on the Allen Institute [aisynphys](https://github.com/AllenInstitute/aisynphys) library and [ipfx](https://github.com/AllenInstitute/ipfx). The preprocessing pipeline (RobustScaler → KNNImputer → UMAP) follows `synphys_culture_classifier.ipynb` (included for reference).

# Skin Permeation Benchmark

A reproducible benchmark for predicting human skin permeability (log Kp), curated from five public
data sources, with four reconstructed literature baselines and the proposed GATE-LGBM evaluated
under shared 10-fold cross-validation.

This repository is the reproduction package for:

> **A reproducible benchmark for skin permeation prediction from curated public data and evaluation
> of foundation-model features** (in preparation).

Running the four stages in [Quick Start](#2-quick-start--reproducing-the-manuscript-results)
rebuilds the 521-compound benchmark from the raw sources and reproduces **Table 1**, **Table 3**, and
**Fig 4 / S3_Table** of the manuscript. Everything needed is in the clone: the raw data, the frozen
feature tables, and the configs.

---

## 1. Requirements

### 1.1 Environment setup

- Python 3.10 or later (recommended: Miniconda/Anaconda).
- `requirements.txt` lists the direct dependencies, pinned to the versions used to produce the
  reported results. It is a curated list rather than a full `pip freeze`; pip resolves the
  transitive dependencies.
- No Java/CDK toolchain is required. The Abdallah et al. feature table ships as a frozen CSV, so no
  JVM is needed.
- The FP-ADMET PubChem fingerprint table also ships frozen
  (`outputs/features/fpadmet_pubchem.csv`), so `scikit-fingerprints` is not needed either.
- Neither table can be rebuilt from this package — it ships no feature-rebuild stage, and all five
  feature tables are loaded as shipped. See [§3.3](#33-what-is-frozen-and-what-is-recomputed).

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

### 1.2 Raw data sources

The seven raw files under `data_raw/` are already included, so no download is needed. They originate
from the five published sources below. **Please cite the original papers when reusing the data.**
Citations are reproduced verbatim from `configs/source_registry.yaml`.

| Source | Raw file(s) in `data_raw/` | Reference |
| --- | --- | --- |
| huskin | `huskinDB.xlsx` | Waters, Laura J., and Xin Ling Quah. "Predicting skin permeability using HuskinDB." Scientific Data 9.1 (2022): 584. |
| skinpix | `SkinPix_20230620_cleanedDB.xlsx` | Chedik, Lisa, et al. "An update of skin permeability data based on a systematic review of recent research." Scientific Data 11.1 (2024): 224. |
| zeng | `Zeng et al.xlsx` | Zeng, R., Deng, J., Dang, L. et al. Correlation between the structure and skin permeability of compounds. Sci Rep 11, 10076 (2021). |
| cheruvu | `Cheruvu et al.xlsx` | H. S. Cheruvu et al., An updated database of human maximum skin fluxes and epidermal permeability coefficients for drugs, xenobiotics, and other solutes applied as aqueous solutions, Data in Brief, 42, 108242 (2022). |
| deeppk | `deeppk_skin_permeability_{train,val,test}.csv` | Myung, Y., et al. Deep-PK: deep learning for small molecule pharmacokinetic and toxicity prediction, 52, 469-475 (2024). |

> **Data reuse.** The files under `data_raw/` are redistributed from the published sources above.
> The code license in [`LICENSE`](LICENSE) governs the code in this repository and **does not
> govern the data**. Reuse of the data is subject to the terms of each original publication and
> publisher; please check those terms directly for your intended use.

---

## 2. Quick Start — Reproducing the Manuscript Results

Run the four stages below in order. Each regenerates the inputs of the next, so no pre-built
benchmark is required.

```bash
conda activate <env_name>
cd skin-permeation-benchmark

# 1. Rebuild the benchmark from the raw data  ->  outputs/final/  (Table 1 pipeline counts)
python run_merge.py

# 2. Regenerate the 10-fold splits            ->  outputs/final/splits/
python run_splits.py

# 3. Train + evaluate all 5 models            ->  runs/<run-id>/  (Table 3 CV metrics)
python run_benchmark.py \
  --run-id table3 \
  --split-methods random_10fold target_stratified_10fold \
  --folds 1 2 3 4 5 6 7 8 9 10 \
  --models zeng_svr_proxy abdallah_lgbm waters_fragment_linear fpadmet_rf gate_lgbm \
  --paper-sanity off

# 4. Trio case study                          ->  _regen/figure4/  (Fig 4 / S3_Table)
#    Use the run directory name printed by step 3.
python make_figure4.py --benchmark-run-dir runs/<run-id from step 3>
```

Steps 1–2 take well under a minute; step 3 is the longest, training all five models across 10 folds ×
2 split schemes. Table 3 metrics land in `runs/<run-id>/metrics/summary_by_model_and_split.csv`; the
Fig 4 panel and the S3_Table metrics land in `_regen/figure4/`.

`--paper-sanity off` disables an optional audit that records whether machine-readable external
validation data is available on disk. It writes `runs/<run-id>/external_validation/status.csv` and
does not affect any reported metric.

### 2.1 Models

Five models are evaluated: four reconstructed literature baselines and the proposed GATE-LGBM. The
manuscript refers to them by display name; the code and the output CSVs use the model ID. Use this
table to match Table 3 rows to `summary_by_model_and_split.csv`.

| Model ID (code, CLI, output CSVs) | Name in the manuscript | Representation | Estimator |
| --- | --- | --- | --- |
| `zeng_svr_proxy` | Zeng SVR | RDKit proxy descriptors | SVR (RBF) |
| `abdallah_lgbm` | Abdallah LGBM | CDK 2.8 curated 141-descriptor subset | LightGBM |
| `waters_fragment_linear` | Waters Linear | RDKit SMARTS functional-group counts | Linear regression |
| `fpadmet_rf` | FP-ADMET RF | PubChem-style 881-bit fingerprint | Random forest |
| `gate_lgbm` | **GATE-LGBM** | GATE predicted-property features (30) | LightGBM |

**GATE** is the foundation model whose predicted molecular properties are used as the feature
representation for `gate_lgbm`. Its weights are not part of this package; the features it
produced are shipped as a frozen table (see [§3.3](#33-what-is-frozen-and-what-is-recomputed)).

**Trio** refers to the three cosmetic active ingredients used in the small-scale external validation
of Fig 4. Their measured permeability values come from ex vivo porcine skin experiments and are
shipped in `outputs/features/hnh/trio_labels.csv`.

**Cross-validated performance.** Mean ± standard deviation across the 10 folds of both split
schemes, read from [`results/table3/summary_by_model.csv`](results/table3/summary_by_model.csv).
Lower is better for RMSE and MAE, higher for the other three.

| Model | RMSE | MAE | R² | Pearson r | Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| Zeng SVR | 0.863 ± 0.155 | 0.575 ± 0.099 | 0.417 ± 0.197 | 0.681 ± 0.116 | 0.716 ± 0.091 |
| Abdallah LGBM | 0.762 ± 0.102 | 0.524 ± 0.068 | 0.551 ± 0.097 | 0.753 ± 0.063 | 0.757 ± 0.060 |
| Waters Linear | 0.960 ± 0.099 | 0.757 ± 0.073 | 0.287 ± 0.128 | 0.547 ± 0.111 | 0.580 ± 0.090 |
| FP-ADMET RF | 0.761 ± 0.068 | 0.560 ± 0.056 | 0.553 ± 0.067 | 0.749 ± 0.046 | 0.738 ± 0.058 |
| **GATE-LGBM** | 0.725 ± 0.109 | 0.524 ± 0.080 | 0.591 ± 0.113 | 0.775 ± 0.077 | 0.783 ± 0.068 |

### 2.2 Reference outputs

You do not have to run anything to see the results. The manuscript's figures are committed under
`figures/`, and `results/` holds the machine-readable outputs of a maintainer run of the four stages.

| Figure | What it shows | File |
| --- | --- | --- |
| Fig 1 | Study overview | [`figures/fig1.png`](figures/fig1.png) |
| Fig 2 | Distribution and cross-source concordance of the benchmark | [`figures/fig2.png`](figures/fig2.png) |
| Fig 3 | GATE-LGBM under scaffold holdout | [`figures/fig3.png`](figures/fig3.png) |
| Fig 4 | Industrial case study on three cosmetic active ingredients | [`figures/fig4.png`](figures/fig4.png) |
| S1 Fig | Distribution of pooled conflict ranges | [`figures/s1_fig.png`](figures/s1_fig.png) |
| S2 Fig | PCA projections of each model's feature space | [`figures/s2_fig.png`](figures/s2_fig.png) |

```text
results/
├── table1/    benchmark_build_report.md, benchmark_conflict_filtered.csv, merged_all_records.csv
├── table3/    summary_by_model_and_split.csv, summary_by_model.csv, fold_metrics.csv,
│              leakage_checks.csv
├── figure4/   figure4_panel.png, table_s3_metrics.csv
└── figures/   the Fig 2 and S1 Fig panels as the pipeline emits them
```

Of the six figures, Fig 2, Fig 4 and S1 Fig are produced by the four stages below, and Fig 1 is a
schematic. The scripts behind Fig 3 (scaffold holdout, training fractions, gain-based feature
importance) and S2 Fig (feature-space PCA), and the descriptor-generation workflows of Supplementary
Note 1, are not part of this package; they are available from the corresponding author on request, as
stated in the manuscript's Code availability section.

`table3/leakage_checks.csv` is the evidence behind the manuscript's claim that no compound appears in
both the training and test partition of any fold. It has one row per split method and fold (20 in
total), and `run_benchmark.py` aborts the run if any row fails, so a completed run is itself a
passing check. The rows are keyed by split and fold only — they do not depend on which models ran.

These are reference copies for inspection and comparison. The pipeline never writes into `results/`;
your own run writes to `outputs/final/`, `reports/`, `runs/` and `_regen/`, which stay untracked, so a
clone remains clean after you reproduce the results. Metrics should match `results/` to well beyond
the three decimals reported in the manuscript (see the note on parallel nondeterminism in
[§4](#4-notes)).

The per-fold model files that `run_benchmark.py` persists under `runs/<run-id>/artifacts/` are **not**
committed — they total roughly 390 MB. Stage 3 regenerates them, and `make_figure4.py` consumes them
from your own run directory.

### 2.3 Automated verification (maintainer)

`verify_tables.py` re-runs the stages and diffs the results to three decimals against frozen
ground-truth values. Beyond the inputs used above it requires two directories that are kept out of
version control and are therefore **absent from a public clone**:

- `reference/` — the frozen ground-truth tables the results are diffed against.
- `outputs/final/` — the frozen benchmark and splits. Part A diffs the regenerated benchmark against
  the frozen `benchmark_conflict_filtered.csv`, and Part B trains on the frozen splits rather than on
  regenerated ones, so verification is anchored to those files rather than being fully closed-loop.

This check is therefore intended for maintainers running from the full working tree.

The shipped `reference/` CSVs still carry the model ID used before this rename. Before running the
check, rewrite the `model_id` column in your local copy of `table3_metrics.csv` and
`table_s3_metrics.csv` so that the fifth model reads `gate_lgbm`. No other column needs changing.

```bash
python verify_tables.py
```

Expected output.

```
=== Part A: Table 1 (benchmark pipeline counts) ===
Running run_merge.py into _regen/ ...
Table 1: PASS

=== Part B: Table 3 (CV metrics) ===
Running run_benchmark.py (5 models × 2 splits) ...
Run directory: verify_table3_YYMMDDHHMM
Table 3: PASS

=== Part C: Fig 4 / S3_Table (trio case study metrics) ===
Fig 4 / S3: PASS
```

If Part A, Part B, and Part C all PASS the exit code is 0; otherwise the mismatching cells are
printed and exit code 1 is returned. Part C reuses the persisted fold models produced by Part B; if
Part B fails, Part C is skipped.

### 2.4 Plugging in your own model

`run_benchmark.py` can also evaluate a user-supplied model under the exact same splits, metrics, and
output layout as the five models in [§2.1](#21-models). This hook is independent of the manuscript
reproduction — none of the reported numbers depend on it. Point
`--external-model` at a Python factory:

```bash
PYTHONPATH=/path/to/your/code python run_benchmark.py \
  --run-id my_model \
  --split-methods target_stratified_10fold \
  --folds 1 2 3 4 5 6 7 8 9 10 \
  --models external_model \
  --external-model my_package.my_module:make_adapter \
  --paper-sanity off
```

The factory is given as `package.module:factory`, is called with no arguments, and must return an
object with:

| Member | Contract |
| --- | --- |
| `model_id` | str, used as the `model_id` value in the metric CSVs |
| `requires_smiles` | bool, recorded in the per-fold adapter summary |
| `fit(train_df, seed=..., artifact_dir=...)` | trains on the fold's training rows only |
| `predict(test_df)` | returns one value per test row (`len(test_df)`) |

Both dataframes carry the benchmark columns, including `smiles_std` and the target `target_logkp`.
The target column is present in `test_df` only so the runner can score the fold — **`predict` must
not read it**, and any preprocessing needed by the model must be fitted inside `fit` on the training
rows alone. Fold predictions land in `runs/<run-id>/predictions/<split>/external_model/fold_<n>.csv`,
and any files `fit` writes land in `runs/<run-id>/artifacts/<split>/external_model/`. Add
`external_model` to `--models` to run it; combine it with the shipped model IDs to compare side by
side. Note that `--run-id` is a label, not the directory name: the actual directory is
`runs/<label>_<timestamp>/`, printed at the end of the run.

---

## 3. Directory Structure

```text
skin-permeation-benchmark/
├── README.md
├── LICENSE                      # Code license (BSD-3-Clause-LG AI Research)
├── NOTICE                       # Third-party open-source components and their licenses
├── requirements.txt             # Direct dependencies, pinned
├── run_merge.py                 # Stage 1: preprocessing pipeline      (Table 1)
├── run_splits.py                # Stage 2: split regeneration, seed=42
├── run_benchmark.py             # Stage 3: 5 models × 2 splits         (Table 3)
├── make_figure4.py              # Stage 4: trio inference              (Fig 4 / S3_Table)
├── verify_tables.py             # Maintainer regression check (needs reference/, see §2.3)
├── configs/                     # source_registry, merge_config, descriptor subset, pubchem cache
├── data_raw/                    # 7 raw data files (xlsx/csv)
├── src/skin_benchmark/          # Package source
├── figures/                     # The six manuscript figures (see §2.2)
├── results/                     # Reference outputs of a maintainer run (see §2.2)
│   ├── table1/                  # build report + benchmark tables
│   ├── table3/                  # CV metric summaries
│   ├── figure4/                 # Fig 4 panel + S3_Table metrics
│   └── figures/                 # Fig 2 and S1 Fig panels as the pipeline emits them
└── outputs/
    └── features/                # 5 frozen feature tables (basis for Table 3)
        ├── abdallah_descriptor_table.csv
        ├── fpadmet_pubchem.csv
        ├── gate_predicted_properties.csv
        ├── waters_functional_groups.csv
        ├── zeng_proxy_descriptors.csv
        └── hnh/                 # 5 frozen trio feature tables (basis for Fig 4)
            ├── trio_abdallah_descriptor_table.csv
            ├── trio_fpadmet_pubchem.csv
            ├── trio_gate_predicted_properties.csv
            ├── trio_waters_functional_groups.csv
            ├── trio_zeng_proxy_descriptors.csv
            └── trio_labels.csv  # measured values for the 3 trio compounds
```

Everything a run produces is excluded from version control, so a clean clone stays clean after you
reproduce the results: `outputs/final/` (the 521-compound benchmark and the 10-fold split
assignments), `outputs/intermediate/`, `reports/`, `runs/` and `_regen/`. The manuscript figures are
committed under `figures/`, and reference copies of the tables live in `results/`. `reference/` — the ground truth
`verify_tables.py` diffs against — is maintainer-only and also absent from a clone.

### 3.1 Package module responsibilities (`src/skin_benchmark/`)

```text
parsers/        5 source-specific raw readers (huskin, skinpix, zeng, cheruvu, deeppk)
standardize/    RDKit-based SMILES standardization
resolve/        PubChem CAS->SMILES resolution, served from a shipped cache
merge/          within-source deduplication + cross-source pooling + conflict filter
splits.py       deterministic 10-fold split generation
benchmarking/   data, features, abdallah_cdk (descriptor contract), registry (models),
                metrics, external_validation (optional audit), pipeline, reporting
reporting/      Table 1 counts, duplicate audit, figure generation
config.py, utils/  config loader + io/logging helpers
```

### 3.2 Pipeline / data flow

```text
=========================== PIPELINE / DATA FLOW ===========================

 data_raw/ (7 raw files)            configs/
 huskinDB, SkinPix, Zeng,           source_registry.yaml, merge_config.yaml,
 Cheruvu, Deep-PK {train,val,test}  pubchem_cas_cache.csv
        |                              |
        +--------------+---------------+
                       v
       +------------------------------------------+
       |  run_merge.py            -> Table 1       |
       |  skin_benchmark.pipeline:                 |
       |    parsers -> standardize -> resolve      |
       |            -> merge -> reporting          |
       +------------------------------------------+
                       |
                       v
   outputs/final/benchmark_conflict_filtered.csv      (521 compounds)
                       |
        +--------------+-------------------------------+
        |                                              |
        v                                              |
   +-------------------+                               |
   | run_splits.py     |  seed=42, deterministic       |
   +-------------------+                               |
        |                                              |
        v                                              |
   outputs/final/splits/                               |
   random_10fold/, target_stratified_10fold/           |
        |                                              |
        |   outputs/features/ (5 frozen tables:        |
        |   GATE, Abdallah, Zeng, Waters, FP-ADMET)    |
        |        |                                     |
        v        v                                     v
       +------------------------------------------------------+
       |  run_benchmark.py        -> Table 3                   |
       |  skin_benchmark.benchmarking:                        |
       |    data -> features -> registry (5 models)           |
       |         -> metrics -> reporting                      |
       +------------------------------------------------------+
                       |
                       v
   runs/<run-id>/      (CV metrics  +  persisted per-fold models)
                       |
   outputs/features/hnh/ (trio_*.csv x5 + trio_labels.csv)
        |              |
        v              v
       +------------------------------------------------------+
       |  make_figure4.py         -> Fig 4 / S3_Table         |
       |  ensemble inference over persisted folds             |
       |  (NO retraining)                                     |
       +------------------------------------------------------+
                       |
                       v
   _regen/figure4/     (figure4_panel.png + table_s3_metrics.csv)
```

### 3.3 What is frozen and what is recomputed

**Frozen (used as provided)**

- **GATE features** (`outputs/features/gate_predicted_properties.csv` and the corresponding trio
  table). The GATE model weights are not part of this package, so the features it produced are
  shipped as a frozen table.
- **Abdallah features** (`outputs/features/abdallah_descriptor_table.csv` and the trio table). CDK
  2.8 descriptor generation needs a Java/CDK toolchain, which this package does not include. The
  frozen CSV is validated against the curated 141-descriptor contract in
  `configs/abdallah_descriptor_subset.yaml` on every run.
- **FP-ADMET features** (`outputs/features/fpadmet_pubchem.csv` and the trio table). Fingerprint
  generation needs `scikit-fingerprints`, which this package does not depend on.
- **Zeng / Waters features** (and the corresponding trio tables). These are RDKit-derived and
  reproducible in principle, but are shipped frozen so that every rerun trains on identical inputs.
- **Trio labels** (`outputs/features/hnh/trio_labels.csv`), the measured permeability values for the
  three cosmetic actives of Fig 4, from ex vivo porcine skin experiments (shared with permission).

There is no feature-rebuild stage in this package. All five tables are read from
`outputs/features/`, and the four stages of [§2](#2-quick-start--reproducing-the-manuscript-results)
never write into it, so the feature inputs behind Table 3 are fixed by the clone.

**Recomputed**

- **Benchmark and splits** (`outputs/final/`): `run_merge.py` rebuilds the 521-compound benchmark
  from `data_raw/`, and `run_splits.py` regenerates the 10-fold split assignments deterministically
  with seed 42.
- **Table 1**: the benchmark-construction counts produced by `run_merge.py`.
- **Table 3**: `run_benchmark.py` reads the frozen feature tables plus the regenerated benchmark and
  splits, then trains and evaluates the models from scratch.
- **Fig 4 / S3_Table**: `make_figure4.py` reuses the fold models persisted during the Table 3 run to
  perform ensemble inference on the three trio compounds. No additional training.

---

## 4. Notes

- Only the Cheruvu source needs CAS-to-structure resolution, and the shipped
  `configs/pubchem_cas_cache.csv` covers every CAS number it contains, so stage 1 needs no network
  access. A cache miss would fall back to a live PubChem query.
- `runs/` is created by `run_benchmark.py`; `_regen/` is created by `make_figure4.py` (step 4) and
  by `verify_tables.py`. Neither overwrites the frozen `outputs/features/`.
- Metrics may differ in the last decimal place between runs. The random forest and LightGBM models
  run with `n_jobs=-1`, and the order in which parallel results are accumulated is not fixed, so
  floating-point sums vary at machine-epsilon scale. Values agree well beyond the three decimals
  reported in the manuscript.

---

## 5. Citation

If you use this benchmark or the code, please cite the manuscript:

```bibtex
@article{skin_permeation_benchmark,
  title   = {A reproducible benchmark for skin permeation prediction from curated public data
             and evaluation of foundation-model features},
  note    = {In preparation},
  year    = {2026}
}
```

This entry will be updated with the full author list, volume, and DOI once the paper is accepted.
Please also cite the five original data sources listed in [§1.2](#12-raw-data-sources).

---

## 6. License

- **Code**: BSD-3-Clause-LG AI Research License. See [`LICENSE`](LICENSE).
- **Third-party components**: the open-source packages this project depends on, together with their
  licenses and copyright notices, are listed in [`NOTICE`](NOTICE).
- **Data**: the files under `data_raw/` are redistributed from the published sources in
  [§1.2](#12-raw-data-sources) and are not covered by the code license. See the note in that section.

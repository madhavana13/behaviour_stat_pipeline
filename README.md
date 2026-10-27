# Behaviour Statistical Pipeline

A **config-driven, reproducible analysis pipeline** for behavioural assays in rodents, designed to produce **Prism-like statistical outputs and figures** while avoiding manual copy–paste workflows.

This repository was built to support **fast, defensible first-pass analyses** on complex behavioural batteries, while remaining transparent, extensible, and robust to pipeline updates.

---

## What this pipeline does

- Runs **statistics, posthoc tests, correlations, and figures** from configuration files  
- Produces **Prism-shaped output tables** (two-way ANOVA, posthoc cell means, etc.)
- Avoids manual graphing and repeated analysis errors
- Supports multiple assays with a **shared, stable analysis core**

Currently supported assays include:
- NOR (Novel Object Recognition)
- OFT (Open Field Test)
- EPM (Elevated Plus Maze)
- VIT (Voluntary Interaction Test)

---

## Core design principles

### 1. Config-driven, not hard-coded
All analyses are defined in `configs/stats.yaml`:
- which assays run
- which endpoints are analysed
- which figures are generated
- which corrections are applied

No changes to analysis code are required to add or remove endpoints.

---

### 2. One statistical unit per row (by design)
All `*_processed.csv` files are **analysis-ready**:
- one row = one animal
- between-subject factors (e.g. group, sex) are explicit
- outputs are safe for standard ANOVA-based inference

Assays with repeated measures (e.g. VIT) are **derived into wide, per-animal summaries** before analysis.

---

### 3. Lossless data handling
For assays with complex structure (e.g. VIT):
- a **canonical long file** is preserved (`data/interim/`)
- a **derived wide file** is generated for analysis (`data/processed/`)
- no raw measures are discarded
- all derived quantities are **append-only and explicitly named**

This allows future mixed-model or alternative analyses without reprocessing raw data.

---

## How to read the outputs

### Processed data

data/processed/{ASSAY}_processed.csv


- One row per animal
- Raw condition-specific columns are preserved (e.g. `hand__After__distance moved`)
- Derived within-animal contrasts are appended:
  - `delta_*` → within-condition changes
  - `did__*` → difference-in-differences (handling-specific effects)

Derived `did__*` columns are the **primary analysis endpoints** for repeated-measures assays.

---

### Statistical outputs

outputs/tables/


Includes:
- `{ASSAY}_secondary_two_way_anova.csv`
- `{ASSAY}_secondary_posthoc_cellmeans.csv`
- correlation tables (if enabled)

These tables are intentionally shaped to match **GraphPad Prism outputs**.

---

### Figures

outputs/figures/


- Auto-generated from endpoint definitions
- Consistent naming across assays
- Intended for **screening, sanity checks, and iteration**
  (not final publication figures)

---

## Requirements

- Python ≥ 3.9 with the packages in `pyproject.toml` — everything except linear/mixed models.
- **Optional: R ≥ 4.0** for `linear_models` (mixed models, ANCOVA, unequal variances).
  Only the recommended package `nlme` and `jsonlite` are used. The pipeline calls
  `Rscript` on `src/behaviour_stat_pipeline/stats/r/fit_nlme.R`; it is found via
  `linear_models.rscript` in `stats.yaml`, the `BSP_RSCRIPT` environment variable,
  `PATH`, or the standard Windows install location. If a config enables
  `linear_models` and R is missing, the run stops with an error. R and nlme versions
  are recorded in each model's provenance file. Tests needing R are skipped without it.

---

## Typical usage

Each experiment has its own config (`configs/experiments/<EXP>.yaml`) holding its paths
and its **analysis plan** (`analysis:` block): assays, endpoints, models, correlation specs
and `primary_endpoints`. `configs/stats.yaml` holds method defaults shared by all
experiments; the experiment's plan is merged over it.

Run all assays in the experiment's plan:
```bash
python scripts/run_stats.py --config configs/stats.yaml --experiment-config configs/experiments/EXP-2026-002.yaml
python scripts/run_figures.py --config configs/stats.yaml --experiment-config configs/experiments/EXP-2026-002.yaml --assays nor lm
```

Run specific assays:
```bash
python scripts/run_stats.py --config configs/stats.yaml --experiment-config configs/experiments/EXP-2026-002.yaml --assays nor lm
```

For results you will report, add `--require-clean`: the run refuses uncommitted changes
and a changed frozen analysis plan, so every table traces to a commit (`run_id` column ->
`{ASSAY}_provenance.json`).

### Primary vs exploratory endpoints

Declare the endpoints that test the experiment's hypotheses in
`analysis.primary_endpoints` (by assay instance, assay type or linear-model name).
Every output table has an `endpoint_role` column; primary and exploratory endpoints
are corrected for multiple comparisons as separate families. To show the plan was fixed
before the analysis, commit it and set `analysis_frozen_at: <commit sha>` in the
experiment config; each run records whether the plan is unchanged since then.

What this pipeline is (and is not)

This pipeline is:

    a reproducible analysis scaffold

    a way to standardise behavioural stats across assays

    a guard against manual errors

This pipeline is not:

    a black-box modelling framework

    a replacement for assay-specific interpretation

    a substitute for careful experimental design

Project status

    Config-driven stats, figures, and correlations are stable

    VIT integration uses derived, within-animal contrasts for valid inference

    The pipeline is under active development as assays are added and refined

Philosophy

    Make assumptions explicit, preserve raw information,
    and let configuration — not copy-paste — define the analysis.

VIT endpoint panel (March 2026)

This analysis includes a broad set of VIT endpoints (raw and derived) for first-pass screening and assay characterization. These results are not treated as confirmatory and will be used to define a restricted set of candidate endpoints for subsequent hypothesis-driven analyses.
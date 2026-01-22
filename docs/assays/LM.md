# Linemaze (LM)

The linemaze (LM) assay is a successive alleyway task with three zones:
- **Protected**: wide alley with walls (closed-like)
- **Center**: wide alley without walls (unprotected)
- **Narrow**: thin open alley (most aversive / open-like)

The processing pipeline preserves raw tracking outputs and appends analysis-ready derived endpoints (append-only).

## Input files

Place LM tracking summary CSVs under:

`data/raw/lm/`

Example:
- `_output_summary.csv`

LM exports include an internal identifier column (e.g., `ID10`). The pipeline supports both:
- cohort-style identifiers (e.g., `27A_ID9`)
- internal IDs (e.g., `ID10`)

## Running processing

From repo root:

```bash
python scripts/run_assay.py --assay LM --root ".\\data\\raw\\lm"


Processed output is written to:

data/processed/LM_processed.csv

outputs/logs/LM_processed_summary.txt

Processed schema (row level)

1 row per mouse_id × timepoint (single-session runs typically have a single timepoint).

Core ID columns:

mouse_id, group, sex, timepoint, assay, source_file, raw_id

Raw (pass-through) columns

The following columns are preserved from the tracking export:

time_in_arena

time_in_protected_alley, time_in_central_alley, time_in_narrow_alley

total_distance_moved

distance_moved_in_protected_alley, distance_moved_in_central_alley, distance_moved_in_narrow_alley

transitions_into_protected_alley, transitions_into_central_alley, transitions_into_narrow_alley

Derived endpoints (append-only)

These endpoints are created during LM derivation and are intended for stats/figures. All are prefixed lm_.

Time endpoints (seconds)

lm_time_total_s

lm_time_protected_s

lm_time_center_s

lm_time_narrow_s

Distance endpoints

lm_dist_total

lm_dist_protected

lm_dist_center

lm_dist_narrow

Transition counts

lm_trans_protected_n

lm_trans_center_n

lm_trans_narrow_n

Normalized metrics

lm_pct_time_narrow = lm_time_narrow_s / lm_time_total_s

lm_pct_time_center = lm_time_center_s / lm_time_total_s

Open/unprotected indices

Because the center zone is unprotected (no walls), LM includes both an EPM-analogue index and a true unprotected index:

lm_open_like_index = lm_time_narrow_s / (lm_time_narrow_s + lm_time_protected_s)

interpretable as “thin-open vs protected”

lm_time_unprotected_s = lm_time_center_s + lm_time_narrow_s

lm_pct_time_unprotected = lm_time_unprotected_s / lm_time_total_s

interpretable as “unprotected (center+narrow) vs total”

lm_pct_narrow_within_unprotected = lm_time_narrow_s / lm_time_unprotected_s

interpretable as “thin-open preference within unprotected exploration”

lm_delta_time_narrow_minus_center_s = lm_time_narrow_s - lm_time_center_s

signed narrow-vs-center contrast (Prism-friendly)

Stats + figures

LM endpoints are analyzed via config:

configs/stats.yaml controls endpoints and models

configs/figures.yaml controls plots


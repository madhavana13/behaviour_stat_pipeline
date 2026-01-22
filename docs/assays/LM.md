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

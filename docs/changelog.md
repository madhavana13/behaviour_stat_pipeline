2026-03-19
Checkpoint: assay registry + stats pipeline refactor

- Moved assay-specific ingestion logic into registry-based runners
- Simplified run_assay.py to a thin CLI
- Refactored run_stats.py into staged execution
- Validated statistical output equivalence for NOR and VIT
- Confirmed VIT has no primary one-sample output by config
- Noted prior NOR reference mismatch came from manual exclusion in processed file

2026-03-23
Checkpoint – Schema Normalization + Robust OFT Handling

### Added
- Global schema normalization layer (`common/schema.py`)
  - YAML-driven alias mapping (`configs/schema_aliases.yaml`)
  - Assay-aware header normalization at ingestion
  - Duplicate column detection after normalization
  - Logging of renamed and suspicious columns
- Auto-detection of malformed CSV headers (e.g. semicolon-delimited exports)
- New experiment config: `EXP-2025-001.yaml`
- Test for schema normalization (`tests/schema_normalisation.py`)
- Utility script: `rename_columns_new_results_v2.py`

### Changed
- `load_tracking_csv()`:
  - Now applies schema normalization before ID extraction
  - Supports assay-specific header aliasing
- `registry.py`:
  - Passes assay context into IO layer for schema normalization
- `oft.py`:
  - Replaced strict duplicate enforcement with **coalescing duplicate resolution**
  - Allows merging complementary partial records across files
  - Preserves strict failure for conflicting non-missing values
- `nor.py`:
  - Updated to be compatible with schema-normalized inputs

### Fixed
- NOR pipeline failure due to vendor header changes (space vs underscore naming)
- OFT ingestion failure due to mixed export formats
- Handling of duplicate rows arising from multi-file ingestion
- CSV parsing issues with non-standard delimiters

### Removed
- Temporary lock file:
  - `.~lock.NOR_processed.csv#`

### Notes
- Pipeline now assumes raw vendor exports may vary in header naming and format
- Schema normalization ensures canonical internal representation before QC/derivation
- Duplicate handling in OFT is now **value-aware (coalescing)** rather than purely structural

### Adding an exclusion file - rules

| column         | meaning                                       |
| -------------- | --------------------------------------------- |
| `mouse_id`     | canonical ID, matching processed data         |
| `assay`        | `*` for all assays, or `OFT2`, `NOR`, etc.    |
| `endpoint`     | `*` for all endpoints, or exact endpoint name |
| `exclude_from` | `primary`, `sensitivity`, or `all`            |
| `reason`       | required free-text reason                     |

2026-10-26
Checkpoint – Stats audit: correctness fixes, config validation, provenance, reference tests

### Changed outputs (re-run stats for any experiment analysed before this date)
- `primary_one_sample`: duplicate YAML keys silently dropped `misc.rct_di`; MISC one-sample test now runs
- Correlations: `defaults` and `default_groupings` were ignored (pooled only); now honoured
- Correlations and diagnostics now apply endpoint-level exclusions (previously only one-sample/ANOVA/posthoc did)
- Diagnostics: MAD-z flags computed within each grouping cell (previously pooled); ROUT uses the file's grouping (previously always `group`)
- VIT: duplicated endpoint removed from ANOVA and diagnostics lists (Holm family 8 → 7)
- Posthoc: `posthoc.enabled: false` is now respected
- OLS with robust/cluster SE: p-values use t(df_resid) instead of the normal (was anti-conservative); no current spec uses OLS
- Two-way ANOVA `type: 3` now uses sum-to-zero contrasts (Treatment coding gave wrong Type III tests); `type: 2` results unchanged

### Added
- Config validation (`stats/common.py`): duplicate YAML keys, unknown keys, unsupported methods and duplicated endpoints fail with a list of all problems
- Provenance (`stats/provenance.py`): `{ASSAY}_provenance.json` per run (git commit + dirty files, SHA-256 of configs and inputs, package versions); every table carries `run_id`
- `run_stats.py --require-clean`: refuse to run with uncommitted changes (use for reported results)
- Reference validation (`tests/reference/`): one-sample t/Wilcoxon, Type II/III ANOVA, posthoc, Pearson/Spearman, OLS-HC3 and p-value adjustments checked against base R 4.5.0

### Fixed
- A configured exclusion file that does not exist is now an error (previously: silently no exclusions)
- Smoke tests use the EXP-2026-002 OFT instances (OFT1–4)

### Known limitations
- Spearman p-values use the asymptotic t approximation (as R `cor.test(exact = FALSE)`); R's default for small n is exact
- `exclude_from: sensitivity` rows are loaded but no sensitivity analysis is run yet

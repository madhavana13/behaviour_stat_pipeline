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

2026-10-26 (b)
Checkpoint – Censored latencies analysed as time-to-event

### Changed outputs
- LM latencies (`latency_to_[re_]enter_{protected,narrow}_alley`) removed from `secondary_two_way_anova`;
  ANOVA dropped animals for which the event never happened (in EXP-2026-002: 1 mouse never entered the
  narrow alley, 5 never re-entered it, 4 of them IFS). They are now in `time_to_event`.

### Added
- `stats/endpoints/time_to_event.py`: missing latency in a recorded animal is right-censored at that
  animal's recording length (`time_to_event.duration_col`, e.g. `time_in_arena`), falling back to
  `assay_instances.<NAME>.trial_duration_s` in the experiment config. Animals without a recording are excluded.
  Fails if an observed latency exceeds the recording length or no duration is available.
- Outputs `{ASSAY}_time_to_event.csv`: Cox PH (Efron ties) hazard ratios + 95% CI for group and sex from the
  additive model, likelihood-ratio p-values for group, sex and the interaction (Type II analogue), and a
  log-rank test of group stratified by sex; correction across endpoints per term (default Holm).
  `{ASSAY}_time_to_event_summary.csv`: n, events, censored and Kaplan-Meier median per group and group x sex.
  HR > 1 = event happens sooner (shorter latency).
- Config validation: an endpoint listed in both `time_to_event` and `secondary_two_way_anova` is an error.
- Reference tests for censoring, Cox, LR tests, log-rank and KM medians against R `survival` 3.8.3.
- Kaplan-Meier figures (`figures/km.py`, `figures.auto_from_time_to_event`): one per time_to_event endpoint,
  sexes pooled and annotated with HR [95% CI], Cox and log-rank p (with correction named), plus a
  female | male panel version. Built from the same censoring function as the statistics; censored animals
  marked '|', n and n censored in the legend. Replaces the LM latency box plots (which dropped censored animals).

### Known limitations
- Proportional-hazards assumption is not tested yet (no Schoenfeld residual test in statsmodels).

2026-10-27
Checkpoint – Linear and linear mixed models (optional R/nlme backend)

### Added
- `linear_models` section in `stats.yaml`: per model, fixed effects, random effects (nlme syntax),
  ANCOVA covariates, a `baseline` assay instance as per-mouse covariate, unequal residual variances
  (`variance_by` -> nlme `varIdent`) and pairwise contrasts of equally weighted cell means.
- Several assay instances can be stacked as a repeated factor (`stack_as`, e.g. OFT2/OFT3/OFT4 as session).
- Engine: `nlme::lme` (random effects) or `nlme::gls` (none), REML by default; Type III F-tests with
  sum-to-zero contrasts and covariates centred at the mean over animals; nlme denominator df.
  Contrast df in mixed models = smallest df among the coefficients involved (conservative).
- Outputs in `<out>/MODELS/tables/<model>_{anova,coefficients,contrasts,random_effects,model_info}.csv`;
  `model_info` reports variance ratios, a likelihood-ratio test of unequal vs equal variances, and a
  Shapiro-Wilk test of normalised residuals. Holm across endpoints per term; Holm across contrast levels.
- Config validation: stacked repeated measures without a random effect, a baseline that is also a stacked
  level (covariate OR repeated level), unknown variables in `fixed`, and malformed contrasts are errors.
- Validation tests: gls == OLS Type III and cell contrasts (base R), lme == statsmodels MixedLM (REML),
  gls+varIdent == WLS with the estimated weights. Skipped if R is unavailable.
- Example model for EXP-2026-002: OFT2-4 as sessions with OFT1 as baseline covariate.
- Per-endpoint `transforms` (log, log1p, sqrt, logit), applied to the endpoint and its baseline; values outside
  the domain are an error. Every model table has a `transform` column (estimates are on that scale).
  EXP-2026-002: `time_in_center` log-transformed (Shapiro-Wilk p of normalised residuals 0.0001 -> 0.23).

### Known limitations
- Type III only (Type II needs the `car` package).
- No residual correlation structures (e.g. AR1) yet; the random intercept implies compound symmetry.
- No figures for model-based contrasts yet.

2026-10-27 (b)
Checkpoint – Per-experiment analysis plan; primary vs exploratory endpoints

### Changed
- Experiment configs now hold the analysis plan (`analysis:`): assays, endpoints (one-sample nulls,
  diagnostics, ANOVA, time-to-event), linear models, correlation specs and `primary_endpoints`.
  It is merged over `configs/stats.yaml`, which now holds method defaults only (456 -> 209 lines).
  Lists in the experiment replace shared ones; mappings merge.
- EXP-2025-001 and EXP-2026-002 received the settings they previously inherited; all 126 output
  tables of both experiments are identical before/after (verified), apart from the new column.
- Running without `--assays` now uses the experiment's `analysis.assays` (EXP-2026-002: OFT1-4 instead of
  the non-existent OFT/EPM).
- `run_figures.py` uses the same merged config.

### Added
- `endpoint_role` (primary/exploratory) in every output table. ANOVA, time-to-event and linear-model
  corrections across endpoints are computed within role, so primary endpoints form their own family.
  Correlations are always exploratory. With no `primary_endpoints`, everything is exploratory and
  p_adj is unchanged.
- A primary endpoint that is not analysed for its assay/model is a config error (catches typos).
- `analysis_frozen_at: <commit>`: each run compares the `analysis` block with that commit and records
  the result in provenance; a change is a warning, or an error with `--require-clean`.

### Not yet moved
- Ingestion settings in `stats.yaml` (`subjects`, `misc.raw_file` pointing at EXP-2026-002) and figure
  specs in `figures.yaml` are still shared.

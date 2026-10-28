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

2026-10-27 (c)
Checkpoint – Declared test order; pipeline guide

### Added
- `assay_instances.<NAME>.order` (position in the test battery) and optional `.day` in the experiment
  config. Stacked sessions in linear models follow `order`, never instance names; a baseline must
  come before the sessions it adjusts. Missing order stops only the affected model, with a message
  naming the fields to set; the run then exits non-zero after writing all other outputs.
  Session order is recorded in the model's provenance.
- `day` as a numeric time variable in stacked models (e.g. `fixed: "group * day"`).
- Order values must be unique integers; days numbers (config errors otherwise).
- `docs/pipeline_guide.md`: end-to-end guide; `scripts/cheatsheet.txt` reorganised by workflow step.

### Action needed
- EXP-2026-002: fill `order` (and optionally `day`) for every assay instance. Until then
  `oft_post_sessions` is not fitted.

2026-10-27 (d)
Checkpoint – Sucrose splash test (SPL) assay

### Added
- New assay type `SPL` (sucrose splash test), ingested with the same parameters as OFT:
  `SPLRunner` reuses OFT's ingestion, classifier merge, duplicate coalescing, QC and derivation
  (messages now name the assay); `schema_aliases.yaml -> spl` is a YAML alias of the OFT mapping;
  SPL rows in `labels.csv`; `design.within_factors.spl`.
- Test: the same raw export ingested as OFT and as SPL gives identical processed tables apart from
  columns that only carry the assay name.
- No experiment uses SPL yet; add it under `assay_instances` and `analysis` (see guide, section 4).

2026-10-27 (e)
Checkpoint – Remove hardcoded assay details and the 2 x 2 assumption

### Changed (no change to existing results)
- All 12 processed files of EXP-2025-001 and EXP-2026-002 and all 121 stats tables are identical
  before/after (verified; run_id aside).

### Added
- NOR: object types are configured (`assay_instances.NOR.objects`), no longer hardcoded as
  flask / lego_tower; any number of copies per object; FAM/TEST file patterns (`fam_pattern`,
  `test_pattern`) and the low-exploration threshold (`min_explore_s`) are configurable. TEST data in
  more than one copy of an object is an error.
- VIT: timepoint/phase are read from the file name via `assay_instances.VIT.filename_tokens`;
  a file name matching none or several tokens is an error (previously first match won).
- Runners receive their `assay_instances.<NAME>` settings. `classifier_pattern` in the experiment
  config is now honoured (it was silently ignored in favour of the default).
- ANOVA: one or two factors, any number of levels (e.g. single-sex studies, three groups).
- Posthoc: one or two factors, any number of levels; cells named by level or by position;
  `comparisons: [{factor, by}]` generates all pairs.
- Time to event: one or two factors, any number of levels; omnibus LR test (df = levels − 1) plus
  per-level hazard ratios; unstratified log-rank with one factor.
- Figures: two-group bracket only for two groups (omnibus result as text otherwise); KM by-sex
  figure only when sex is a factor.
- Reference tests for one-way ANOVA, 3 x 2 Type II/III, generated posthoc pairs, and Cox/log-rank
  with a three-level factor (one and two factors) against base R / survival; ingestion tests for
  configurable NOR objects and VIT tokens.

### Known issues
- Group colours in figures.yaml are shared across experiments: `ifs` and `stressed` are both
  darkorange (to be fixed with per-experiment group order/labels/colours).

2026-10-27 (f)
Checkpoint – Group order, labels and colours declared once per experiment

### Changed
- New experiment-config block `groups: {order, labels, colors}` replaces: experiment `group_order`,
  figures.yaml `defaults.group_order` / `group_edgecolors` / per-figure `group_order`,
  stats.yaml `factors.*.order` and `factor_orders.group`, and the hardcoded "IFS" label.
  Setting group order/colours anywhere else is now an error.
- EXP-2025-001 figures now show its declared order (cup, tail); before, figures.yaml's shared order
  put tail first. All 121 stats tables unchanged (posthoc already ordered cup, tail alphabetically).
- Groups without a colour get distinct Okabe-Ito colours; duplicate colours, undeclared groups in the
  data, and the old `group_order` key are errors. (figures.yaml had ifs and stressed both darkorange.)
- EXP-2025-001 plan: removed the 8 OFT `classifier__*` endpoints, which do not exist in that
  experiment's OFT export (they were skipped with a warning by stats and made run_figures fail).
  Its export records rearing as `supported_rearing_time` etc.; those 4 rearing endpoints (time and
  count, supported and unsupported) were then ADDED to its OFT ANOVA and diagnostics (labels added to
  labels.csv). This changes EXP-2025-001 OFT results: 4 new endpoints, and the Holm family per term
  grows from 4 to 8 endpoints, so p_adj of the existing OFT endpoints increases.

2026-10-27 (g)
Checkpoint - Remaining hardcoded / dead / cumbersome parts

### Changed
- Output layout: experiment configs set `paths.out_dir`; results go to `<out_dir>/<ASSAY>/{tables,figures,logs}`
  and `<out_dir>/MODELS`, exploratory scripts to `<out_dir>/exploratory`. `out_tables/out_figures/out_logs`
  (not used as written: the code wrote to their parent folder) are now an error.
- Smoke tests write to a temporary directory; tests no longer overwrite `outputs/`.
- `is_after` removed from OFT/EPM/LM processed files (always 0; session order is declared in the
  experiment config). VIT keeps it (Before/After phases).
- Removed settings that had no effect: `primary_one_sample.collapse_over`, `optional.cluster_col`,
  `design.within_factors`, `subjects.groups` (ingestion now checks subjects against `groups.order`),
  `misc.raw_file` (MISC raw path comes from the experiment config), `id_cols` (the id column is always
  `mouse_id`; any other value is an error). Unknown top-level keys in the merged config are errors.
- ROUT thresholds configurable (`diagnostics.rout_q`, `rout_min_n`); all diagnostics thresholds written
  out in stats.yaml.
- `run_figures.py` defaults to the experiment's `analysis.assays`.
- Scripts must be run from the repository root (clear error otherwise); schema aliases are found
  relative to the repository.

### Added
- Sensitivity analysis: when `exclude_from: primary/sensitivity` rows make the two exclusion sets differ
  for an assay, one-sample, ANOVA, posthoc and time-to-event are re-run and written as `*_sensitivity.csv`.

### Removed / moved
- Dead modules: stats/endpoints/{epm,oft,vit}_endpoints.py (empty), simple_effects.py, nor_endpoints.py,
  stats/report.py, figures/cellmeans.py.
- One-off analyses moved to scripts/exploratory/, superseded ingestion scripts to scripts/legacy/.

### Internal
- One significance-label function; one helper for missing-endpoint warnings; one correlation-spec
  resolver shared by stats and figures; PlotStyle built by one generic override (two ~45-line
  field-by-field copies removed).

### Verified
- All 121 stats tables and all 158 figures byte-identical before/after; processed files identical
  except the removed `is_after` column.

### Not done
- Figures still read the configuration as a plain dict while the statistics use the typed StatsConfig
  (two representations). Correlation specs and group settings are shared now; a single typed config
  is a larger refactor.

2026-10-28
Checkpoint - Secondary endpoint role, correction per role, posthoc gating

### Added
- `analysis.secondary_endpoints` (experiment config): preregistered outcomes that did not set the
  sample size. `endpoint_role` is now `primary`, `secondary` or `exploratory`; each role is a separate
  correction family per model term. Declaring an endpoint both primary and secondary, or declaring one
  that is not analysed, is a config error.
- `correction_across_endpoints` (ANOVA, time-to-event, linear models) accepts a mapping per role.
  New defaults in stats.yaml: `{primary: holm, secondary: fdr_bh, exploratory: fdr_bh}`. ANOVA tables
  now carry `correction_method`.
- Posthoc gate (`secondary_two_way_anova.posthoc.gate_on_anova`, default true): a comparison is
  labelled significant only if a corrected ANOVA term involving the compared factor has p_adj < 0.05;
  otherwise `anova_gate = failed`, label `ns (ANOVA)`.

### Changed (results)
- Exploratory ANOVA / time-to-event / linear-model endpoints: Holm -> BH-FDR (smaller p_adj).
- Posthoc comparisons not supported by the corrected ANOVA lose their stars. Example, EXP-2025-001
  EPM: count_psap and total_distance_moved `group_within_sex_1` (Šidák p = 0.017, 0.016) were
  labelled <0.05 while the corrected group term was not significant.

### Background
- Multiple correction was switched on in cf201a2 (2026-07-13: Holm per term across all of an assay's
  endpoints). Before that, `correction_across_endpoints` defaulted to `none` and ANOVA labels used raw p.
  That change was not recorded here; it is why e.g. EPM head dips / PSAP stopped being labelled
  significant.

### Not changed
- `manual_brackets` labels in configs/figures.yaml are typed by hand and are not checked against any
  table.

2026-10-28 (b)
Checkpoint - Social interaction test (SIT) assay

### Added
- SIT assay (`assays/sit.py`, `SITRunner`): FAM (empty mesh cage, 2 min) and TEST (target mouse,
  5 min) files matched per mouse, like NOR. `di_iz` / `di_explore` (and `pi_*`) compare the first
  2 min of TEST (export's "(first 2 min)" columns) with FAM, each as a fraction of tracked time
  (`total_time_in_arena`; TEST capped at `test_window_s`); positive = preference for the target.
  Advisory flags for low interaction-zone time and short recordings.
- `target_col`: subject-metadata column naming the target mouse, copied to `target_id`.
- `FamTestRunner`: shared FAM/TEST ingestion for NOR and SIT (NOR output unchanged).
- Raw IDs with `HAB` as the familiarisation token (`..._27A_HAB_ID4`) are parsed.
- EXP-2025-001: SIT ingestion and analysis plan (one-sample DI vs 0, ANOVA, and the mixed model
  `sit_target`: group * sex + random intercept per target mouse). Each target met 4 test mice of one
  group x sex cell (2 targets per cell), so the model, not the ANOVA, is the SIT analysis to report;
  the preregistered secondary endpoints are declared on it. Exclusions file added
  (`metadata/exclusions.csv`: 27B_ID14 from `test_total_distance_moved`).

### Fixed
- `schema_aliases.yaml` had an empty `sit: raw: {}` placeholder that silently replaced any real
  SIT mapping (duplicate YAML key, last one wins). Schema aliases, and the configs read by
  `run_assay.py` and `run_figures.py`, are now loaded with the strict loader (duplicate keys are
  an error), as the stats configs already were. Loader moved to `common/yaml_io.py`.

### Data notes (EXP-2025-001)
- 32 mice. 27B_ID14's TEST recording is 184.8 s tracked (median 288 s): first-2-min measures
  and the DI are unaffected; excluded from `test_total_distance_moved`.
- FAM tracked time is ~109 s (106-110 s), not 120 s; hence the DI on fractions of tracked time.
- In `sit_target`, sex is confounded with target sex (male test mice met male targets), and group
  and sex are tested on 4 denominator df (8 targets).
- Not handled: the one-sample DI tests and the SIT ANOVA treat mice sharing a target as independent.

### Also
- NOR `test_familiar_explore_time` label added (endpoint added to the NOR analysis lists earlier today).

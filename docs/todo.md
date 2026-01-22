# Behaviour Stats Pipeline — TODO / Roadmap

This document tracks known limitations, planned improvements, and polish items
for the behavioural statistics and figure pipeline.

Items are grouped by priority and subsystem.

---

## High priority

### Figures: labels and axis handling

- [ ] Implement label mapping for figures
  - Map raw values to display labels (e.g. cup → Cup, tail → Tail)
  - Apply to:
    - group labels
    - sex labels
    - auto-generated x-axis labels
  - Config-driven via:
    stats.yaml → figures.defaults.label_maps
  - Motivation:
    avoid hard-coding capitalization into metadata files

- [ ] Improve handling of single-level x-axes
  - Auto-generated endpoint plots currently use a single x-level ("Total"/"All")
  - Desired behavior:
    - if only one x-level exists, suppress x-axis label
    - plot by group only
  - Current workaround:
    rename "Total" → "All" in auto specs

---

### Figures: statistical annotations

- [ ] Add significance annotations to boxplots
  - Read directly from stats outputs:
    - *_secondary_two_way_anova.csv
    - *_secondary_posthoc_cellmeans.csv
  - Prism-style annotations:
    - stars (*, **, ***)
    - optional brackets for pairwise comparisons
  - Configurable per assay and per endpoint
  - Must fail loudly if stats are missing or ambiguous

---

## Medium priority

### Figures: time-series and line plots

- [ ] Add line / timecourse plot support
  - For assays with ordered phases (e.g. VIT)
  - New figure types:
    - linepoints
    - line_mean_sem
  - Support:
    - within-subject trajectories
    - group mean ± SEM
  - Fully config-driven via figures.specs

---

### Stats: correlations and exploratory models

- [ ] Extend correlation figures
  - Optional regression lines per subgroup
  - Grouping options:
    - pooled
    - by group
    - by sex
  - Config flags:
    correlations.auto_from_correlations.fit.by

- [ ] Optional nonparametric tests
  - Mann–Whitney U
  - Kruskal–Wallis
  - Explicit opt-in per endpoint
  - No silent switching based on assumptions

---

## Low priority

### Config ergonomics and validation

- [ ] Add explicit config validation
  - Catch typos and missing keys in stats.yaml
  - Fail early with clear error messages
  - Prefer lightweight schema validation

- [ ] Improve defaults for single-session assays
  - Auto-detect non-informative timepoints
  - Reduce boilerplate in stats and figures configs

---

### Documentation

- [ ] Assay-level documentation
  - LM (done)
  - OFT
  - EPM
  - NOR
  - VIT

- [ ] End-to-end worked example
  - Raw data → processed → stats → figures
  - Minimal dataset
  - Minimal config snippet

---

### Engineering and robustness

- [ ] Add smoke tests for figure generation
  - Ensure figures run on minimal datasets
- [ ] Add CI check for imports and config parsing
- [ ] Maintain a changelog for behavior-affecting changes

---

## Design constraints (do not violate)

- One YAML (stats.yaml) is the single source of truth for:
  - statistics
  - figures
  - correlations
- Derived data must be append-only
- No silent statistical behavior changes
- Prefer config over branching logic
- Shared code paths must remain backward compatible

---

## Notes

This roadmap is intentionally conservative.
Items are added only when motivated by real analysis needs,
not speculative completeness.

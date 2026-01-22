High-priority (next iterations)
Figures: label handling & UX

 Implement label_maps support for figures

Map raw values → display labels (e.g. cup → Cup, tail → Tail)

Apply to:

group labels

sex labels

auto-generated x-axis labels

Config-driven via stats.yaml → figures.defaults.label_maps

Motivation: avoid hard-coding capitalization into metadata

 Hide or simplify single-level x-axes

For auto-generated endpoint plots, avoid showing redundant labels (formerly "Total")

Current workaround: rename "Total" → "All" in auto specs

Desired behavior: if only one x-level exists, label by group only

Figures: significance annotations

 Add significance stars / brackets to boxplots

Read directly from stats outputs:

*_secondary_two_way_anova.csv

*_secondary_posthoc_cellmeans.csv

Prism-style annotations:

*, **, ***

optional brackets for pairwise comparisons

Configurable per assay / endpoint

Should fail gracefully if stats are missing

Medium-priority (feature completeness)
Figures: time-series / line plots

 Add line / timecourse plot support

For assays with ordered phases (e.g. VIT: first min → second min)

New figure type: linepoints or line_mean_sem

Support:

within-subject trajectories

group mean ± SEM

Config-driven via figures.specs

Stats: correlations & exploratory models

 Extend correlations auto-figures

Allow grouping by sex / group in scatter plots

Optional regression lines per subgroup

Config flags:

fit.by: pooled | group | sex

 Optional nonparametric tests

Add support for:

Mann–Whitney U

Kruskal–Wallis

Explicit opt-in per endpoint (avoid silent switching)

Low-priority (polish / robustness)
Config ergonomics

 Validate config schema explicitly

Catch typos in stats.yaml early

Helpful error messages for missing keys

Possibly via a lightweight schema validator

 Better defaults for single-session assays

Auto-detect when timepoint is non-informative

Reduce boilerplate in figures/stats configs

Documentation

 Assay-level READMEs

LM: done

 OFT

 EPM

 NOR

 VIT

 End-to-end example

Raw data → processed → stats → figures

Minimal dataset + config snippet

Repo / engineering hygiene

 Add smoke tests for figures

Ensure figure generation doesn’t crash on minimal data

 Add CI check for config + import sanity

 Versioned changelog

Track behavior-affecting changes (especially stats/figures)

Design principles (do not violate)

One YAML (stats.yaml) is the single source of truth for:

stats

figures

correlations

Derived data is append-only

No silent statistical behavior changes

Prefer config over code branching

Shared code paths must remain backward compatible

Nice-to-have / aspirational (py3r polish)

 Interactive HTML figure export

 Small multiples for multi-endpoint assays

 Metadata-aware color palettes

 Automatic figure captions (methods-ready)

Notes

This list is intentionally conservative.
Items are added only when they solve a real pain point observed during analysis, not for speculative completeness.
# Exploratory scripts

One-off analyses that sit outside the pipeline. They are **not** validated like the pipeline
statistics (no reference tests, no provenance stamping) and some contain experiment-specific
choices (e.g. `cross-assay_locomotor_consistency.py` is written for EXP-2026-002, OFT2 vs LM).
Treat their output as exploratory. If one becomes part of the analysis you report, move it into
the pipeline as a configurable stage.

Run from the repository root, e.g.

    python scripts/exploratory/nor_check_individual_change.py --experiment-config configs/experiments/EXP-2026-002.yaml

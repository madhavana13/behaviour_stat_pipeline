2026-03-19
Checkpoint: assay registry + stats pipeline refactor

- Moved assay-specific ingestion logic into registry-based runners
- Simplified run_assay.py to a thin CLI
- Refactored run_stats.py into staged execution
- Validated statistical output equivalence for NOR and VIT
- Confirmed VIT has no primary one-sample output by config
- Noted prior NOR reference mismatch came from manual exclusion in processed file
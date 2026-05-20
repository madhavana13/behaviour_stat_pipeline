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
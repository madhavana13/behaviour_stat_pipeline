from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from behaviour_stat_pipeline.io import (
    extract_mouse_id,
    load_subjects,
    load_tracking_csv,
    merge_subject_metadata,
)
from behaviour_stat_pipeline.common.schema import normalize_headers
from behaviour_stat_pipeline.common.derive import save_processed
from behaviour_stat_pipeline.common.qc import warn_missing_subjects_coverage

from behaviour_stat_pipeline.assays.vit import (
    qc_vit,
    derive_vit_metrics,
    VITConfig,
    vit_missing_cells_report,
)
from behaviour_stat_pipeline.assays.oft import (
    qc_oft,
    derive_oft_metrics,
    OFTConfig,
)
from behaviour_stat_pipeline.assays.epm import (
    qc_epm,
    derive_epm_metrics,
    EPMConfig,
)
from behaviour_stat_pipeline.assays.nor import (
    derive_nor_metrics,
    NORConfig,
    summarize_nor_sanity,
)
from behaviour_stat_pipeline.assays.lm import (
    qc_lm,
    derive_lm_metrics,
    LMConfig,
)

from behaviour_stat_pipeline.assays.misc import (
    read_misc_csv,
    qc_misc,
    apply_misc_masks,
    derive_misc_metrics,
    summarize_misc,
)

def subjects_validation_from_cfg(cfg: dict) -> tuple[set[str] | None, bool, set[str], bool]:
    sub = cfg.get("subjects", {})

    groups_cfg = sub.get("groups", {})
    groups_enabled = bool(groups_cfg.get("enabled", False))
    allowed_groups = set(map(str.lower, groups_cfg.get("allowed", []))) if groups_enabled else None
    strict_groups = bool(groups_cfg.get("strict", False))

    sex_cfg = sub.get("sex", {})
    allowed_sex = set(map(str.lower, sex_cfg.get("allowed", ["male", "female"])))
    strict_sex = bool(sex_cfg.get("strict", True))

    return allowed_groups, strict_groups, allowed_sex, strict_sex


@dataclass(frozen=True)
class RunContext:
    root: Path
    pattern: str
    subjects_path: str
    raw_id_col: str
    out: str
    cfg: dict
    schema_aliases: dict
    assay_instance: str = ""
    session: str = ""
    # assay_instances.<NAME> from the experiment config (patterns, objects, filename tokens, ...)
    instance_cfg: dict = field(default_factory=dict)


class BaseAssayRunner:
    assay_name: str = ""
    default_output: str = ""
    default_pattern: str = "*_output_summary.csv"

    def find_files(self, root: Path, pattern: str) -> list[Path]:
        files = sorted(root.rglob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found under {root} matching {pattern}")
        return files

    def load_subjects_df(self, ctx: RunContext) -> pd.DataFrame:
        allowed_groups, strict_groups, allowed_sex, strict_sex = subjects_validation_from_cfg(ctx.cfg)
        return load_subjects(
            ctx.subjects_path,
            allowed_groups=allowed_groups,
            strict_groups=strict_groups,
            allowed_sex=allowed_sex,
            strict_sex=strict_sex,
        )

    def load_one_file(self, f: Path, subjects: pd.DataFrame, raw_id_col: str) -> pd.DataFrame:
        df = load_tracking_csv(
            f,
            raw_id_col=raw_id_col,
            assay=self.assay_name,
        )
        df = merge_subject_metadata(df, subjects, on_missing_subjects="warn_drop")
        df["assay"] = self.assay_name
        df["assay_type"] = self.assay_name
        df["assay_instance"] = ""
        df["session"] = ""
        df["source_file"] = str(f)
        return df
    
    def classifier_pattern(self, ctx: RunContext) -> str:
        # Experiment instance setting first (it used to be ignored), then shared config, then default
        return str(
            ctx.instance_cfg.get("classifier_pattern")
            or ctx.cfg.get("classifier_pattern")
            or "*_Report_summary.csv"
        )

    def find_classifier_files(self, ctx: RunContext) -> list[Path]:
        pattern = self.classifier_pattern(ctx)
        if not pattern:
            return []
        return sorted(ctx.root.rglob(pattern))

    def load_classifier_file(
        self,
        f: Path,
        subjects: pd.DataFrame,
        ctx: RunContext,
    ) -> pd.DataFrame:
        """
        Load optional classifier/report summaries.

        Expected structure:
        - semicolon-delimited CSV
        - a 'file' column containing a path or filename with the mouse_id
        """
        report = pd.read_csv(f, sep=";")

        if "file" not in report.columns:
            raise ValueError(f"Classifier/report file missing required 'file' column: {f}")

        report = report.copy()
        report["raw_id"] = report["file"].astype(str).str.strip()
        report["mouse_id"] = report["raw_id"].apply(extract_mouse_id)

        report, rename_log, suspicious = normalize_headers(
            report,
            assay=self.assay_name,
            schema=ctx.schema_aliases,
        )

        if rename_log:
            print(
                f"[Schema] {f.name}: renamed {len(rename_log)} classifier columns "
                f"for assay={self.assay_name}"
            )
            for old, new in rename_log[:20]:
                print(f"  {old} -> {new}")
            if len(rename_log) > 20:
                print(f"  ... and {len(rename_log) - 20} more")

        if suspicious:
            print(
                f"[Schema WARNING] {f.name}: suspicious classifier columns "
                f"retained unchanged: {suspicious}"
            )

        report["assay"] = self.assay_name
        report["assay_type"] = self.assay_name
        report["assay_instance"] = ctx.assay_instance or self.assay_name
        report["session"] = ctx.session or ctx.assay_instance or self.assay_name
        report["source_file"] = str(f)
        report["timepoint"] = getattr(self, "timepoint_label", "Single")

        return report

    def merge_classifier_summary(
        self,
        primary: pd.DataFrame,
        classifier: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge optional classifier/report summaries into the primary assay table.

        Contract
        --------
        - Both primary and classifier are raw-derived tables.
        - All non-key columns from both tables are retained.
        - The only unprefixed shared columns are the merge keys.
        - Classifier non-key columns are namespaced with 'classifier__'
        before merging, so no column is overwritten, dropped, or silently suffixed.
        """
        key_cols = ["mouse_id", "timepoint"]
        key_set = set(key_cols)

        missing_keys = [
            c for c in key_cols
            if c not in primary.columns or c not in classifier.columns
        ]
        if missing_keys:
            raise ValueError(
                f"Cannot merge classifier summary for {self.assay_name}; "
                f"missing key columns: {missing_keys}"
            )

        if primary.duplicated(subset=key_cols).any():
            dups = primary.loc[
                primary.duplicated(subset=key_cols, keep=False),
                key_cols,
            ]
            raise ValueError(
                f"Primary rows are not unique for {self.assay_name} "
                f"at mouse_id × timepoint:\n{dups.to_string(index=False)}"
            )

        if classifier.duplicated(subset=key_cols).any():
            dups = classifier.loc[
                classifier.duplicated(subset=key_cols, keep=False),
                key_cols,
            ]
            raise ValueError(
                f"Classifier/report rows are not unique for {self.assay_name} "
                f"at mouse_id × timepoint:\n{dups.to_string(index=False)}"
            )

        classifier_renamed = classifier.copy()

        rename_map = {
            col: f"classifier__{col}"
            for col in classifier_renamed.columns
            if col not in key_set
        }

        classifier_renamed = classifier_renamed.rename(columns=rename_map)

        remaining_overlap = sorted(
            (set(primary.columns) & set(classifier_renamed.columns)) - key_set
        )

        if remaining_overlap:
            raise ValueError(
                f"Internal merge invariant failed for {self.assay_name}: "
                f"non-key columns still overlap after classifier namespacing: "
                f"{remaining_overlap}"
            )

        merged = primary.merge(
            classifier_renamed,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )

        classifier_value_cols = [
            c for c in classifier_renamed.columns
            if c not in key_set
        ]

        if classifier_value_cols:
            missing_classifier = merged[classifier_value_cols].isna().all(axis=1)
            if missing_classifier.any():
                missing_ids = (
                    merged.loc[missing_classifier, "mouse_id"]
                    .astype(str)
                    .tolist()
                )
                print(
                    f"WARNING [{self.assay_name}]: some primary rows had no matching "
                    f"classifier/report row: {missing_ids}"
                )

        return merged
    
    def _load_session(
        self,
        files: list[Path],
        session: str,
        subjects: pd.DataFrame,
        raw_id_col: str,
    ) -> pd.DataFrame:
        dfs = []
        for f in files:
            df = self.load_one_file(f, subjects, raw_id_col)
            df["timepoint"] = "Single"
            df["session"] = session
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def build_long(self, ctx: RunContext, subjects: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        files = self.find_files(ctx.root, ctx.pattern)

        dfs = []
        for f in files:
            df = self.load_one_file(f, subjects, ctx.raw_id_col)
            df["assay_instance"] = ctx.assay_instance or self.assay_name
            df["session"] = ctx.session or ctx.assay_instance or self.assay_name
            dfs.append(self.annotate_loaded_df(df, f, ctx))

        primary = pd.concat(dfs, ignore_index=True)

        classifier_files = self.find_classifier_files(ctx)
        if not classifier_files:
            return primary, len(files)

        classifier_dfs = [
            self.load_classifier_file(f, subjects, ctx)
            for f in classifier_files
        ]
        classifier = pd.concat(classifier_dfs, ignore_index=True)

        merged = self.merge_classifier_summary(primary, classifier)
        return merged, len(files) + len(classifier_files)

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path, ctx: RunContext) -> pd.DataFrame:
        return df

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run_qc(self, long_df: pd.DataFrame) -> None:
        return None

    def extra_summary_lines(self, long_df: pd.DataFrame, subjects: pd.DataFrame) -> list[str]:
        coverage_msg = warn_missing_subjects_coverage(
            long_df["mouse_id"], subjects, where=self.assay_name
        )
        return [coverage_msg] if coverage_msg else []

    def output_path(self, out: str) -> Path:
        return Path(out) if out else Path(self.default_output)

    def run(self, ctx: RunContext) -> None:
        subjects = self.load_subjects_df(ctx)
        long_df, files_n = self.build_long(ctx, subjects)
        self.run_qc(long_df)
        processed = self.derive_processed(long_df)

        out_path = self.output_path(ctx.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary_assay = ctx.assay_instance or self.assay_name

        res = save_processed(
            processed,
            out_csv=out_path,
            assay=summary_assay,
            files_n=files_n,
            extra_lines=[m for m in self.extra_summary_lines(long_df, subjects) if m],
        )
        print(f"Saved: {res.out_csv}")
        print(f"Summary: {res.out_summary}")


# Used when assay_instances.VIT.filename_tokens is not set: filename token -> value
DEFAULT_VIT_FILENAME_TOKENS = {
    "timepoint": {"first_min": "Before", "second_min": "After"},
    "phase": {"hand": "hand", "standing": "standing"},
}


def _token_value(name: str, tokens: dict[str, str], what: str, filename: str) -> str:
    """Exactly one token of `tokens` must occur in the (lower-case) file name."""
    hits = [t for t in tokens if str(t).lower() in name]
    if len(hits) != 1:
        raise ValueError(
            f"Cannot infer VIT {what} from filename {filename!r}: "
            f"{'none' if not hits else 'several'} of the tokens {list(tokens)} found"
            f"{'' if not hits else ' (' + ', '.join(hits) + ')'}. "
            "Set assay_instances.VIT.filename_tokens in the experiment config."
        )
    return str(tokens[hits[0]])


class VITRunner(BaseAssayRunner):
    assay_name = "VIT"
    default_output = "data/processed/VIT_processed.csv"

    @staticmethod
    def infer_vit_timepoint_phase_from_filename(p: Path, tokens: dict | None = None) -> tuple[str, str]:
        tokens = tokens or DEFAULT_VIT_FILENAME_TOKENS
        name = p.name.lower()
        timepoint = _token_value(name, tokens["timepoint"], "timepoint", p.name)
        phase = _token_value(name, tokens["phase"], "phase", p.name)
        return timepoint, phase

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path, ctx: RunContext) -> pd.DataFrame:
        tp, ph = self.infer_vit_timepoint_phase_from_filename(
            source_file, ctx.instance_cfg.get("filename_tokens")
        )
        df["timepoint"] = tp
        df["phase"] = ph
        return df

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_vit(long_df)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_vit_metrics(long_df, cfg=VITConfig())

    def extra_summary_lines(self, long_df: pd.DataFrame, subjects: pd.DataFrame) -> list[str]:
        lines = super().extra_summary_lines(long_df, subjects)
        completeness_msg = vit_missing_cells_report(long_df)
        if completeness_msg:
            lines.append(completeness_msg)
        return lines


class SingleSessionRunner(BaseAssayRunner):
    timepoint_label = "Single"

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path, ctx: RunContext) -> pd.DataFrame:
        df["timepoint"] = self.timepoint_label
        return df


class OFTRunner(SingleSessionRunner):
    assay_name = "OFT"
    default_output = "data/processed/OFT_processed.csv"

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_oft(long_df)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_oft_metrics(long_df, cfg=OFTConfig())


class SPLRunner(OFTRunner):
    """
    Sucrose splash test (SPL): same ingestion parameters as OFT (EthoVision summary +
    classifier report, one session per mouse). Column aliases: schema_aliases.yaml -> spl,
    which mirrors oft.
    """
    assay_name = "SPL"
    default_output = "data/processed/SPL_processed.csv"

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_oft(long_df, label=self.assay_name)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_oft_metrics(long_df, cfg=OFTConfig(), label=self.assay_name)


class EPMRunner(SingleSessionRunner):
    assay_name = "EPM"
    default_output = "data/processed/EPM_processed.csv"

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_epm(long_df)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_epm_metrics(long_df, cfg=EPMConfig())


class LMRunner(SingleSessionRunner):
    assay_name = "LM"
    default_output = "data/processed/LM_processed.csv"

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_lm(long_df)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_lm_metrics(long_df, cfg=LMConfig())


class NORRunner(BaseAssayRunner):
    assay_name = "NOR"
    default_output = "data/processed/NOR_processed.csv"
    
    def nor_config(self, ctx: RunContext) -> NORConfig:
        objects = ctx.instance_cfg.get("objects")
        if not objects:
            raise ValueError(
                "NOR: the two object types are not configured. Set assay_instances.NOR.objects in the "
                "experiment config, e.g. objects: [flask, lego_tower] (names as in the export columns)."
            )
        return NORConfig(objects=tuple(objects), min_explore_s=float(ctx.instance_cfg.get("min_explore_s", 20.0)))

    def run(self, ctx: RunContext) -> None:
        subjects = self.load_subjects_df(ctx)
        cfg = self.nor_config(ctx)

        fam_pattern = ctx.instance_cfg.get("fam_pattern", "*FAM*_output_summary.csv")
        test_pattern = ctx.instance_cfg.get("test_pattern", "*TEST*_output_summary.csv")
        fam_files = sorted(ctx.root.rglob(fam_pattern))
        test_files = sorted(ctx.root.rglob(test_pattern))

        if not fam_files:
            raise FileNotFoundError(f"No FAM files found under {ctx.root} matching {fam_pattern} (assay_instances.NOR.fam_pattern)")
        if not test_files:
            raise FileNotFoundError(f"No TEST files found under {ctx.root} matching {test_pattern} (assay_instances.NOR.test_pattern)")
        overlap = set(fam_files) & set(test_files)
        if overlap:
            raise ValueError(f"NOR: files match both fam_pattern and test_pattern: {sorted(map(str, overlap))}")

        fam = self._load_session(fam_files, "FAM", subjects, ctx.raw_id_col)
        test = self._load_session(test_files, "TEST", subjects, ctx.raw_id_col)

        processed = derive_nor_metrics(
            fam,
            test,
            schema_aliases=ctx.schema_aliases,
            cfg=cfg,
        )
        sanity = summarize_nor_sanity(processed, min_total_s=cfg.min_explore_s)
        print(sanity)

        out_path = self.output_path(ctx.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        extra_lines = []
        fam_cov = warn_missing_subjects_coverage(fam["mouse_id"], subjects, where="NOR FAM")
        test_cov = warn_missing_subjects_coverage(test["mouse_id"], subjects, where="NOR TEST")
        for msg in (fam_cov, test_cov, sanity):
            if msg:
                extra_lines.append(msg)

        res = save_processed(
            processed,
            out_csv=out_path,
            assay=self.assay_name,
            files_n=len(fam_files) + len(test_files),
            extra_lines=extra_lines,
        )
        print(f"Saved: {res.out_csv}")
        print(f"Summary: {res.out_summary}")

class MISCRunner(BaseAssayRunner):
    assay_name = "MISC"
    default_output = "data/processed/MISC_processed.csv"
    default_pattern = "*.csv"

    def run(self, ctx: RunContext) -> None:
        subjects = self.load_subjects_df(ctx)

        raw = read_misc_csv(ctx.root)
        qc_misc(raw)

        long_df = merge_subject_metadata(raw, subjects, on_missing_subjects="warn_drop")
        long_df["source_file"] = str(ctx.root)

        misc_cfg = (ctx.cfg.get("misc", {}) or {})
        masks = misc_cfg.get("masks", [])
        derived_metrics = misc_cfg.get("derived_metrics", [])

        long_df = apply_misc_masks(
            long_df,
            masks=masks,
        )

        processed = derive_misc_metrics(
            long_df,
            derived_metrics=derived_metrics,
        )

        out_path = self.output_path(ctx.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        coverage_msg = warn_missing_subjects_coverage(
            processed["mouse_id"], subjects, where=self.assay_name
        )
        summary_msg = summarize_misc(processed)

        extra_lines = [m for m in [coverage_msg, summary_msg] if m]

        res = save_processed(
            processed,
            out_csv=out_path,
            assay=self.assay_name,
            files_n=1,
            extra_lines=extra_lines,
        )
        print(f"Saved: {res.out_csv}")
        print(f"Summary: {res.out_summary}")


RUNNER_REGISTRY: dict[str, BaseAssayRunner] = {
    "VIT": VITRunner(),
    "OFT": OFTRunner(),
    "SPL": SPLRunner(),
    "EPM": EPMRunner(),
    "NOR": NORRunner(),
    "LM": LMRunner(),
    "MISC": MISCRunner(),
}
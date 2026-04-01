from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from behaviour_stat_pipeline.io import (
    load_subjects,
    load_tracking_csv,
    merge_subject_metadata,
)
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
        df["source_file"] = str(f)
        return df
    
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
        dfs = [self.annotate_loaded_df(self.load_one_file(f, subjects, ctx.raw_id_col), f) for f in files]
        return pd.concat(dfs, ignore_index=True), len(files)

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
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

        res = save_processed(
            processed,
            out_csv=out_path,
            assay=self.assay_name,
            files_n=files_n,
            extra_lines=[m for m in self.extra_summary_lines(long_df, subjects) if m],
        )
        print(f"Saved: {res.out_csv}")
        print(f"Summary: {res.out_summary}")


class VITRunner(BaseAssayRunner):
    assay_name = "VIT"
    default_output = "data/processed/VIT_processed.csv"

    @staticmethod
    def infer_vit_timepoint_phase_from_filename(p: Path) -> tuple[str, str]:
        name = p.name.lower()

        if "first_min" in name:
            timepoint = "Before"
        elif "second_min" in name:
            timepoint = "After"
        else:
            raise ValueError(f"Cannot infer VIT timepoint from filename: {p.name}")

        if "hand" in name:
            phase = "hand"
        elif "standing" in name:
            phase = "standing"
        else:
            raise ValueError(f"Cannot infer VIT phase from filename: {p.name}")

        return timepoint, phase

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
        tp, ph = self.infer_vit_timepoint_phase_from_filename(source_file)
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

    def annotate_loaded_df(self, df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
        df["timepoint"] = self.timepoint_label
        return df


class OFTRunner(SingleSessionRunner):
    assay_name = "OFT"
    default_output = "data/processed/OFT_processed.csv"

    def run_qc(self, long_df: pd.DataFrame) -> None:
        qc_oft(long_df)

    def derive_processed(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return derive_oft_metrics(long_df, cfg=OFTConfig())


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
    
    def run(self, ctx: RunContext) -> None:
        subjects = self.load_subjects_df(ctx)

        fam_files = sorted(ctx.root.rglob("*FAM*_output_summary.csv"))
        test_files = sorted(ctx.root.rglob("*TEST*_output_summary.csv"))

        if not fam_files:
            raise FileNotFoundError(f"No FAM files found under {ctx.root} matching *FAM*_output_summary.csv")
        if not test_files:
            raise FileNotFoundError(f"No TEST files found under {ctx.root} matching *TEST*_output_summary.csv")

        fam = self._load_session(fam_files, "FAM", subjects, ctx.raw_id_col)
        test = self._load_session(test_files, "TEST", subjects, ctx.raw_id_col)

        processed = derive_nor_metrics(
            fam,
            test,
            schema_aliases=ctx.schema_aliases,
            cfg=NORConfig(),
        )
        sanity = summarize_nor_sanity(processed, min_total_s=20.0)
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
    "EPM": EPMRunner(),
    "NOR": NORRunner(),
    "LM": LMRunner(),
    "MISC": MISCRunner(),
}
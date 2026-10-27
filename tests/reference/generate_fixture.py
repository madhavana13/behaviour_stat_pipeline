"""
Generate the fixed dataset used for reference validation (tests/reference/fixture.csv).

Run once; the CSV is committed. Regenerating it requires regenerating
reference_values.json with generate_reference.R.

Design: unbalanced 2x2 (group x sex), so Type II and Type III ANOVA differ,
with a true interaction. No ties and no zero differences, so exact Wilcoxon
and Spearman p-values are well defined in both R and Python.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(20261026)
    cells = [("control", "female", 6), ("control", "male", 9), ("ifs", "female", 7), ("ifs", "male", 8)]
    rows = []
    i = 0
    for group, sex, n in cells:
        g = group == "ifs"
        s = sex == "male"
        for _ in range(n):
            i += 1
            x2 = rng.normal(10, 2)
            y = 5 + 1.2 * g + 0.6 * s + 1.5 * g * s + 0.4 * (x2 - 10) + rng.normal(0, 1)
            di = rng.normal(0.25 if g else 0.05, 0.3)
            rows.append({"mouse_id": f"M{i:02d}", "group": group, "sex": sex,
                         "y": round(y, 4), "x2": round(x2, 4), "di": round(di, 4)})
    df = pd.DataFrame(rows)

    # Latency with right-censoring (separate RNG stream so the columns above are unchanged).
    # Per-animal recording length ~300 s; events after it are censored (latency missing).
    rng_t = np.random.default_rng(7)
    df["rec_s"] = np.round(rng_t.uniform(295, 304, len(df)), 3)
    rate = 1 / np.where(df["group"] == "ifs", 260, 110) * np.where(df["sex"] == "male", 0.8, 1.0)
    t = np.round(rng_t.exponential(1 / rate), 3)
    df["latency"] = np.where(t <= df["rec_s"], t, np.nan)
    df.loc[df.index[-1], "rec_s"] = np.nan  # one animal without a recording -> excluded, not censored
    df.loc[df.index[-1], "latency"] = np.nan

    # Three-level grouping for k-level designs (separate RNG stream; other columns unchanged).
    # Balanced-ish within sex so every group3 x sex cell has several animals.
    rng_g = np.random.default_rng(3)
    df["group3"] = ""
    for sex in ["female", "male"]:
        idx = df.index[df["sex"] == sex]
        labels = np.resize(["a", "b", "c"], len(idx))
        df.loc[idx, "group3"] = rng_g.permutation(labels)

    for col in ["y", "x2", "di"]:
        assert df[col].is_unique, f"ties in {col}; change the seed"
    assert (df["di"] != 0).all()
    df.to_csv(HERE / "fixture.csv", index=False)

    make_repeated()


def make_repeated() -> None:
    """
    Repeated-measures fixture (tests/reference/fixture_repeated.csv) for mixed models.

    24 mice (group x sex, 6 each) x 3 post sessions, plus a per-mouse baseline.
    Mouse random intercept; residual SD larger in ifs (unequal variances); one
    missing observation so the design is unbalanced.
    """
    rng = np.random.default_rng(31)
    rows = []
    i = 0
    for group in ["control", "ifs"]:
        for sex in ["female", "male"]:
            for _ in range(6):
                i += 1
                g = group == "ifs"
                baseline = rng.normal(20, 3)
                u = rng.normal(0, 2.0)  # mouse random intercept
                for k, session in enumerate(["S1", "S2", "S3"]):
                    mu = 20 + 2.0 * g + 0.8 * (sex == "male") - 1.5 * k * g + 0.5 * (baseline - 20) + u
                    y = mu + rng.normal(0, 2.5 if g else 1.0)
                    rows.append({"mouse_id": f"R{i:02d}", "group": group, "sex": sex, "session": session,
                                 "baseline": round(baseline, 4), "y": round(y, 4)})
    df = pd.DataFrame(rows)
    df.loc[5, "y"] = np.nan
    df.to_csv(HERE / "fixture_repeated.csv", index=False)


if __name__ == "__main__":
    main()

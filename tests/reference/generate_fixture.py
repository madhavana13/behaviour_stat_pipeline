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

    for col in ["y", "x2", "di"]:
        assert df[col].is_unique, f"ties in {col}; change the seed"
    assert (df["di"] != 0).all()
    df.to_csv(HERE / "fixture.csv", index=False)


if __name__ == "__main__":
    main()

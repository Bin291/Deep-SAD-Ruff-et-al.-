"""Shared loaders for the Step-3 count tables, label sets, feature keys and
exact row sampling from counts.

A count row (user, day, act, hour, n, n_win, n_event) is split into "atoms" by
label: n_event event rows (all inside the window, verified in Step 2),
n_win - n_event window-only rows, n - n_win other rows. Rows inside an atom
are indistinguishable for the paper's features and for every label level, so
drawing row indices uniformly from [0, N) and mapping them to atoms through the
cumulative counts is exactly uniform row sampling without replacement.

Label sets (decided after STOP 2 / Step 4):
  main - all insiders of the release (r4.2: 70, r6.2: 5)
  V30a - r4.2 scenario-1 insiders only (30); other insiders count as normal
  V30b - r4.2 scenario-2 insiders only (30); other insiders count as normal
Feature sets:
  F38 - one-hot activity (7) + weekday (7) + hour (24), as in the paper
  F50 - F38 + one-hot month (12): OUR GUESS for the paper's unexplained
        input size 50; the paper never mentions month
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["logon", "device", "file", "email", "http"]
FILE_OF_ACT = {1: "logon", 2: "logon", 3: "device", 4: "device", 5: "email", 6: "file", 7: "http"}
FEATURE_KEYS = {"F38": ["act", "wd", "hour"], "F50": ["act", "wd", "hour", "month"]}
LABEL_SETS = {"4.2": ["main", "V30a", "V30b"], "6.2": ["main"]}


def label_users(rel, labelset="main"):
    ins = pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv")
    if labelset == "V30a":
        ins = ins[ins["scenario"] == 1]
    elif labelset == "V30b":
        ins = ins[ins["scenario"] == 2]
    elif labelset != "main":
        raise ValueError(labelset)
    return sorted(ins["user"])


def load_counts(rel):
    """All count rows of a release with weekday (0=Mon..6), month (1..12) and file."""
    lf = pl.concat([pl.scan_parquet(ROOT / "data" / "counts" / f"r{rel}" / f"{t}.parquet")
                    .with_columns(pl.lit(t).alias("file")) for t in TYPES])
    return lf.with_columns((pl.col("day").dt.weekday() - 1).cast(pl.Int8).alias("wd"),
                           pl.col("day").dt.month().cast(pl.Int8).alias("month")).collect()


def atoms(counts, users):
    """Split count rows into label-homogeneous atoms with weight k (rows)."""
    is_ins = pl.col("user").is_in(pl.Series(users).implode())
    keep = ["user", "day", "file", "act", "wd", "month", "hour"]
    parts = [
        counts.select(*keep, pl.col("n_event").alias("k"), pl.lit(1, pl.Int8).alias("y_win"), pl.lit(1, pl.Int8).alias("y_event")),
        counts.select(*keep, (pl.col("n_win") - pl.col("n_event")).alias("k"), pl.lit(1, pl.Int8).alias("y_win"), pl.lit(0, pl.Int8).alias("y_event")),
        counts.select(*keep, (pl.col("n") - pl.col("n_win")).alias("k"), pl.lit(0, pl.Int8).alias("y_win"), pl.lit(0, pl.Int8).alias("y_event")),
    ]
    a = pl.concat(parts).filter(pl.col("k") > 0).with_columns(is_ins.cast(pl.Int8).alias("y_user"))
    # window / event labels only exist for users of the chosen label set
    return a.with_columns((pl.col("y_win") * pl.col("y_user")).alias("y_win"),
                          (pl.col("y_event") * pl.col("y_user")).alias("y_event"))


def draw(atoms_df, size, rng):
    """Uniform sample of `size` rows without replacement; returns one row per draw."""
    k = atoms_df["k"].to_numpy()
    cum = np.cumsum(k)
    size = min(size, int(cum[-1]))
    idx = rng.choice(int(cum[-1]), size=size, replace=False)
    return atoms_df[np.searchsorted(cum, idx, side="right")].drop("k")


def p1_bal_sample(at, seed, per_file=5000, pos_share=1 / 3):
    """P1-bal: per file, round(per_file*pos_share) rows drawn from user-label
    positive rows and the rest from negative rows (both uniform)."""
    rng = np.random.default_rng(10_000 + seed)
    n_pos = round(per_file * pos_share)
    out = []
    for t in TYPES:
        f = at.filter(pl.col("file") == t)
        out.append(draw(f.filter(pl.col("y_user") == 1), n_pos, rng))
        out.append(draw(f.filter(pl.col("y_user") == 0), per_file - n_pos, rng))
    return pl.concat(out)


def p1_sample(rel, seed, users):
    """P1 main: the raw-file samples drawn in Step 3 (5000 uniform rows/file)."""
    lf = pl.concat([pl.scan_parquet(ROOT / "data" / "samples" / f"r{rel}" / f"{t}.parquet")
                    .filter(pl.col(f"seed{seed}")).with_columns(pl.lit(t).alias("file")) for t in TYPES])
    is_ins = pl.col("user").is_in(pl.Series(users).implode()).cast(pl.Int8)
    return (lf.with_columns((pl.col("ts").dt.weekday() - 1).cast(pl.Int8).alias("wd"),
                            pl.col("ts").dt.month().cast(pl.Int8).alias("month"),
                            (pl.col("ts").dt.hour() + 1).cast(pl.Int8).alias("hour"),
                            pl.col("ts").dt.date().alias("day"),
                            is_ins.alias("y_user"))
              .with_columns((pl.col("in_win").cast(pl.Int8) * pl.col("y_user")).alias("y_win"),
                            (pl.col("is_event").cast(pl.Int8) * pl.col("y_user")).alias("y_event"))
              .select("user", "day", "file", "act", "wd", "month", "hour", "y_user", "y_win", "y_event")
              .collect())


def to_cells(rows, key, weight=None):
    """Aggregate rows (weight 1, or column `weight`) to cells with n, n_user, n_win, n_event."""
    if weight:
        w = pl.col(weight).cast(pl.Int64)
        aggs = [w.sum().alias("n")]
    else:  # NB: pl.lit(1).sum() is 1, not the row count
        w = pl.lit(1, pl.Int64)
        aggs = [pl.len().cast(pl.Int64).alias("n")]
    aggs += [(w * pl.col(c).cast(pl.Int64)).sum().alias(f"n_{c[2:]}") for c in ["y_user", "y_win", "y_event"]]
    return rows.group_by(key).agg(aggs).sort(key).to_pandas()

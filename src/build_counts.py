"""Step 3: stream every log file once and aggregate to a count table, then draw
the P1 samples.

The paper's features depend only on (activity, weekday, hour), so a table
count(user, date, activity, hour) carries everything the models can see. We
also keep, per cell, how many rows are positive under the window and event
labels (cells can straddle a window boundary). The user-level label is a pure
function of `user`, so it is applied later and is not stored here.

Activity codes (paper, Table II): Logon=1 Logoff=2 Connect=3 Disconnect=4
email=5 file=6 http=7. hour = clock hour + 1 (paper: time 1..24).

P1 samples: for seeds 0..9, 5000 row indices per file drawn uniformly without
replacement from [0, N) (N = rows counted in the aggregation pass), then
fetched in one filtered pass. Uniform row indices avoid the length bias of
byte-offset sampling (long http rows would be over-drawn).

Usage: python src/build_counts.py 4.2 [types...]
Outputs: data/counts/r<rel>/<type>.parquet, data/samples/r<rel>/<type>.parquet,
         results/counts_summary_r<rel>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
# local copies, sha256 identical to HF cert_r4.2/ and kaggle/ (results/hash_log.jsonl)
SRC = {
    "4.2": Path(r"D:\Code\paper\data\cert_r4.2"),
    "6.2": Path(r"D:\Code\paper\data\kaggle_cert_insider_threat\extracted"),
}
TYPES = ["logon", "device", "file", "email", "http"]
ACT = {"logon": {"Logon": 1, "Logoff": 2}, "device": {"Connect": 3, "Disconnect": 4},
       "email": 5, "file": 6, "http": 7}
DATE_FMT = "%m/%d/%Y %H:%M:%S"
SAMPLE_SEEDS = range(10)
SAMPLE_N = 5000


def activity_expr(t):
    if isinstance(ACT[t], dict):
        # replace_strict raises on any value outside the mapping
        return pl.col("activity").replace_strict(ACT[t], return_dtype=pl.Int8)
    return pl.lit(ACT[t], dtype=pl.Int8)


def base_frame(path, t, windows, events):
    cols = ["id", "date", "user"] + (["activity"] if isinstance(ACT[t], dict) else [])
    lf = pl.scan_csv(path, infer_schema=False).select(cols)
    ts = pl.col("date").str.strptime(pl.Datetime("us"), DATE_FMT, strict=True)
    lf = lf.with_columns(ts.alias("ts"), activity_expr(t).alias("act"))
    lf = lf.join(windows.lazy(), on="user", how="left")
    lf = lf.join(events.with_columns(pl.lit(True).alias("is_event")).lazy(), on=["user", "id"], how="left")
    return lf.with_columns(
        (pl.col("ts").is_between(pl.col("win_start"), pl.col("win_end"))).fill_null(False).alias("in_win"),
        pl.col("is_event").fill_null(False),
    )


def count_lines(path, bufsize=64 << 20):
    n, last = 0, b""
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (last != b"\n")  # count an unterminated final line


def sample_indices(n_rows, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(SAMPLE_N, n_rows), replace=False))


def main(rel, types):
    ins = pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv", parse_dates=["win_start", "win_end"])
    windows = pl.from_pandas(ins[["user", "win_start", "win_end"]]).with_columns(
        pl.col("win_start").cast(pl.Datetime("us")), pl.col("win_end").cast(pl.Datetime("us")))
    ev = pd.read_csv(ROOT / "data" / "labels" / f"malicious_events_r{rel}.csv", dtype=str)
    ev = ev[ev["user"] == ev["insider"]]

    cdir, sdir = ROOT / "data" / "counts" / f"r{rel}", ROOT / "data" / "samples" / f"r{rel}"
    cdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / f"counts_summary_r{rel}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for t in types:
        path = SRC[rel] / f"{t}.csv"
        events = pl.from_pandas(ev.loc[ev["type"] == t, ["user", "id"]].drop_duplicates())
        t0 = time.time()
        lf = base_frame(path, t, windows, events)
        counts = (lf.group_by("user", pl.col("ts").dt.date().alias("day"), "act",
                              (pl.col("ts").dt.hour() + 1).cast(pl.Int8).alias("hour"))
                    .agg(pl.len().cast(pl.Int64).alias("n"),
                         pl.col("in_win").sum().cast(pl.Int64).alias("n_win"),
                         pl.col("is_event").sum().cast(pl.Int64).alias("n_event"))
                    .collect(engine="streaming"))
        counts.write_parquet(cdir / f"{t}.parquet")
        t_agg = time.time() - t0
        n_rows = int(counts["n"].sum())

        # P1 samples for all seeds in one filtered pass
        t1 = time.time()
        idx = {s: sample_indices(n_rows, s) for s in SAMPLE_SEEDS}
        union = np.unique(np.concatenate(list(idx.values())))
        sample = (base_frame(path, t, windows, events).with_row_index("rid")
                  .filter(pl.col("rid").is_in(pl.Series(union, dtype=pl.UInt32).implode()))
                  .select("rid", "id", "user", "ts", "act", "in_win", "is_event")
                  .collect(engine="streaming"))
        assert len(sample) == len(union), (t, len(sample), len(union))
        for s in SAMPLE_SEEDS:
            sample = sample.with_columns(pl.col("rid").is_in(pl.Series(idx[s], dtype=pl.UInt32).implode()).alias(f"seed{s}"))
        sample.write_parquet(sdir / f"{t}.parquet")
        t_smp = time.time() - t1

        t2 = time.time()
        lines = count_lines(path)
        t_lines = time.time() - t2
        summary[t] = {
            "rows_aggregated": n_rows, "raw_lines": lines, "rows_equal_lines_minus_header": n_rows == lines - 1,
            "cells": len(counts), "n_win": int(counts["n_win"].sum()), "n_event": int(counts["n_event"].sum()),
            "events_expected": len(events), "sample_rows_union": len(union),
            "seconds": {"aggregate": round(t_agg, 1), "sample": round(t_smp, 1), "line_count": round(t_lines, 1)},
        }
        print(t, json.dumps(summary[t]), flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    rel, *types = sys.argv[1:]
    main(rel, types or TYPES)

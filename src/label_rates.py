"""Positive rates of the three label levels over all five log files.

Levels (agreed at STOP 2):
  user   - every row of an insider user is positive (paper's definition)
  window - insider's own rows inside [win_start, win_end] (union of insiders.csv
           [start, end] and the detail-file event span)
  event  - row id is listed in the insider's detail file for that log type AND
           the row's user is the insider

Rates are reported per row and per user-day (a user-day is positive if it holds
at least one positive row). Also reported: window rows that are not detail
events, and detail events that cannot be found in the logs.

Usage: python src/label_rates.py 6.2 [logon device ...]
Writes results/label_rates_r<rel>.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    # local copies, sha256 identical to HF cert_r4.2/ and kaggle/ (results/hash_log.jsonl)
    "4.2": Path(r"D:\Code\paper\data\cert_r4.2"),
    "6.2":Path(r"D:\Code\paper\data\kaggle_cert_insider_threat\extracted"),
}
TYPES = ["logon", "device", "file", "email", "http"]
DATE_FMT = "%m/%d/%Y %H:%M:%S"


def scan(path):
    # all columns as strings; only id/date/user are materialised
    return pl.scan_csv(path, infer_schema=False).select("id", "date", "user")


def file_stats(path, ins, ev_ids):
    users = ins["user"].tolist()
    lf = scan(path)
    t0 = time.time()
    # (1) rows per user-day, (2) insider rows. date[:10] == 'MM/DD/YYYY'
    ud, rows = pl.collect_all([
        lf.group_by("user", pl.col("date").str.slice(0, 10).alias("day")).agg(pl.len().alias("n")),
        lf.filter(pl.col("user").is_in(users)),
    ], engine="streaming")
    ud, rows = ud.to_pandas(), rows.to_pandas()
    rows["ts"] = pd.to_datetime(rows["date"], format=DATE_FMT)
    rows["day"] = rows["date"].str.slice(0, 10)
    rows = rows.merge(ins[["user", "win_start", "win_end"]], on="user")
    rows["in_win"] = (rows["ts"] >= rows["win_start"]) & (rows["ts"] <= rows["win_end"])
    rows["is_event"] = [(u, i) in ev_ids for u, i in zip(rows["user"], rows["id"])]
    found = set(zip(rows["user"], rows["id"]))
    return {
        "seconds": round(time.time() - t0, 1),
        "ud": ud,
        "rows": rows[["user", "day", "in_win", "is_event"]],
        "n_rows": int(ud["n"].sum()),
        "missing_events": [k for k in ev_ids if k not in found],
    }


def main(rel, types):
    ins = pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv", parse_dates=["win_start", "win_end"])
    ev = pd.read_csv(ROOT / "data" / "labels" / f"malicious_events_r{rel}.csv", dtype=str)
    ev = ev[ev["user"] == ev["insider"]]  # event level: insider's own rows only

    per_file, all_ud, all_pos = {}, [], []
    for t in types:
        ids = set(zip(ev.loc[ev["type"] == t, "user"], ev.loc[ev["type"] == t, "id"]))
        s = file_stats(SRC[rel] / f"{t}.csv", ins, ids)
        r = s["rows"]
        n = s["n_rows"]
        per_file[t] = {
            "rows": n, "seconds": s["seconds"],
            "pos_user": int(len(r)), "pos_window": int(r["in_win"].sum()), "pos_event": int(r["is_event"].sum()),
            "window_rows_not_in_detail": int((r["in_win"] & ~r["is_event"]).sum()),
            "event_rows_outside_window": int((r["is_event"] & ~r["in_win"]).sum()),
            "detail_events_expected": len(ids), "detail_events_missing_in_log": len(s["missing_events"]),
        }
        print(t, per_file[t], flush=True)
        all_ud.append(s["ud"][["user", "day"]])
        all_pos.append(r)

    ud = pd.concat(all_ud).drop_duplicates()
    pos = pd.concat(all_pos)
    n_rows = sum(v["rows"] for v in per_file.values())
    tot = {k: sum(v[k] for v in per_file.values()) for k in
           ["pos_user", "pos_window", "pos_event", "window_rows_not_in_detail",
            "event_rows_outside_window", "detail_events_expected", "detail_events_missing_in_log"]}
    ud_user = pos[["user", "day"]].drop_duplicates()
    ud_win = pos.loc[pos["in_win"], ["user", "day"]].drop_duplicates()
    ud_ev = pos.loc[pos["is_event"], ["user", "day"]].drop_duplicates()
    out = {
        "release": rel, "files": types, "per_file": per_file,
        "rows_total": n_rows, "user_days_total": len(ud), "n_insiders": len(ins),
        **tot,
        "row_rate": {"user": tot["pos_user"] / n_rows, "window": tot["pos_window"] / n_rows,
                     "event": tot["pos_event"] / n_rows},
        "user_day_pos": {"user": len(ud_user), "window": len(ud_win), "event": len(ud_ev)},
        "user_day_rate": {"user": len(ud_user) / len(ud), "window": len(ud_win) / len(ud),
                          "event": len(ud_ev) / len(ud)},
        "window_user_days_per_insider": ud_win.groupby("user").size().to_dict(),
        "event_user_days_per_insider": ud_ev.groupby("user").size().to_dict(),
    }
    suffix = "" if types == TYPES else "_" + "-".join(types)
    (ROOT / "results" / f"label_rates_r{rel}{suffix}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["rows_total", "user_days_total", "row_rate", "user_day_pos", "user_day_rate"]}, indent=1))


if __name__ == "__main__":
    rel, *types = sys.argv[1:]
    main(rel, types or TYPES)

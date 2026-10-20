"""Parse CERT answers (insiders.csv + per-incident observable files).

For each insider of a release we keep: scenario, user, start/end from
insiders.csv, and min/max timestamp + event ids/types from the detail file.
The detail file is the ground truth for event-level labels; when a start/end
field in insiders.csv is malformed we fall back to the detail-file bound and
flag it (never guess).

Usage: python src/labels.py [4.2 6.2]   -> data/labels/insiders_r<rel>.csv,
                                           data/labels/malicious_events_r<rel>.csv
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANS = ROOT / "data" / "labels" / "answers"
OUT = ROOT / "data" / "labels"
DATE_FMT = "%m/%d/%Y %H:%M:%S"


def parse_date(s):
    """CERT dates are MM/DD/YYYY HH:MM:SS, month/day sometimes without zero padding."""
    try:
        return datetime.strptime(s.strip(), DATE_FMT)
    except (ValueError, AttributeError):
        return None


def detail_path(release, details):
    p = ANS / details
    if p.exists():
        return p
    # r4.2 is a "dense needle" release: files live in r4.2-<scenario>/ subfolders
    sub = "-".join(details.split("-")[:2])
    return ANS / sub / details


def read_detail(path):
    """Rows are variable length: type,id,date,user,pc,... (quoted in some releases)."""
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(f):
            if len(r) < 5:
                continue
            rows.append({"type": r[0].strip(), "id": r[1].strip(), "date": parse_date(r[2]),
                         "user": r[3].strip(), "pc": r[4].strip()})
    return rows


def load_release(release):
    ins = pd.read_csv(ANS / "insiders.csv", dtype=str)
    ins = ins[ins["dataset"] == release].reset_index(drop=True)
    recs, events = [], []
    for _, r in ins.iterrows():
        rows = read_detail(detail_path(release, r["details"]))
        dates = [x["date"] for x in rows if x["date"]]
        users = {x["user"] for x in rows}
        start, end = parse_date(r["start"]), parse_date(r["end"])
        dmin, dmax = (min(dates), max(dates)) if dates else (None, None)
        flags = []
        if start is None:
            flags.append(f"malformed start '{r['start']}' -> detail min")
            start = dmin
        if end is None:
            flags.append(f"malformed end '{r['end']}' -> detail max")
            end = dmax
        if dmin and (dmin < start or dmax > end):
            flags.append("detail events outside [start,end]")
        if users - {r["user"]}:
            flags.append(f"detail rows by other users: {sorted(users - {r['user']})}")
        counts = pd.Series([x["type"] for x in rows]).value_counts().to_dict()
        # Window label (agreed at STOP 2): union of insiders.csv [start, end] and
        # the detail-file event span; applied only to the insider's own user id.
        own = [x["date"] for x in rows if x["date"] and x["user"] == r["user"]]
        win_start = min([start] + own)
        win_end = max([end] + own)
        recs.append({"release": release, "scenario": int(r["scenario"]), "user": r["user"],
                     "start": start, "end": end, "detail_min": dmin, "detail_max": dmax,
                     "win_start": win_start, "win_end": win_end,
                     "n_events": len(rows), **{f"n_{k}": v for k, v in counts.items()},
                     "flags": "; ".join(flags)})
        for x in rows:
            events.append({"release": release, "scenario": int(r["scenario"]), "insider": r["user"], **x})
    return pd.DataFrame(recs).fillna({"flags": ""}), pd.DataFrame(events)


if __name__ == "__main__":
    for rel in sys.argv[1:] or ["4.2", "6.2"]:
        df, ev = load_release(rel)
        df.to_csv(OUT / f"insiders_r{rel}.csv", index=False)
        ev.to_csv(OUT / f"malicious_events_r{rel}.csv", index=False)
        print(f"r{rel}: {len(df)} insiders, {df['user'].nunique()} unique users, {len(ev)} malicious events")

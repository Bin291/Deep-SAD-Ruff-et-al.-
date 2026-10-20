"""STOP-2 sanity checks: insiders exist in the user roster, scenario windows lie
inside the log period, and the per-user/per-day positive rates implied by the
two label definitions (computed on logon.csv only at this stage).

Usage: python src/check_labels.py
"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\Code\paper\data\hf_full")
LOCAL_R62 = Path(r"D:\Code\paper\data\kaggle_cert_insider_threat\extracted")  # sha256 == HF (results/hash_log.jsonl)
SRC = {"4.2": DATA / "cert_r4.2", "6.2": LOCAL_R62}
DATE_FMT = "%m/%d/%Y %H:%M:%S"


def roster(rel):
    ldap = pd.concat([pd.read_csv(f, usecols=["user_id"]) for f in sorted(glob.glob(str(SRC[rel] / "LDAP" / "*.csv")))])
    users = set(ldap["user_id"])
    extra = {}
    if (SRC[rel] / "users.csv").exists():
        u = pd.read_csv(SRC[rel] / "users.csv")
        extra["users_csv_columns"] = list(u.columns)
        col = "user_id" if "user_id" in u.columns else u.columns[0]
        extra["users_csv_n"] = int(u[col].nunique())
        extra["users_csv_vs_ldap_same"] = set(u[col]) == users
    return users, extra


def main():
    report = {}
    for rel in ["4.2", "6.2"]:
        ins = pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv", parse_dates=["start", "end"])
        users, extra = roster(rel)
        logon = pd.read_csv(SRC[rel] / "logon.csv", usecols=["date", "user"])
        logon["date"] = pd.to_datetime(logon["date"], format=DATE_FMT)
        lo, hi = logon["date"].min(), logon["date"].max()
        logon["day"] = logon["date"].dt.normalize()
        ud = logon[["user", "day"]].drop_duplicates()
        n_rows, n_ud = len(logon), len(ud)

        # Level 1 (paper): every row of an insider is positive.
        pos_user_rows = logon["user"].isin(set(ins["user"])).sum()
        pos_user_ud = ud["user"].isin(set(ins["user"])).sum()
        # Level 2: only rows/days inside [start, end] (end day inclusive).
        win = ins[["user", "start", "end"]]
        m = logon.merge(win, on="user")
        pos_win_rows = ((m["date"] >= m["start"]) & (m["date"] <= m["end"])).sum()
        mu = ud.merge(win, on="user")
        pos_win_ud = ((mu["day"] >= mu["start"].dt.normalize()) & (mu["day"] <= mu["end"].dt.normalize())).sum()

        report[f"r{rel}"] = {
            "n_insiders": int(len(ins)),
            "by_scenario": ins.groupby("scenario").size().to_dict(),
            "insiders_missing_from_LDAP": sorted(set(ins["user"]) - users),
            "n_users_LDAP": len(users), **extra,
            "logon_period": [str(lo), str(hi)],
            "windows_outside_log_period": ins.loc[(ins["start"] < lo) | (ins["end"] > hi), "user"].tolist(),
            "insiders_without_logon_rows": sorted(set(ins["user"]) - set(logon["user"])),
            "logon_rows": n_rows, "logon_user_days": n_ud,
            "pos_rate_user_label_rows": round(pos_user_rows / n_rows, 5),
            "pos_rate_user_label_userdays": round(pos_user_ud / n_ud, 5),
            "pos_rate_window_label_rows": round(pos_win_rows / n_rows, 6),
            "pos_rate_window_label_userdays": round(pos_win_ud / n_ud, 6),
            "pos_window_userdays": int(pos_win_ud),
        }
    out = ROOT / "results" / "label_checks.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

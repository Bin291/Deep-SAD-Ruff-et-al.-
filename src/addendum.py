"""Report addendum (after Step 5 review). Definitions fixed before running.

(b) P3 user level: is the user-level signal (model AUC 0.74-0.91) just activity
    volume? Baselines on exactly the Step-5 P3 folds, thresholds T1/T2 as for
    the models:
      volume        user score = log(total rows of the user)
      rare_any      rare-cell indicator per row -> user-day max -> user max
      rare_share    rare-cell indicator per row -> user-day mean -> user max
      rare_count    user score = log1p(number of the user's rows in rare cells)
    rare cells = the rarest cells (by row count among FIT users) that together
    hold <= 1% of fit rows; recomputed per fold from fit users only.

(c) Bayes ceiling per activity file, P2 (full data) and P1 (Step-3 samples,
    in-sample), plus a decomposition of "P1 easier than P2":
      P2            full data, natural file shares (http ~87% of rows)
      P2-reweighted full data, each file re-weighted to 20% of rows (P1's
                    file mix) -- no sampling noise
      P1            5000 rows/file samples, in-sample (mean of 10 seeds)
    P2-reweighted vs P2 isolates the file-mix effect; P1 vs P2-reweighted
    isolates the small-sample (in-sample) effect.

Usage: python src/addendum.py -> results/addendum.json, fills REPORT.md markers
       <!--TABLE:userbase--> and <!--TABLE:byact-->
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import data5
from bayes_ceiling import best_f1, ranking
from counts import FEATURE_KEYS, TYPES, label_users, load_counts, p1_sample, to_cells
from make_tables import load_runs, md_table
from metrics import evaluate, threshold_t1, threshold_t2

ROOT = Path(__file__).resolve().parents[1]
RARE_SHARE = 0.01


def ms(v, pct=True, nd=1):
    v = np.asarray(v, float)
    k = 100 if pct else 1
    return f"{k*np.nanmean(v):.{nd}f} ± {k*np.nanstd(v):.{nd}f} [{k*np.nanmin(v):.{nd}f}, {k*np.nanmax(v):.{nd}f}]"


# ------------------------------------------------------------------ (b)
def rare_cells(fitN):
    order = np.argsort(fitN, kind="stable")
    cum = np.cumsum(fitN[order]) / fitN.sum()
    rare = np.zeros_like(fitN, dtype=bool)
    rare[order[cum <= RARE_SHARE]] = True
    return rare


def eval_user(score, d, sp):
    lab = d.is_ins.astype(float)
    m = {"train": np.r_[sp["fit"], sp["val"]], "val": sp["val"], "test": sp["test"]}
    thr = {"T1": threshold_t1(score[m["train"]], lab[m["train"]], 1 - lab[m["train"]]),
           "T2": threshold_t2(score[m["val"]], lab[m["val"]], 1 - lab[m["val"]])}
    return evaluate(score[m["test"]], lab[m["test"]], 1 - lab[m["test"]], thr)


def user_baselines(rel, ls, counts):
    d = data5.p3_data(rel, ls, "F38", counts)
    splits = data5.p3_folds(d, rel)
    tot = d.N.sum(1)
    out = []
    for sp in splits:
        fitN = d.N[sp["fit"]].sum(0)
        rare = rare_cells(fitN)
        agg = data5.p3_aggregate(d, rare.astype(float))
        scores = {"volume": np.log(np.maximum(tot, 1)),
                  "rare_any": agg["max"]["user"], "rare_share": agg["mean"]["user"],
                  "rare_count": np.log1p(d.N[:, rare].sum(1))}
        out.append({k: eval_user(s, d, sp) for k, s in scores.items()}
                   | {"rare_cells": int(rare.sum()), "rare_rows_share_fit": float(fitN[rare].sum() / fitN.sum())})
    return out


def userbase_table(res, R, P3C):
    rows = []
    for key, per in res.items():
        rel, ls = key.split("_")[0][1:], key.split("_")[1]

        def add(name, evs):
            rows.append({"Cấu hình": f"r{rel} {ls}", "Bộ chấm điểm user": name,
                         "AUC": ms([e["roc_auc"] for e in evs], pct=False, nd=3),
                         "F1 T1": ms([e["T1"]["f1"] for e in evs]), "F1 T2": ms([e["T2"]["f1"] for e in evs])})
        for model in ["AE", "VAE"]:
            for reading in ["R1", "R2"]:
                n = f"r{rel}_{ls}_F38_P3_{model}_{reading}"
                if n in R:
                    add(f"{model} {reading} (max)", [r["user_max"] for r in R[n]["runs"]])
        pc = P3C[f"r{rel}_{ls}_F38"]
        add("*Tần suất nghịch của ô (max)*", [x["inv_freq"]["user_max"] for x in pc])
        add("*Bảng tra (max)*", [x["lookup"]["user_max"] for x in pc])
        for b, lab in [("volume", "*log(tổng số dòng)*"), ("rare_count", "*log(1 + số dòng ở ô hiếm)*"),
                       ("rare_share", "*tỉ lệ dòng ô hiếm (mean → max)*"), ("rare_any", "*có dòng ô hiếm (max → max)*")]:
            add(lab, [x[b] for x in per])
    return md_table(pd.DataFrame(rows))


# ------------------------------------------------------------------ (c)
def ceiling(cells):
    pos = cells["n_user"].to_numpy(float)
    neg = cells["n"].to_numpy(float) - pos
    rate = pos / np.maximum(pos + neg, 1)
    b = best_f1(pos, neg, rate) | ranking(pos, neg, rate)
    return {"rows": float(pos.sum() + neg.sum()), "prevalence": pos.sum() / (pos.sum() + neg.sum()),
            "f1": b["f1"], "precision": b["precision"], "recall": b["recall"], "auc": b["roc_auc"]}


def by_activity(rel, counts):
    users = label_users(rel, "main")
    key = FEATURE_KEYS["F38"]
    full = counts.with_columns(pl.when(pl.col("user").is_in(pl.Series(users).implode())).then(pl.col("n")).otherwise(0).alias("n_user"))
    full_cells = full.group_by(key + ["file"]).agg(pl.col("n").sum(), pl.col("n_user").sum()).to_pandas()
    res = {"P2": {}, "P1": {}}
    for f in TYPES:
        res["P2"][f] = ceiling(full_cells[full_cells["file"] == f])
    res["P2"]["all"] = ceiling(full_cells.groupby(key)[["n", "n_user"]].sum().reset_index())
    share = full_cells.groupby("file")["n"].sum() / full_cells["n"].sum()
    rw = full_cells.copy()
    w = (0.2 / share).reindex(rw["file"]).to_numpy()
    rw["n"], rw["n_user"] = rw["n"] * w, rw["n_user"] * w
    res["P2-reweighted"] = {"all": ceiling(rw.groupby(key)[["n", "n_user"]].sum().reset_index())}
    res["file_share_P2"] = share.to_dict()
    per_seed = []
    for s in range(10):
        rows = p1_sample(rel, s, users)
        r = {f: ceiling(to_cells(rows.filter(pl.col("file") == f), key)) for f in TYPES}
        r["all"] = ceiling(to_cells(rows, key))
        per_seed.append(r)
    for f in TYPES + ["all"]:
        res["P1"][f] = {k: float(np.nanmean([p[f][k] for p in per_seed])) for k in per_seed[0][f]}
    return res


def byact_table(res):
    rows = []
    for rel, r in res.items():
        for f in TYPES + ["all"]:
            p2, p1 = r["P2"][f], r["P1"][f]
            rows.append({"Bản": rel, "File": f if f != "all" else "**tất cả**",
                         "Tỉ trọng dòng P2": f"{100*r['file_share_P2'][f]:.1f}%" if f != "all" else "100%",
                         "Tỉ trọng dòng P1": "20%" if f != "all" else "100%",
                         "Dương P2": f"{100*p2['prevalence']:.2f}%", "Dương P1": f"{100*p1['prevalence']:.2f}%",
                         "Trần F1 P2": f"{100*p2['f1']:.1f}", "Trần F1 P1 (in-sample)": f"{100*p1['f1']:.1f}",
                         "AUC P2": f"{p2['auc']:.3f}", "AUC P1 (in-sample)": f"{p1['auc']:.3f}"})
        rw = r["P2-reweighted"]["all"]
        rows.append({"Bản": rel, "File": "**tất cả, P2 cân lại 20%/file**", "Tỉ trọng dòng P2": "20% mỗi file",
                     "Tỉ trọng dòng P1": "—", "Dương P2": f"{100*rw['prevalence']:.2f}%", "Dương P1": "—",
                     "Trần F1 P2": f"{100*rw['f1']:.1f}", "Trần F1 P1 (in-sample)": "—",
                     "AUC P2": f"{rw['auc']:.3f}", "AUC P1 (in-sample)": "—"})
    return md_table(pd.DataFrame(rows))


def fill(markers):
    rep = ROOT / "REPORT.md"
    text = rep.read_text(encoding="utf-8")
    for k, t in markers.items():
        text, n = re.subn(rf"<!--TABLE:{k}-->.*?<!--/TABLE-->", f"<!--TABLE:{k}-->\n{t}\n<!--/TABLE-->", text, flags=re.S)
        assert n == 1, f"marker {k} not found"
    rep.write_text(text, encoding="utf-8")


def main():
    R = load_runs()
    P3C = json.loads((ROOT / "results" / "p3_ceiling.json").read_text())
    ub, ba = {}, {}
    for rel in ["4.2", "6.2"]:
        counts = load_counts(rel)
        for ls in (["main", "V30a", "V30b"] if rel == "4.2" else ["main"]):
            ub[f"r{rel}_{ls}"] = user_baselines(rel, ls, counts)
            print(f"(b) r{rel} {ls} done", flush=True)
        ba[f"r{rel}"] = by_activity(rel, counts)
        print(f"(c) r{rel} done", flush=True)
    (ROOT / "results" / "addendum.json").write_text(json.dumps({"user_baselines": ub, "by_activity": ba}, default=float))
    t_ub, t_ba = userbase_table(ub, R, P3C), byact_table(ba)
    (ROOT / "results" / "addendum_tables.md").write_text("## (b)\n\n" + t_ub + "\n\n## (c)\n\n" + t_ba + "\n", encoding="utf-8")
    print(t_ub)
    print(t_ba)
    try:
        fill({"userbase": t_ub, "byact": t_ba})
        print("REPORT.md filled")
    except AssertionError as e:
        print("REPORT markers missing:", e)


if __name__ == "__main__":
    main()

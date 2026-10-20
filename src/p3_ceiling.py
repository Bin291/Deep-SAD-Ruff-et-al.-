"""Step 4 addendum: ceiling and baselines for P3 (user split, user-day / user
evaluation), on exactly the folds and aggregation rules Step 5 uses.

Scorers (cell -> row score), each aggregated to user-day by max / mean and to
user by max over user-days (docs/step5_preregistration.md §5):
  oracle    true window-positive rate of each cell on the TEST users themselves,
            cut-off = best F1 on the test units (in-sample). Approximate ceiling
            for any "score rows, then aggregate" model; not a strict bound.
  lookup    positive rate of each cell on the FIT users; thresholds T1 (train)
            and T2 (validation users), as for the models
  inv_freq  -log frequency of the cell among FIT users' rows; T1 / T2
plus flag-nobody / flag-all on the test units (both metric families).

Usage: python src/p3_ceiling.py -> results/p3_ceiling.json, results/p3_ceiling_table.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data5
from metrics import evaluate, flag_all, flag_nobody, threshold_t1, threshold_t2

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("4.2", "main", "F38"), ("4.2", "V30a", "F38"), ("4.2", "V30b", "F38"),
           ("4.2", "main", "F50"), ("6.2", "main", "F38"), ("6.2", "main", "F50")]


def units(d, split, unit):
    owner = d.ud_user if unit == "ud" else np.arange(len(d.users))
    lab = (d.ud_label if unit == "ud" else d.is_ins).astype(float)
    m = {k: np.isin(owner, np.r_[split["fit"], split["val"]] if k == "train" else split[k]) for k in ["train", "val", "test"]}
    return m, lab


def scorer_eval(d, split, cell_score, oracle=False):
    agg = data5.p3_aggregate(d, cell_score)
    out = {}
    for unit in ["ud", "user"]:
        m, lab = units(d, split, unit)
        pos = {k: lab[v] for k, v in m.items()}
        neg = {k: 1 - pos[k] for k in m}
        for rule in ["max", "mean"]:
            sc = {k: agg[rule][unit][v] for k, v in m.items()}
            if oracle:
                thr = {"best": threshold_t2(sc["test"], pos["test"], neg["test"])}
            else:
                thr = {"T1": threshold_t1(sc["train"], pos["train"], neg["train"]),
                       "T2": threshold_t2(sc["val"], pos["val"], neg["val"])}
            out[f"{unit}_{rule}"] = evaluate(sc["test"], pos["test"], neg["test"], thr)
        out[f"{unit}_flag_nobody"] = flag_nobody(pos["test"], neg["test"])
        out[f"{unit}_flag_all"] = flag_all(pos["test"], neg["test"])
    return out


def main():
    res, rows = {}, []
    counts = {}
    for rel, ls, feat in CONFIGS:
        if rel not in counts:
            counts = {rel: data5.load_counts(rel)}
        d = data5.p3_data(rel, ls, feat, counts[rel])
        splits = data5.p3_folds(d, rel)
        per = []
        for sp in splits:
            fitN, fitW = d.N[sp["fit"]].sum(0), d.NW[sp["fit"]].sum(0)
            teN, teW = d.N[sp["test"]].sum(0), d.NW[sp["test"]].sum(0)
            prev_fit = fitW.sum() / fitN.sum()
            lookup = np.where(fitN > 0, fitW / np.maximum(fitN, 1), prev_fit)
            oracle = np.where(teN > 0, teW / np.maximum(teN, 1), 0.0)
            inv = -np.log(np.maximum(fitN, 0.5) / fitN.sum())
            r = {"info": data5.p3_run_weights(d, sp).info,
                 "oracle": scorer_eval(d, sp, oracle, oracle=True),
                 "lookup": scorer_eval(d, sp, lookup), "inv_freq": scorer_eval(d, sp, inv)}
            per.append(r)
        key = f"r{rel}_{ls}_{feat}"
        res[key] = per
        for unit in ["ud", "user"]:
            for rule in ["max", "mean"]:
                ev = f"{unit}_{rule}"
                def col(f):
                    v = np.array([f(p) for p in per], float)
                    return f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f} [{np.nanmin(v):.3f},{np.nanmax(v):.3f}]"
                rows.append({
                    "config": key, "unit": unit, "rule": rule,
                    "test_pos_units": col(lambda p: p["oracle"][ev]["positives"]),
                    "test_units": col(lambda p: p["oracle"][ev]["units"]),
                    "ceiling_f1": col(lambda p: p["oracle"][ev]["best"]["f1"]),
                    "ceiling_auc": col(lambda p: p["oracle"][ev]["roc_auc"]),
                    "lookup_T2_f1": col(lambda p: p["lookup"][ev]["T2"]["f1"]),
                    "lookup_T1_f1": col(lambda p: p["lookup"][ev]["T1"]["f1"]),
                    "lookup_auc": col(lambda p: p["lookup"][ev]["roc_auc"]),
                    "invfreq_auc": col(lambda p: p["inv_freq"][ev]["roc_auc"]),
                    "flag_all_f1": col(lambda p: p["oracle"][f"{unit}_flag_all"]["f1"]),
                    "flag_nobody_acc": col(lambda p: p["oracle"][f"{unit}_flag_nobody"]["accuracy"]),
                    "flag_nobody_wF1": col(lambda p: p["oracle"][f"{unit}_flag_nobody"]["w_f1"]),
                })
        print(key, "done", flush=True)
    (ROOT / "results" / "p3_ceiling.json").write_text(json.dumps(res, default=float))
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "results" / "p3_ceiling_table.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()

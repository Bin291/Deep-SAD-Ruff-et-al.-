"""Baselines and ceiling on exactly the Step-5 test sets (P1, P1-bal, P2).
P3 baselines come from src/p3_ceiling.py (same folds).

Per run: oracle ceiling (test cell rates, in-sample best F1 + AUC), lookup
(fit-set cell rates; T1 on train, T2 on validation), inverse cell frequency
(fit set; T1, T2), flag-nobody and flag-all, both metric families.

Usage: python src/baselines5.py -> results/baselines5.json
"""
import json
from pathlib import Path

import numpy as np

import data5
from metrics import evaluate, flag_all, flag_nobody, threshold_t1, threshold_t2

ROOT = Path(__file__).resolve().parents[1]


def configs():
    out = [("4.2", ls, "F38", p) for ls in ["main", "V30a", "V30b"] for p in ["P1", "P1-bal", "P2"]]
    out += [("6.2", "main", "F38", p) for p in ["P1", "P1-bal", "P2"]]
    out += [(r, "main", "F50", p) for r in ["4.2", "6.2"] for p in ["P1", "P2"]]
    return out


def run_baselines(r):
    tr_pos, tr_neg = r.fit_pos + r.val_pos, r.fit_neg + r.val_neg
    fit_n = r.fit_pos + r.fit_neg
    te_n = r.test_pos + r.test_neg
    prev = r.fit_pos.sum() / fit_n.sum()
    scorers = {
        "lookup": np.where(fit_n > 0, r.fit_pos / np.maximum(fit_n, 1), prev),
        "inv_freq": -np.log(np.maximum(fit_n, 0.5) / fit_n.sum()),
    }
    out = {"info": r.info, "flag_nobody": flag_nobody(r.test_pos, r.test_neg), "flag_all": flag_all(r.test_pos, r.test_neg)}
    oracle = np.where(te_n > 0, r.test_pos / np.maximum(te_n, 1), 0.0)
    out["oracle"] = evaluate(oracle, r.test_pos, r.test_neg, {"best": threshold_t2(oracle, r.test_pos, r.test_neg)})
    for name, s in scorers.items():
        out[name] = evaluate(s, r.test_pos, r.test_neg,
                             {"T1": threshold_t1(s, tr_pos, tr_neg), "T2": threshold_t2(s, r.val_pos, r.val_neg)})
    return out


def main():
    res, counts = {}, {}
    for rel, ls, feat, proto in configs():
        if proto == "P1":
            runs = data5.p1_runs(rel, ls, feat)
        else:
            if rel not in counts:
                counts = {rel: data5.load_counts(rel)}
            runs = (data5.p1_bal_runs if proto == "P1-bal" else data5.p2_runs)(rel, ls, feat, counts[rel])
        key = f"r{rel}_{ls}_{feat}_{proto}"
        res[key] = [run_baselines(r) for r in runs]
        f1 = [x["oracle"]["best"]["f1"] for x in res[key]]
        lk = [x["lookup"]["T2"]["f1"] for x in res[key]]
        print(f"{key:28s} oracle F1 {np.mean(f1):.3f}  lookup T2 F1 {np.mean(lk):.3f}", flush=True)
    (ROOT / "results" / "baselines5.json").write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()

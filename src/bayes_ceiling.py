"""Step 4: ceiling of any model that only sees the paper's features, plus
baselines, for every configuration that Step 5 will train.

Every row reduces to a cell (activity, weekday, hour[, month]). Any model on
these features -- R1 (reconstruction error) or R2 (encoder + supervised
classifier) alike -- outputs the same score for all rows of a cell, so the
metrics below bound both readings.

  ceiling       oracle, in-sample: cells ranked by their true positive rate in
                the evaluated data itself; best-F1 cut-off (P, R, Acc there),
                ROC-AUC, PR-AUC, and max accuracy (majority label per cell)
  all_normal    predict everything normal (positive-class metrics), and the
                same predictor scored as sklearn "weighted avg" (diagnostic for
                how the paper may have averaged; not our reporting metric)
  random        AUC 0.5, PR-AUC = prevalence
  inv_freq      score = -log(cell frequency); AUC, PR-AUC, oracle best-F1
  lookup        honest supervised plug-in: cell positive rates learnt on a
                75% row split, cut-off chosen by max F1 on that train split,
                evaluated on the other 25%

Configurations: release x label set (main / V30a / V30b) x feature set
(F38 / F50) x data (P2 full; P1 = Step-3 raw samples, seeds 0..9; P1-bal =
1/3 positive samples drawn from the counts, seeds 0..9).

Usage: python src/bayes_ceiling.py  -> results/bayes_ceiling.json, results/bayes_ceiling_table.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from counts import FEATURE_KEYS, LABEL_SETS, atoms, label_users, load_counts, p1_bal_sample, p1_sample, to_cells

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {"user": "n_user", "window": "n_win", "event": "n_event"}
SEEDS = range(10)
TEST_SHARE = 0.25


def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "accuracy": (tp + tn) / (tp + fp + fn + tn), "f1": f1}


def sweep(pos, neg, score):
    """Cumulative TP/FP over cut-offs of cells sorted by score (ties kept together)."""
    df = pd.DataFrame({"pos": pos, "neg": neg, "s": score}).groupby("s", sort=True)[["pos", "neg"]].sum()[::-1]
    return df["pos"].cumsum().to_numpy(), df["neg"].cumsum().to_numpy(), df.index.to_numpy(), pos.sum(), neg.sum()


def best_f1(pos, neg, score):
    tp, fp, thr, P, N = sweep(pos, neg, score)
    f1 = np.where(tp > 0, 2 * tp / (tp + fp + P), 0.0)
    k = int(np.argmax(f1))
    return prf(tp[k], fp[k], P - tp[k], N - fp[k]) | {"flagged_share": float((tp[k] + fp[k]) / (P + N)),
                                                     "threshold": float(thr[k])}


def ranking(pos, neg, score):
    if pos.sum() == 0 or neg.sum() == 0:  # single class: ranking metrics undefined
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    w = np.r_[pos, neg].astype(float)
    s = np.r_[score, score]
    keep = w > 0
    return {"roc_auc": float(roc_auc_score(y[keep], s[keep], sample_weight=w[keep])),
            "pr_auc": float(average_precision_score(y[keep], s[keep], sample_weight=w[keep]))}


def weighted_avg_all_normal(prev):
    q = 1 - prev  # normal share; insider class gets P = R = F1 = 0
    return {"precision": q * q, "recall": q, "accuracy": q, "f1": q * 2 * q / (1 + q)}


def analyse(cells):
    """Ceiling + unsupervised baselines on one evaluated set (cells with n, n_<level>)."""
    out = {"rows": int(cells["n"].sum()), "cells_nonempty": int((cells["n"] > 0).sum())}
    for lvl, col in LEVELS.items():
        pos = cells[col].to_numpy(float)
        neg = cells["n"].to_numpy(float) - pos
        P, N = pos.sum(), neg.sum()
        rate = pos / np.maximum(pos + neg, 1)
        inv_freq = -np.log((pos + neg) / (P + N))
        maj = pos > neg
        out[lvl] = {
            "positives": int(P), "prevalence": P / (P + N),
            "ceiling": best_f1(pos, neg, rate) | ranking(pos, neg, rate),
            "ceiling_max_acc": prf(pos[maj].sum(), neg[maj].sum(), pos[~maj].sum(), neg[~maj].sum())["accuracy"],
            "all_normal": prf(0, 0, P, N),
            "all_positive": prf(P, N, 0, 0),  # flag everything: F1 = 2p/(1+p)
            "all_normal_weighted_avg": weighted_avg_all_normal(P / (P + N)),
            "random": {"roc_auc": 0.5, "pr_auc": P / (P + N)},
            "inv_freq": ranking(pos, neg, inv_freq) | {"best_f1_oracle": best_f1(pos, neg, inv_freq)},
        }
    return out


def lookup(train, test, key):
    """Honest supervised plug-in: train cell rates -> threshold by train max-F1 -> test."""
    m = test.merge(train, on=key, how="left", suffixes=("", "_tr"))
    out = {}
    for lvl, col in LEVELS.items():
        prev_tr = train[col].sum() / train["n"].sum()
        tr_rate = train[col] / train["n"]
        thr = best_f1(train[col].to_numpy(float), (train["n"] - train[col]).to_numpy(float), tr_rate.to_numpy())["threshold"] \
            if train[col].sum() > 0 else np.inf
        score = (m[f"{col}_tr"] / m["n_tr"]).fillna(prev_tr).to_numpy()  # unseen cell -> train prevalence
        pos = m[col].to_numpy(float)
        neg = m["n"].to_numpy(float) - pos
        flag = score >= thr
        out[lvl] = prf(pos[flag].sum(), neg[flag].sum(), pos[~flag].sum(), neg[~flag].sum()) \
            | ranking(pos, neg, score) | {"test_positives": int(pos.sum())}
    return out


def split_rows(rows, seed):
    rng = np.random.default_rng(20_000 + seed)
    is_test = rng.random(len(rows)) < TEST_SHARE
    return rows.filter(pl.Series(~is_test)), rows.filter(pl.Series(is_test))


def split_atoms(at, seed):
    """Exact 75/25 row split of weighted atoms (binomial thinning per atom)."""
    rng = np.random.default_rng(30_000 + seed)
    k = at["k"].to_numpy()
    k_te = rng.binomial(k, TEST_SHARE)
    return at.with_columns(pl.Series("k", k - k_te)), at.with_columns(pl.Series("k", k_te))


def summarise(per_seed):
    """mean, std, min, max over seeds for the headline numbers."""
    out = {}
    for lvl in LEVELS:
        rows = {"positives": [d["eval"][lvl]["positives"] for d in per_seed]}
        for m in ["precision", "recall", "accuracy", "f1", "roc_auc", "pr_auc"]:
            rows[f"ceiling_{m}"] = [d["eval"][lvl]["ceiling"][m] for d in per_seed]
            rows[f"lookup_{m}"] = [d["lookup"][lvl][m] for d in per_seed]
        rows["lookup_test_positives"] = [d["lookup"][lvl]["test_positives"] for d in per_seed]
        out[lvl] = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                        "min": float(np.nanmin(v)), "max": float(np.nanmax(v))} for k, v in rows.items()}
        out[lvl]["seeds_without_positives"] = int(sum(p == 0 for p in rows["positives"]))
    return out


def run_config(rel, counts, labelset, feat):
    users = label_users(rel, labelset)
    key = FEATURE_KEYS[feat]
    at = atoms(counts, users)
    res = {"n_insiders": len(users)}

    # P2: full data (in-sample ceiling) + honest lookup on an exact 75/25 row split
    tr, te = split_atoms(at, 0)
    res["P2"] = {"eval": analyse(to_cells(at, key, "k")),
                 "lookup": lookup(to_cells(tr, key, "k"), to_cells(te, key, "k"), key)}

    for proto in ["P1", "P1-bal"]:
        per_seed = []
        for s in SEEDS:
            rows = p1_sample(rel, s, users) if proto == "P1" else p1_bal_sample(at, s)
            tr_r, te_r = split_rows(rows, s)
            per_seed.append({"eval": analyse(to_cells(rows, key)),
                             "lookup": lookup(to_cells(tr_r, key), to_cells(te_r, key), key)})
        res[proto] = {"per_seed": per_seed, "summary": summarise(per_seed)}
    return res


def flat_table(res):
    """One CSV row per (release, labelset, features, protocol, level)."""
    rows = []
    for rel, by_ls in res.items():
        for ls, by_feat in by_ls.items():
            for feat, r in by_feat.items():
                for proto in ["P2", "P1", "P1-bal"]:
                    for lvl in LEVELS:
                        base = {"release": rel, "labelset": ls, "features": feat, "protocol": proto, "level": lvl}
                        if proto == "P2":
                            e, lk = r["P2"]["eval"][lvl], r["P2"]["lookup"][lvl]
                            rows.append(base | {"prevalence": e["prevalence"], "positives": e["positives"]}
                                        | {f"ceil_{m}": e["ceiling"][m] for m in ["precision", "recall", "accuracy", "f1", "roc_auc", "pr_auc"]}
                                        | {"ceil_max_acc": e["ceiling_max_acc"], "invfreq_auc": e["inv_freq"]["roc_auc"],
                                           "all_positive_f1": e["all_positive"]["f1"]}
                                        | {f"lookup_{m}": lk[m] for m in ["precision", "recall", "accuracy", "f1", "roc_auc"]})
                        else:
                            s = r[proto]["summary"][lvl]
                            ev = [d["eval"][lvl] for d in r[proto]["per_seed"]]
                            rows.append(base | {"prevalence": float(np.mean([e["prevalence"] for e in ev])),
                                                "positives": s["positives"]["mean"],
                                                "seeds_without_positives": s["seeds_without_positives"]}
                                        | {f"ceil_{m}": s[f"ceiling_{m}"]["mean"] for m in ["precision", "recall", "accuracy", "f1", "roc_auc", "pr_auc"]}
                                        | {"ceil_f1_std": s["ceiling_f1"]["std"],
                                           "ceil_max_acc": float(np.mean([e["ceiling_max_acc"] for e in ev])),
                                           "invfreq_auc": float(np.nanmean([e["inv_freq"]["roc_auc"] for e in ev])),
                                           "all_positive_f1": float(np.mean([e["all_positive"]["f1"] for e in ev]))}
                                        | {f"lookup_{m}": s[f"lookup_{m}"]["mean"] for m in ["precision", "recall", "accuracy", "f1", "roc_auc"]}
                                        | {"lookup_f1_std": s["lookup_f1"]["std"], "lookup_f1_min": s["lookup_f1"]["min"],
                                           "lookup_f1_max": s["lookup_f1"]["max"]})
    return pd.DataFrame(rows)


def main():
    res = {}
    for rel in ["4.2", "6.2"]:
        counts = load_counts(rel)
        res[f"r{rel}"] = {}
        for ls in LABEL_SETS[rel]:
            res[f"r{rel}"][ls] = {}
            for feat in FEATURE_KEYS:
                print(f"r{rel} {ls} {feat} ...", flush=True)
                res[f"r{rel}"][ls][feat] = run_config(rel, counts, ls, feat)
    (ROOT / "results" / "bayes_ceiling.json").write_text(json.dumps(res, indent=1, default=float))
    tab = flat_table(res)
    tab.to_csv(ROOT / "results" / "bayes_ceiling_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["release", "labelset", "features", "protocol", "level", "prevalence", "positives",
            "ceil_precision", "ceil_recall", "ceil_accuracy", "ceil_f1", "ceil_roc_auc", "ceil_max_acc",
            "lookup_precision", "lookup_recall", "lookup_accuracy", "lookup_f1", "lookup_roc_auc", "invfreq_auc"]
    print(tab[tab["level"] == "user"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

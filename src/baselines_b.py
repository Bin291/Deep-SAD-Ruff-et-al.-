"""B1 baselines on user-day features (docs/phaseB_spec.md §7 + B0 v2).

Baselines (unsupervised unless noted; every score: higher = more suspicious)
  random            analytic (AUC 0.5, PR-AUC = prevalence, recall@b = b)
  volume+/volume-   +/- log(rows of the user-day); sign chosen by AUC on fit (V2.7)
  rare              share of the user-day's rows in rare F38 cells (cells from fit rows)
  rule              log1p(off-hours connect) + log1p(off-hours file) + first-USB flag (V2.7,
                    designed knowing the scenario shapes)
  zmax / zpos       max |z| and sum of positive z over the 14 D z-scores
  IF_GD (+IF_GDR)   IsolationForest(200 trees, max_samples 256) on G+D (+R, r6.2)
References
  HGB_GD (+HGB_GDR) HistGradientBoostingClassifier, defaults, train labels (supervised)
  IF_GDI, HGB_GDI   + group I keywords ("có thông tin kịch bản")

Usage: python src/baselines_b.py 4.2 PBU|PBT   -> results/phaseB/b1_r<rel>_<proto>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

import protocols_b as pb
import userday
from addendum import rare_cells
from counts import load_counts
from data5 import cell_expr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phaseB"
SCEN = {"4.2": [1, 2, 3], "6.2": [1, 2, 3, 4, 5]}
PBT_SEEDS = range(5)


def load(rel):
    f = pl.read_parquet(ROOT / "data" / "userday" / f"r{rel}.parquet")
    lab = pl.read_parquet(ROOT / "data" / "userday" / f"labels_r{rel}.parquet")
    assert f.select("user", "day").equals(lab.select("user", "day")), "feature / label keys differ"
    assert (f["n_rows"].cast(pl.Int64) == lab["n_rows_counts"].cast(pl.Int64)).all()
    return f, lab


class Rare:
    """Rare-cell share per user-day; rare cells recomputed from fit rows of each split."""

    def __init__(self, rel, lab):
        c = load_counts(rel).select("user", "day", cell_expr("F38").alias("cid"), "n")
        c = c.join(lab.select("user", "day").with_row_index("ud"), on=["user", "day"], how="left")
        assert c["ud"].null_count() == 0
        self.ud, self.cid, self.n = c["ud"].to_numpy(), c["cid"].to_numpy(), c["n"].to_numpy().astype(np.float64)
        self.U = len(lab)
        self.tot = np.bincount(self.ud, self.n, self.U)

    def score(self, fit_idx):
        m = np.zeros(self.U, bool)
        m[fit_idx] = True
        rm = m[self.ud]
        fitN = np.bincount(self.cid[rm], self.n[rm], 1176)
        rare = rare_cells(fitN)
        return np.bincount(self.ud, self.n * rare[self.cid], self.U) / self.tot, int(rare.sum())


def fixed_scores(f, g):
    """Split-independent baseline scores over all user-days."""
    lv = np.log(f["n_rows"].to_numpy().astype(np.float64))
    rule = (np.log1p(f["g2_off_connect"].to_numpy().astype(np.float64)) + np.log1p(f["g2_off_file"].to_numpy().astype(np.float64))
            + f["g5_usb_first"].to_numpy().astype(np.float64))
    Z = f.select(g["Z"]).to_numpy().astype(np.float64)
    return {"volume+": lv, "volume-": -lv, "rule": rule, "zmax": np.abs(Z).max(1), "zpos": np.clip(Z, 0, None).sum(1)}


def model_sets(rel, g):
    s = {"GD": g["G"] + g["D"], "GDI": g["G"] + g["D"] + g["I"]}
    if rel == "6.2":
        s["GDR"] = g["G"] + g["D"] + g["R"]
    return s


def fit_models(X, y, fit, score_idx, sets, seed):
    """IF on every set; HGB on every set. Returns {name: (scores on score_idx, meta)}."""
    out = {}
    for sname, cols in sets.items():
        Xf = X[sname][fit]
        t0 = time.time()
        iso = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=-1).fit(Xf)
        out[f"IF_{sname}"] = (-iso.score_samples(X[sname][score_idx]),
                              {"params": iso.get_params(), "seconds": round(time.time() - t0, 1)})
        t0 = time.time()
        hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xf, y[fit])
        out[f"HGB_{sname}"] = (hgb.predict_proba(X[sname][score_idx])[:, 1],
                               {"params": hgb.get_params(), "n_iter": int(hgb.n_iter_),
                                "early_stopping_active": bool(hgb.do_early_stopping_),
                                "fit_pos": int(y[fit].sum()), "seconds": round(time.time() - t0, 1)})
    return out


def run(rel, proto):
    f, lab = load(rel)
    g = userday.groups(f, rel)
    sets = model_sets(rel, g)
    X = {k: userday.matrix(f, v) for k, v in sets.items()}
    y = lab["y_win"].to_numpy()
    ins = pb.insiders(rel)
    fixed = fixed_scores(f, g)
    rare = Rare(rel, lab)
    splits = pb.pbu_splits(lab, rel) if proto == "PBU" else [pb.pbt_split(lab, rel)]
    meta = {"release": rel, "protocol": proto, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "feature_schema_sha256": userday.schema_hash(f), "feature_sets": {k: v for k, v in sets.items()},
            "group_sizes": {k: len(v) for k, v in g.items()}, "budgets": pb.BUDGETS}
    runs = []
    for ri, sp in enumerate(splits):
        t0 = time.time()
        fit, val, test = sp["fit"], sp["val"], sp["test"]
        lab_t, lab_v = lab[test], lab[val]

        def ev(score_all=None, s_val=None, s_test=None):
            sv = score_all[val] if s_val is None else s_val
            st = score_all[test] if s_test is None else s_test
            return pb.evaluate(st, lab_t, sv, lab_v, ins, SCEN[rel])

        res, info = {}, {}
        prev = float(lab_t["y_win"].mean())
        res["random"] = {"roc_auc": 0.5, "pr_auc": prev, "analytic": True,
                         "budget": {str(b): {"recall": b, "precision": prev, "ceiling": min(1.0, max(1, round(b * len(test))) / max(1, lab_t["y_win"].sum()))}
                                    for b in pb.BUDGETS}}
        auc_fit = pb.ranking(fixed["volume+"][fit], y[fit])["roc_auc"]
        info["volume_sign"] = {"auc_fit_volume+": auc_fit, "chosen": "volume+" if auc_fit >= 0.5 else "volume-"}
        for name, s in fixed.items():
            res[name] = ev(s)
        rs, n_rare = rare.score(fit)
        info["rare_cells"] = n_rare
        res["rare"] = ev(rs)
        idx = np.r_[val, test]
        seeds = [1000 + ri] if proto == "PBU" else list(PBT_SEEDS)
        for seed in seeds:
            for name, (s, m) in fit_models(X, y, fit, idx, sets, seed).items():
                key = name if proto == "PBU" else f"{name}@{seed}"
                res[key] = ev(s_val=s[:len(val)], s_test=s[len(val):])
                info[key] = m | {"random_state": seed}
        runs.append({"rep": sp["rep"], "fold": sp["fold"], "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
                     "fit_pos": int(y[fit].sum()), "val_pos": int(y[val].sum()), "test_pos": int(y[test].sum()),
                     "info": info, "results": res, "seconds": round(time.time() - t0, 1)})
        print(f"r{rel} {proto} run {ri}: {runs[-1]['seconds']}s  " +
              "  ".join(f"{k}={v['pr_auc']:.3f}" for k, v in res.items() if not k.startswith(("random",))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"b1_r{rel}_{proto}.json").write_text(json.dumps({"meta": meta, "runs": runs}, default=str))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])

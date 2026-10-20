"""Step-5 data: every run of every protocol as weights over a fixed cell grid.

Cell ids (feature set):
  F38: cid = ((act-1)*7 + wd)*24 + (hour-1)            C = 1176
  F50: cid = F38_cid*12 + (month-1)                     C = 14112
A run is a set of (C,) row-count vectors: fit/val/test x pos/neg, under the
protocol's label (see docs/step5_preregistration.md). P3 runs also carry user
sets and a user-day view of the count table for aggregation.
"""
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from counts import atoms, label_users, load_counts, p1_bal_sample, p1_sample

N_RUNS = 10
TEST, VAL = 0.25, 0.20  # VAL = share of the train part
C = {"F38": 1176, "F50": 14112}


def cell_expr(feat):
    c38 = ((pl.col("act").cast(pl.Int32) - 1) * 7 + pl.col("wd").cast(pl.Int32)) * 24 + (pl.col("hour").cast(pl.Int32) - 1)
    return (c38 * 12 + pl.col("month").cast(pl.Int32) - 1) if feat == "F50" else c38


def cell_features(feat):
    """(C, d) float32 one-hot matrix: act(7) | weekday(7) | hour(24) [| month(12)]."""
    cid = np.arange(C[feat])
    if feat == "F50":
        month, cid = cid % 12, cid // 12
    hour, rest = cid % 24, cid // 24
    wd, act = rest % 7, rest // 7
    parts = [np.eye(7)[act], np.eye(7)[wd], np.eye(24)[hour]] + ([np.eye(12)[month]] if feat == "F50" else [])
    return np.concatenate(parts, axis=1).astype(np.float32)


def bincount(cid, w, n):
    return np.bincount(cid, weights=w, minlength=n).astype(np.float64)


@dataclass
class Run:
    fit_pos: np.ndarray
    fit_neg: np.ndarray
    val_pos: np.ndarray
    val_neg: np.ndarray
    test_pos: np.ndarray
    test_neg: np.ndarray
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- P1, P1-bal
def rows_to_runs(rows_per_seed, feat):
    """rows: DataFrame with cid and y. Exact 75/25 split, then 80/20 fit/val."""
    runs = []
    for s, rows in enumerate(rows_per_seed):
        rng = np.random.default_rng(20_000 + s)
        n = len(rows)
        perm = rng.permutation(n)
        n_te = round(n * TEST)
        n_va = round((n - n_te) * VAL)
        part = np.empty(n, dtype=np.int8)
        part[perm[:n_te]], part[perm[n_te:n_te + n_va]], part[perm[n_te + n_va:]] = 2, 1, 0
        cid, y = rows["cid"].to_numpy(), rows["y"].to_numpy()
        w = {}
        for name, code in [("fit", 0), ("val", 1), ("test", 2)]:
            m = part == code
            w[f"{name}_pos"] = bincount(cid[m], y[m], C[feat])
            w[f"{name}_neg"] = bincount(cid[m], 1 - y[m], C[feat])
        runs.append(Run(**w, info={"rows": n, "test_rows": int(n_te), "test_pos": int(w["test_pos"].sum()),
                                   "test_pos_users": int(rows.filter(pl.Series(part == 2) & (pl.col("y") == 1))["user"].n_unique())}))
    return runs


def p1_runs(rel, labelset, feat):
    users = label_users(rel, labelset)
    out = []
    for s in range(N_RUNS):
        r = p1_sample(rel, s, users)
        out.append(r.with_columns(cell_expr(feat).alias("cid"), pl.col("y_user").cast(pl.Int64).alias("y")))
    return rows_to_runs(out, feat)


def p1_bal_runs(rel, labelset, feat, counts=None):
    users = label_users(rel, labelset)
    at = atoms(counts if counts is not None else load_counts(rel), users)
    out = []
    for s in range(N_RUNS):
        r = p1_bal_sample(at, s)
        out.append(r.with_columns(cell_expr(feat).alias("cid"), pl.col("y_user").cast(pl.Int64).alias("y")))
    return rows_to_runs(out, feat)


# ---------------------------------------------------------------------- P2
def p2_runs(rel, labelset, feat, counts=None):
    """Aggregate to (cell, y) first: a sum of Binomial(k_i, p) is Binomial(sum k_i, p),
    so thinning the aggregate is exactly a uniform row split."""
    users = label_users(rel, labelset)
    counts = counts if counts is not None else load_counts(rel)
    agg = (counts.with_columns(cell_expr(feat).alias("cid"),
                               pl.col("user").is_in(pl.Series(users).implode()).cast(pl.Int64).alias("y"))
                 .group_by("cid", "y").agg(pl.col("n").sum()))
    cid, y, k = agg["cid"].to_numpy(), agg["y"].to_numpy(), agg["n"].to_numpy()
    runs = []
    for s in range(N_RUNS):
        rng = np.random.default_rng(30_000 + s)
        k_te = rng.binomial(k, TEST)
        k_va = rng.binomial(k - k_te, VAL)
        k_fit = k - k_te - k_va
        w = {}
        for name, kk in [("fit", k_fit), ("val", k_va), ("test", k_te)]:
            w[f"{name}_pos"] = bincount(cid, kk * y, C[feat])
            w[f"{name}_neg"] = bincount(cid, kk * (1 - y), C[feat])
        runs.append(Run(**w, info={"rows": int(k.sum()), "test_rows": int(k_te.sum()), "test_pos": int(w["test_pos"].sum()),
                                   "test_pos_users": len(users)}))
    return runs


# ---------------------------------------------------------------------- P3
@dataclass
class P3Data:
    users: np.ndarray            # (U,) user ids
    is_ins: np.ndarray           # (U,) bool, insider of the label set
    N: np.ndarray                # (U, C) rows per user and cell
    NW: np.ndarray               # (U, C) window-positive rows per user and cell
    ud_user: np.ndarray          # user-day view, sorted by (user, day): user index per user-day
    ud_label: np.ndarray         # (UD,) user-day positive (any window row)
    row_ud: np.ndarray           # (R,) user-day index of each count row
    row_cid: np.ndarray          # (R,) cell of each count row
    row_n: np.ndarray            # (R,) rows in that count row
    ud_start: np.ndarray         # (UD,) first count row of each user-day (rows sorted by user-day)


def p3_data(rel, labelset, feat, counts=None):
    ins = label_users(rel, labelset)
    counts = counts if counts is not None else load_counts(rel)
    df = (counts.with_columns(cell_expr(feat).alias("cid"),
                              pl.when(pl.col("user").is_in(pl.Series(ins).implode())).then(pl.col("n_win")).otherwise(0).alias("w"))
                .select("user", "day", "cid", "n", "w")
                .sort("user", "day"))
    users = np.array(sorted(df["user"].unique().to_list()))
    uidx = np.searchsorted(users, df["user"].to_numpy())
    cid, n, w = df["cid"].to_numpy(), df["n"].to_numpy(), df["w"].to_numpy()
    U, Cn = len(users), C[feat]
    N = np.zeros((U, Cn), np.float32)
    NW = np.zeros((U, Cn), np.float32)
    np.add.at(N, (uidx, cid), n)
    np.add.at(NW, (uidx, cid), w)
    day = df["day"].cast(pl.Int32).to_numpy()
    new_ud = np.r_[True, (uidx[1:] != uidx[:-1]) | (day[1:] != day[:-1])]
    row_ud = np.cumsum(new_ud) - 1
    ud_start = np.flatnonzero(new_ud)
    ud_label = np.add.reduceat(w, ud_start) > 0
    return P3Data(users=users, is_ins=np.isin(users, ins), N=N, NW=NW, ud_user=uidx[ud_start], ud_label=ud_label,
                  row_ud=row_ud, row_cid=cid, row_n=n.astype(np.float64), ud_start=ud_start)


def p3_folds(d: P3Data, rel):
    """10 runs of (fit, val, test) user-index arrays. Fixed before training."""
    U = len(d.users)
    ins_idx, nor_idx = np.flatnonzero(d.is_ins), np.flatnonzero(~d.is_ins)
    runs = []
    for rep in range(2):
        rng = np.random.default_rng(40_000 + rep)
        if rel == "6.2" and len(ins_idx) == 5:            # leave-one-insider-out
            ins_fold = rng.permutation(ins_idx)
            nor_fold = np.array_split(rng.permutation(nor_idx), 5)
            tests = [np.r_[ins_fold[f], nor_fold[f]] for f in range(5)]
        else:                                               # 5-fold stratified by insider
            i_split = np.array_split(rng.permutation(ins_idx), 5)
            n_split = np.array_split(rng.permutation(nor_idx), 5)
            tests = [np.r_[i_split[f], n_split[f]] for f in range(5)]
        for f in range(5):
            test = np.sort(tests[f])
            train = np.setdiff1d(np.arange(U), test)
            tr_ins, tr_nor = train[d.is_ins[train]], train[~d.is_ins[train]]
            r2 = np.random.default_rng(50_000 + rep * 10 + f)
            n_vi = max(1, round(len(tr_ins) * VAL)) if len(tr_ins) >= 2 else 0
            v_ins = r2.choice(tr_ins, n_vi, replace=False)
            v_nor = r2.choice(tr_nor, round(len(tr_nor) * VAL), replace=False)
            val = np.sort(np.r_[v_ins, v_nor])
            fit = np.setdiff1d(train, val)
            assert not (set(fit) & set(val)) and not (set(fit) & set(test)) and not (set(val) & set(test)), "user leak"
            assert len(fit) + len(val) + len(test) == U
            if d.is_ins[test].sum() < 1:
                print(f"WARNING: rep {rep} fold {f}: no insider in test")
            runs.append({"rep": rep, "fold": f, "fit": fit, "val": val, "test": test})
    return runs


def p3_run_weights(d: P3Data, split):
    """Row-level weights (window label) for training the cell scorer."""
    w = {}
    for name in ["fit", "val", "test"]:
        u = split[name]
        pos = d.NW[u].sum(0).astype(np.float64)
        w[f"{name}_pos"] = pos
        w[f"{name}_neg"] = d.N[u].sum(0).astype(np.float64) - pos
    test_ud = np.isin(d.ud_user, split["test"])
    return Run(**w, info={"test_users": len(split["test"]), "test_pos_users": int(d.is_ins[split["test"]].sum()),
                          "test_user_days": int(test_ud.sum()), "test_pos_user_days": int(d.ud_label[test_ud].sum()),
                          "test_rows": int(d.N[split["test"]].sum()), "test_pos": int(w["test_pos"].sum()),
                          "rep": split["rep"], "fold": split["fold"]})


def p3_aggregate(d: P3Data, cell_score):
    """Row scores -> user-day max / mean (row-weighted) -> user max of each."""
    s = cell_score[d.row_cid]
    ud_max = np.maximum.reduceat(s, d.ud_start)
    ud_mean = np.add.reduceat(s * d.row_n, d.ud_start) / np.add.reduceat(d.row_n, d.ud_start)
    U = len(d.users)
    out = {}
    for rule, uds in [("max", ud_max), ("mean", ud_mean)]:
        user = np.full(U, -np.inf)
        np.maximum.at(user, d.ud_user, uds)
        out[rule] = {"ud": uds, "user": user}
    return out

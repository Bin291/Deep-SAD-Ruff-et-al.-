"""Phase B protocols and metrics (docs/phaseB_spec.md §5, §6, B0 v2).

PB-U: exactly the Phase-A P3 folds (data5.p3_folds on the same sorted user list
      and insider flags), mapped to user-day row indices.
PB-T: fit 2010-01-02..2010-10-31, val 2010-11-01..2010-12-31, test 2011-01-01..
Budget metrics: recall@b / precision@b with k = max(1, round(b*N)); ties at the
k-th score are counted in expectation (random order). Days to first alert use
a fixed threshold tau_b = (1-b) quantile of the val scores, alert = score > tau_b.
"""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import average_precision_score, roc_auc_score

import data5
from counts import ROOT, label_users
from metrics import at_threshold, threshold_t2

BUDGETS = [0.001, 0.005, 0.01, 0.05]
PBT = {"fit_end": date(2010, 10, 31), "val_end": date(2010, 12, 31)}


class Scaler:
    """Standardisation fitted on train rows only (§4.2)."""

    def fit(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


# ------------------------------------------------------------------ splits
def pbu_splits(lab, rel):
    """lab: user-day labels sorted by (user, day). Returns 10 dicts of row indices."""
    users_ud = lab["user"].to_numpy()
    users = np.array(sorted(set(users_ud)))
    d = SimpleNamespace(users=users, is_ins=np.isin(users, label_users(rel, "main")))
    uidx = np.searchsorted(users, users_ud)
    out = []
    for sp in data5.p3_folds(d, rel):
        r = {"rep": sp["rep"], "fold": sp["fold"]}
        for k in ["fit", "val", "test"]:
            r[k] = np.flatnonzero(np.isin(uidx, sp[k]))
        out.append(r)
    return out


def pbt_split(lab, rel):
    day = lab["day"].to_numpy()
    fe, ve = np.datetime64(PBT["fit_end"]), np.datetime64(PBT["val_end"])
    return {"rep": 0, "fold": 0, "fit": np.flatnonzero(day <= fe),
            "val": np.flatnonzero((day > fe) & (day <= ve)), "test": np.flatnonzero(day > ve)}


# ------------------------------------------------------------------ metrics
def ranking(score, y):
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}


def topk_cut(score, b):
    """k, the k-th largest score s_k, #units strictly above, #ties at s_k, #ties taken."""
    n = len(score)
    k = max(1, int(round(b * n)))
    s_k = np.partition(score, n - k)[n - k]
    n_gt = int((score > s_k).sum())
    n_eq = int((score == s_k).sum())
    return k, s_k, n_gt, n_eq, k - n_gt


def recall_at(score, y, b, cut=None):
    k, s_k, n_gt, n_eq, m = cut or topk_cut(score, b)
    tp = y[score > s_k].sum() + m * y[score == s_k].sum() / n_eq
    P = y.sum()
    return {"recall": float(tp / P) if P else float("nan"), "precision": float(tp / k), "k": k,
            "ceiling": float(min(1.0, k / P)) if P else float("nan")}


def p_any_tied(j, n_eq, m):
    """P(at least one of j tied units is among m drawn from n_eq without replacement)."""
    if j == 0 or m == 0:
        return 0.0
    if n_eq - j < m:
        return 1.0
    lg = gammaln(n_eq - j + 1) - gammaln(n_eq - j - m + 1) - gammaln(n_eq + 1) + gammaln(n_eq - m + 1)
    return float(1 - np.exp(lg))


def insider_level(score, lab_t, ins, tau, b):
    """Per test insider: best positive rank, detected @b (expected), days to first alert at tau."""
    cut = topk_cut(score, b)
    _, s_k, _, n_eq, m = cut
    users = lab_t["user"].to_numpy()
    y = lab_t["y_win"].to_numpy()
    day = lab_t["day"].to_numpy()
    out = []
    for _, r in ins.iterrows():
        sel = (users == r["user"]) & (y == 1)
        if not sel.any():
            continue
        sp = score[sel]
        best = sp.max()
        rank_pct = float(((score > best).sum() + 0.5 * ((score == best).sum() - 1) + 1) / len(score))
        det = 1.0 if (sp > s_k).any() else p_any_tied(int((sp == s_k).sum()), n_eq, m)
        ws, we = np.datetime64(r["win_start"].date()), np.datetime64(r["win_end"].date())
        inwin = (users == r["user"]) & (day >= ws) & (day <= we)
        alerts = np.sort(day[inwin & (score > tau)])
        first = int((alerts[0] - ws).astype(int)) if len(alerts) else None
        out.append({"user": r["user"], "scenario": int(r["scenario"]), "pos_days": int(sel.sum()),
                    "best_rank_pct": rank_pct, "detected": det, "days_to_alert": first})
    return out


def evaluate(score, lab_t, score_val, lab_v, ins, scenarios):
    """All §6 / V2 metrics of one score vector on one test set."""
    y, ye = lab_t["y_win"].to_numpy(), lab_t["y_event"].to_numpy()
    res = {"n": int(len(y)), "pos": int(y.sum()), "pos_event": int(ye.sum()),
           "insiders": int(lab_t.filter(lab_t["y_win"] == 1)["user"].n_unique())}
    res |= ranking(score, y)
    ev = ranking(score, ye)
    res |= {"roc_auc_event": ev["roc_auc"], "pr_auc_event": ev["pr_auc"]}
    cuts = {b: topk_cut(score, b) for b in BUDGETS}
    res["budget"] = {str(b): recall_at(score, y, b, cuts[b]) for b in BUDGETS}
    sc = lab_t["scenario"].fill_null(0).to_numpy()
    res["scenario"] = {}
    for s in scenarios:
        ys = (y == 1) & (sc == s)
        keep = (y == 0) | ys
        if ys.sum() == 0:
            continue
        r = ranking(score[keep], ys[keep].astype(int))
        r["pos"] = int(ys.sum())
        r["budget"] = {str(b): recall_at(score, ys.astype(int), b, cuts[b]) for b in BUDGETS}
        res["scenario"][str(s)] = r
    yv = lab_v["y_win"].to_numpy()
    res["tau"] = {str(b): float(np.quantile(score_val, 1 - b)) for b in BUDGETS}
    res["insider"] = {str(b): insider_level(score, lab_t, ins, res["tau"][str(b)], b) for b in BUDGETS}
    if yv.sum() > 0:
        thr = threshold_t2(score_val, yv.astype(float), 1 - yv.astype(float))
        res["T2"] = at_threshold(score, y.astype(float), 1 - y.astype(float), thr) | {"threshold": thr}
    else:
        res["T2"] = None
    return res


def insiders(rel):
    return pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv", parse_dates=["win_start", "win_end"])

"""Thresholds and metrics on weighted units.

A unit set is (score, pos, neg): for cells, pos/neg are row counts; for
user-days / users, pos = label and neg = 1 - label. Two metric families:
  standard     malicious class positive: P, R, Acc, F1 = 2PR/(P+R)
  weighted     per-class P/R/F1 weighted by support (sklearn "weighted avg"),
               diagnostic only
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe(a, b):
    return a / b if b else 0.0


def confusion_metrics(tp, fp, fn, tn):
    P, N = tp + fn, tn + fp
    p1, r1 = _safe(tp, tp + fp), _safe(tp, P)
    p0, r0 = _safe(tn, tn + fn), _safe(tn, N)
    f1 = _safe(2 * p1 * r1, p1 + r1)
    f0 = _safe(2 * p0 * r0, p0 + r0)
    tot = P + N
    return {
        "precision": p1, "recall": r1, "accuracy": _safe(tp + tn, tot), "f1": f1,
        "w_precision": _safe(P * p1 + N * p0, tot), "w_recall": _safe(P * r1 + N * r0, tot),
        "w_f1": _safe(P * f1 + N * f0, tot),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


def at_threshold(score, pos, neg, thr):
    flag = score >= thr
    return confusion_metrics(pos[flag].sum(), neg[flag].sum(), pos[~flag].sum(), neg[~flag].sum())


def ranking(score, pos, neg):
    if pos.sum() == 0 or neg.sum() == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    keep_p, keep_n = pos > 0, neg > 0
    y = np.r_[np.ones(keep_p.sum()), np.zeros(keep_n.sum())]
    s = np.r_[score[keep_p], score[keep_n]]
    w = np.r_[pos[keep_p], neg[keep_n]].astype(float)
    return {"roc_auc": float(roc_auc_score(y, s, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, s, sample_weight=w))}


def _sorted_cum(score, pos, neg):
    order = np.argsort(-score, kind="stable")
    s, p, n = score[order], pos[order], neg[order]
    # cut-offs only between distinct scores (ties flagged together)
    last = np.r_[s[1:] != s[:-1], True]
    return s[last], np.cumsum(p)[last], np.cumsum(n)[last]


def threshold_t1(score, pos, neg):
    """Flag the top share of units equal to the positive prevalence."""
    tot = pos.sum() + neg.sum()
    prev = pos.sum() / tot
    s, cp, cn = _sorted_cum(score, pos, neg)
    k = int(np.searchsorted((cp + cn) / tot, prev))
    return float(s[min(k, len(s) - 1)])


def threshold_t2(score, pos, neg):
    """Threshold with the best F1 (malicious class) on this set."""
    if pos.sum() == 0:
        return float("inf")
    s, cp, cn = _sorted_cum(score, pos, neg)
    f1 = 2 * cp / (cp + cn + pos.sum())
    return float(s[int(np.argmax(f1))])


def evaluate(score, pos, neg, thresholds):
    out = {"positives": float(pos.sum()), "units": float(pos.sum() + neg.sum())} | ranking(score, pos, neg)
    for name, thr in thresholds.items():
        out[name] = at_threshold(score, pos, neg, thr) | {"threshold": thr}
    return out


def flag_nobody(pos, neg):
    return confusion_metrics(0, 0, pos.sum(), neg.sum())


def flag_all(pos, neg):
    return confusion_metrics(pos.sum(), neg.sum(), 0, 0)

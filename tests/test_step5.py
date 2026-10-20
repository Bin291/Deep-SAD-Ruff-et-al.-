import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

import data5
from metrics import confusion_metrics, threshold_t1, threshold_t2
from models import AE


def test_cell_features_roundtrip():
    for feat, d in [("F38", 38), ("F50", 50)]:
        X = data5.cell_features(feat)
        assert X.shape == (data5.C[feat], d)
        assert (X.sum(1) == (3 if feat == "F38" else 4)).all()     # one hot per group
        assert len(np.unique(X, axis=0)) == data5.C[feat]           # cells are distinct vectors


def test_weighted_loss_equals_repeated_rows():
    """Training on cell weights == training on the rows they stand for."""
    torch.manual_seed(0)
    X = torch.as_tensor(data5.cell_features("F38")[:5])
    w = torch.tensor([3.0, 1.0, 0.0, 2.0, 4.0])
    rows = X.repeat_interleave(w.long(), dim=0)
    grads = []
    for inp, weight in [(X, w), (rows, None)]:
        m = AE(1, 38, torch.Generator().manual_seed(1))
        err = ((m.decode(m.encode(inp[None])) - inp[None]) ** 2).mean(-1)[0]
        loss = (err * weight).sum() / weight.sum() if weight is not None else err.mean()
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for n, p in m.named_parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_multinomial_sampling_matches_weights():
    g = torch.Generator().manual_seed(0)
    w = torch.tensor([[1.0, 0.0, 3.0, 6.0]])
    idx = torch.multinomial(w, 200_000, replacement=True, generator=g)
    freq = torch.bincount(idx[0], minlength=4).float() / 200_000
    assert torch.allclose(freq, w[0] / w.sum(), atol=0.005)


def toy_p3():
    # 3 users; user 0 has two days, users 1 and 2 one day each; 4 cells
    users = np.array(["A", "B", "C"])
    row_ud = np.array([0, 0, 1, 2, 3])
    row_cid = np.array([0, 1, 2, 0, 3])
    row_n = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    ud_start = np.array([0, 2, 3, 4])
    ud_user = np.array([0, 0, 1, 2])
    return data5.P3Data(users=users, is_ins=np.array([True, False, False]), N=np.zeros((3, 4)), NW=np.zeros((3, 4)),
                        ud_user=ud_user, ud_label=np.array([True, False, False, False]),
                        row_ud=row_ud, row_cid=row_cid, row_n=row_n, ud_start=ud_start)


def test_p3_aggregate_max_mean():
    d = toy_p3()
    score = np.array([0.1, 0.5, 0.9, 0.2])
    a = data5.p3_aggregate(d, score)
    assert np.allclose(a["max"]["ud"], [0.5, 0.9, 0.1, 0.2])
    assert np.allclose(a["mean"]["ud"], [(0.1 * 1 + 0.5 * 3) / 4, 0.9, 0.1, 0.2])   # row-weighted
    assert np.allclose(a["max"]["user"], [0.9, 0.1, 0.2])                          # max over user-days
    assert np.allclose(a["mean"]["user"], [0.9, 0.1, 0.2])


@pytest.fixture(scope="module")
def p3_r42():
    return data5.p3_data("4.2", "main", "F38")


def test_p3_folds_no_leak_and_insiders(p3_r42):
    d = p3_r42
    splits = data5.p3_folds(d, "4.2")
    assert len(splits) == 10
    for rep in range(2):
        tests = np.concatenate([s["test"] for s in splits if s["rep"] == rep])
        assert len(tests) == len(np.unique(tests)) == len(d.users)   # each user tested once per repetition
    for s in splits:
        assert not set(s["fit"]) & set(s["test"]) and not set(s["val"]) & set(s["test"]) and not set(s["fit"]) & set(s["val"])
        assert d.is_ins[s["test"]].sum() == 14 and d.is_ins[s["val"]].sum() >= 1


def test_p3_label_consistency(p3_r42):
    d = p3_r42
    assert d.N.sum() == 32_770_222                                   # every row of r4.2
    assert d.NW.sum() == 131_342                                     # window rows (Step 2)
    assert d.ud_label.sum() == 1364                                  # positive user-days (Step 2)
    assert set(d.users[d.ud_user[d.ud_label]]) <= set(d.users[d.is_ins])


def test_thresholds():
    s = np.array([0.9, 0.8, 0.3, 0.1])
    pos = np.array([1.0, 0, 1, 0])
    neg = 1 - pos
    assert threshold_t1(s, pos, neg) == 0.8                          # flags top 50% = prevalence
    t = threshold_t2(s, pos, neg)
    m = confusion_metrics(*(lambda f: (pos[f].sum(), neg[f].sum(), pos[~f].sum(), neg[~f].sum()))(s >= t))
    assert m["f1"] == pytest.approx(max(2 * 1 / (1 + 2), 2 * 2 / (3 + 2), 2 * 2 / (4 + 2)))


def test_reproducible_same_seed(tmp_path):
    import train
    c = {"rel": "4.2", "labelset": "main", "feat": "F38", "proto": "P1", "model": "AE", "reading": "R1"}
    a = train.run_config(c, epochs_scale=0.1, out_dir=tmp_path / "a")
    b = train.run_config(c, epochs_scale=0.1, out_dir=tmp_path / "b")
    fa = [r["test"]["roc_auc"] for r in a["runs"]]
    fb = [r["test"]["roc_auc"] for r in b["runs"]]
    assert fa == fb
    assert a["loss_first_last"] == b["loss_first_last"]

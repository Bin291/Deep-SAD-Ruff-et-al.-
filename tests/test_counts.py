import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_counts  # noqa: E402
from bayes_ceiling import best_f1, prf, weighted_avg_all_normal  # noqa: E402
from counts import atoms, draw, p1_bal_sample, to_cells  # noqa: E402


@pytest.fixture
def logon_csv(tmp_path):
    p = tmp_path / "logon.csv"
    p.write_text(
        "id,date,user,pc,activity\n"
        "{A},01/04/2010 00:15:00,U1,PC-1,Logon\n"     # Monday, hour 0 -> 1
        "{B},01/04/2010 23:59:59,U1,PC-1,Logoff\n"    # hour 23 -> 24
        "{C},01/10/2010 08:00:00,U2,PC-2,Logon\n"     # Sunday
        "{D},01/10/2010 08:30:00,U2,PC-2,Logon\n")
    return p


def test_base_frame_codes_and_labels(logon_csv):
    windows = pl.DataFrame({"user": ["U2"], "win_start": [datetime(2010, 1, 10, 8, 10)],
                            "win_end": [datetime(2010, 1, 10, 9)]})
    events = pl.DataFrame({"user": ["U2"], "id": ["{D}"]})
    df = build_counts.base_frame(logon_csv, "logon", windows, events).collect().sort("id")
    assert df["act"].to_list() == [1, 2, 1, 1]
    assert df["in_win"].to_list() == [False, False, False, True]
    assert df["is_event"].to_list() == [False, False, False, True]


def test_unknown_activity_raises(tmp_path):
    p = tmp_path / "logon.csv"
    p.write_text("id,date,user,pc,activity\n{A},01/04/2010 00:15:00,U1,PC-1,Unlock\n")
    w = pl.DataFrame({"user": ["X"], "win_start": [datetime(2010, 1, 1)], "win_end": [datetime(2010, 1, 2)]})
    e = pl.DataFrame({"user": ["X"], "id": ["x"]})
    with pytest.raises(Exception):
        build_counts.base_frame(p, "logon", w, e).collect()


def test_count_lines(logon_csv):
    assert build_counts.count_lines(logon_csv) == 5


def counts_fixture():
    return pl.DataFrame({
        "user": ["U1", "U1", "U2"], "day": [date(2010, 1, 4)] * 3, "file": ["logon", "http", "http"],
        "act": [1, 7, 7], "wd": [0, 0, 0], "month": [1, 1, 1], "hour": [1, 1, 2],
        "n": [5, 7, 4], "n_win": [2, 3, 1], "n_event": [1, 0, 1],
    })


def test_atoms_conserve_counts():
    c = counts_fixture()
    a = atoms(c, ["U1"])
    assert a["k"].sum() == c["n"].sum()
    assert (a["k"] * a["y_user"]).sum() == 12            # all U1 rows
    assert (a["k"] * a["y_win"]).sum() == 5              # U2 window rows dropped: not in label set
    assert (a["k"] * a["y_event"]).sum() == 1
    assert ((a["y_event"] == 1) & (a["y_win"] == 0)).sum() == 0  # event implies window


def test_draw_all_rows_reproduces_counts():
    a = atoms(counts_fixture(), ["U1", "U2"])
    rows = draw(a, 10**6, np.random.default_rng(0))     # size capped at N -> every row once
    assert len(rows) == 16
    cells = to_cells(rows, ["act", "hour"])
    assert cells["n"].tolist() == [5, 7, 4]
    assert cells["n_win"].tolist() == [2, 3, 1]


def test_p1_bal_share():
    rng = np.random.default_rng(1)
    n = 5
    c = pl.DataFrame({
        "user": [f"U{i}" for i in range(n)] * 5, "day": [date(2010, 1, 4)] * (5 * n),
        "file": sum([[t] * n for t in ["logon", "device", "file", "email", "http"]], []),
        "act": sum([[a] * n for a in [1, 3, 6, 5, 7]], []), "wd": [0] * (5 * n), "month": [1] * (5 * n),
        "hour": list(rng.integers(1, 25, 5 * n)), "n": [10_000] * (5 * n), "n_win": [0] * (5 * n), "n_event": [0] * (5 * n),
    })
    s = p1_bal_sample(atoms(c, ["U0"]), seed=0)
    per_file = s.group_by("file").agg(pl.len(), pl.col("y_user").sum())
    assert per_file["len"].to_list() == [5000] * 5
    assert set(per_file["y_user"].to_list()) == {1667}


def test_metrics_standard_formulas():
    m = prf(tp=95, fp=10, fn=5, tn=890)
    assert m["recall"] == pytest.approx(0.95)
    assert m["f1"] == pytest.approx(2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]))
    # paper's P/R give F1 92.43 and 93.96 under the standard formula
    f = lambda p, r: 2 * p * r / (p + r)
    assert f(0.90, 0.95) == pytest.approx(0.9243, abs=1e-4)
    assert f(0.92, 0.96) == pytest.approx(0.9396, abs=1e-4)


def test_weighted_avg_matches_sklearn():
    y = np.r_[np.ones(47), np.zeros(953)]
    rep = classification_report(y, np.zeros_like(y), output_dict=True, zero_division=0)["weighted avg"]
    w = weighted_avg_all_normal(0.047)
    assert w["precision"] == pytest.approx(rep["precision"])
    assert w["recall"] == pytest.approx(rep["recall"])
    assert w["f1"] == pytest.approx(rep["f1-score"])


def test_best_f1_picks_pure_cell():
    pos = np.array([10.0, 0.0, 1.0])
    neg = np.array([0.0, 100.0, 9.0])
    r = best_f1(pos, neg, pos / (pos + neg))
    assert r["precision"] == 1.0 and r["recall"] == pytest.approx(10 / 11)

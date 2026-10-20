"""Phase B2 tests (docs/phaseB_spec.md B2-0.d d5):
(f) GPU bitwise repeat: AE pretrain + Deep SVDD (2 epochs each) and Temporal AE (1 epoch), run 0 PB-U r4.2
(g) stacked MLP == per-model nn.Module reference (CPU, float64, <= 1e-10); models in a stack are independent
(h) Temporal AE windows: past only, padding/mask, every user-day has a window, future data does not move day d
(i) Deep SAD loss and centre push rule on hand-built values
(j) Deep SAD labelled rows are fit rows; val/test labels unreadable until the final scoring; test scored once
(k) Temporal AE windows never cross a user boundary
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import deep_b2 as deep  # noqa: E402
import run_b2  # noqa: E402

TOL = 1e-10


# ------------------------------------------------------------------ synthetic data
def synth(N_users=12, seed=0, d=6):
    """Rows sorted by (user, day); heavy-ish tails; one constant column (dropped by the scaler)."""
    rng = np.random.default_rng(seed)
    codes, days = [], []
    for u in range(N_users):
        n = int(rng.integers(60, 140))
        dd = np.sort(rng.choice(400, n, replace=False))
        codes += [u] * n
        days += dd.tolist()
    X = rng.standard_t(3, size=(len(codes), d)) * rng.uniform(0.5, 3, d)
    X[:, -1] = 7.0
    return np.array(codes), np.array(days), X


def split_users(codes, groups):
    return [np.flatnonzero(np.isin(codes, g)) for g in groups]


def make_run(name, seed, X, fit, val, test):
    return {"name": name, "seed": seed, "fit": fit, "val": val, "test": test, "rep": 0, "fold": 0,
            "prm": deep.prep_params(X, fit)}


# ------------------------------------------------------------------ (g) reference
class RefMLP(nn.Module):
    def __init__(self, d, rep, decoder):
        super().__init__()
        self.e1, self.e2 = nn.Linear(d, 64, bias=False), nn.Linear(64, rep, bias=False)
        self.bn_e = nn.BatchNorm1d(64, eps=1e-4, affine=False, momentum=0.1)
        if decoder:
            self.d1, self.d2 = nn.Linear(rep, 64, bias=False), nn.Linear(64, d, bias=False)
            self.bn_d = nn.BatchNorm1d(64, eps=1e-4, affine=False, momentum=0.1)

    def encode(self, x):
        return self.e2(F.leaky_relu(self.bn_e(self.e1(x)), 0.1))

    def decode(self, z):
        return self.d2(F.leaky_relu(self.bn_d(self.d1(z)), 0.1))


def ref_prep(x, prm):
    s = (x - torch.as_tensor(prm["mean"])) / torch.as_tensor(prm["std"]) * torch.as_tensor(prm["keep"])
    return torch.sign(s) * torch.log1p(s.abs())


def ref_ae(Xt, run, rep, epochs, batch, lr):
    """One AE trained alone with torch modules and torch.optim.Adam; same seed protocol as the stack:
    generator(seed) draws e1, e2, d1, d2 (U(-1/sqrt(in), 1/sqrt(in)), (in, out)), then one randperm per epoch."""
    d = Xt.shape[1]
    g = torch.Generator().manual_seed(run["seed"])
    m = RefMLP(d, rep, True).double()
    with torch.no_grad():
        for lin, (i, o) in zip([m.e1, m.e2, m.d1, m.d2], [(d, 64), (64, rep), (rep, 64), (64, d)]):
            lin.weight.copy_(((torch.rand(i, o, generator=g, dtype=torch.float64) * 2 - 1) / np.sqrt(i)).T)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
    fit = torch.as_tensor(run["fit"])
    steps = len(fit) // batch
    curve = []
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(fit), generator=g)[: steps * batch]
        acc = 0.0
        for s in range(steps):
            x = ref_prep(Xt[fit[perm[s * batch:(s + 1) * batch]]], run["prm"])
            loss = ((m.decode(m.encode(x)) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc += float(loss)
        curve.append(acc / steps)
    m.eval()
    with torch.no_grad():
        c = deep.push_center(m.encode(ref_prep(Xt[fit], run["prm"])).mean(0))
    return m, c, np.array(curve), g


def ref_deep(Xt, run, ae_m, c, pos, eta, epochs, batch, lr):
    """Deep SVDD (pos None) / Deep SAD trained alone from a copy of the reference AE encoder."""
    d, rep = Xt.shape[1], c.shape[0]
    m = RefMLP(d, rep, False).double()
    m.e1.load_state_dict(ae_m.e1.state_dict())
    m.e2.load_state_dict(ae_m.e2.state_dict())
    m.bn_e.load_state_dict(ae_m.bn_e.state_dict())
    g = torch.Generator().manual_seed(run["seed"])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
    fit = torch.as_tensor(run["fit"])
    lab_all = torch.zeros(Xt.shape[0], dtype=torch.bool)
    if pos is not None:
        lab_all[torch.as_tensor(pos)] = True
    steps = len(fit) // batch
    curve = []
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(fit), generator=g)[: steps * batch]
        acc = 0.0
        for s in range(steps):
            b = fit[perm[s * batch:(s + 1) * batch]]
            dist = ((m.encode(ref_prep(Xt[b], run["prm"])) - c) ** 2).sum(1)
            loss = torch.where(lab_all[b], eta / (dist + 1e-6), dist).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc += float(loss)
        curve.append(acc / steps)
    return m, np.array(curve)


def ref_dist(m, Xt, rows, prm, c):
    m.eval()
    with torch.no_grad():
        return ((m.encode(ref_prep(Xt[torch.as_tensor(rows)], prm)) - c) ** 2).sum(1)


def close(a, b):
    return float(torch.as_tensor(a, dtype=torch.float64).sub(torch.as_tensor(b, dtype=torch.float64)).abs().max())


@pytest.fixture
def stack_setup(monkeypatch):
    monkeypatch.setattr(run_b2, "LR", 1e-3)   # larger steps so 3 epochs move the weights visibly
    codes, days, X = synth()
    Xt = torch.as_tensor(X, dtype=torch.float64)
    users = np.arange(12)
    # three runs, fit sets of different size (different numbers of batches per epoch -> freezing is exercised)
    groups = [(users[:3], users[3:4], users[4:5]), (users[4:8], users[8:9], users[9:10]), (users[6:11], users[11:12], users[0:1])]
    runs = []
    for i, (gf, gv, gt) in enumerate(groups):
        fit, val, test = split_users(codes, [gf, gv, gt])
        runs.append(make_run(f"run{i}", 1000 + i, X, fit, val, test))
    y = np.zeros(len(codes), np.int8)
    y[np.random.default_rng(5).choice(len(codes), 40, replace=False)] = 1
    return codes, days, X, Xt, runs, y


def test_g_stack_matches_reference_and_is_independent(stack_setup):
    codes, days, X, Xt, runs, y = stack_setup
    E, B, rep = 3, 64, 8
    assert len({len(r["fit"]) // B for r in runs}) == 3, "fit sets must differ in batches per epoch"
    ae = run_b2.stage_ae(Xt, runs, rep, "log", epochs=E, batch=B)
    worst = 0.0
    refs = []
    for k, r in enumerate(runs):
        m, c, curve, _ = ref_ae(Xt, r, rep, E, B, run_b2.LR)
        refs.append((m, c))
        st = ae[k]["state"]
        for n, lin in [("e1", m.e1), ("e2", m.e2), ("d1", m.d1), ("d2", m.d2)]:
            worst = max(worst, close(st[n], lin.weight.detach().T))
        for key, bn in [("e", m.bn_e), ("d", m.bn_d)]:
            worst = max(worst, close(st[f"bn_{key}_mean"][0], bn.running_mean), close(st[f"bn_{key}_var"][0], bn.running_var))
        worst = max(worst, close(ae[k]["curve"], curve), close(ae[k]["c"], c))
    assert worst <= TOL, f"AE stage differs from reference by {worst:.3g}"

    # stage 2: one Deep SVDD + two Deep SAD per run, all 9 in one stack
    pos = {r["name"]: r["fit"][y[r["fit"]] == 1] for r in runs}
    models = run_b2.deep_models(runs, {r["name"]: a for r, a in zip(runs, ae)}, pos, (0.5, 2.0))
    out = run_b2.stage_deep(Xt, models, "log", epochs=E, batch=B)
    worst = 0.0
    for mdl, o in zip(models, out):
        k = int(mdl["run"][-1])
        am, c = refs[k]
        m, curve = ref_deep(Xt, runs[k], am, c, mdl["pos"], mdl["eta"], E, B, run_b2.LR)
        worst = max(worst, close(o["state"]["e1"], m.e1.weight.detach().T), close(o["state"]["e2"], m.e2.weight.detach().T),
                    close(o["state"]["bn_e_mean"][0], m.bn_e.running_mean), close(o["state"]["bn_e_var"][0], m.bn_e.running_var),
                    close(o["curve"], curve))
        net = deep.StackMLP.from_states([o["state"]], False, torch.float64, "cpu")
        prm = deep.stack_params([runs[k]["prm"]], torch.float64, "cpu")
        s_stack = deep.apply_k(net, 0, Xt, torch.as_tensor(runs[k]["val"]), prm, "log", "dist", o["c"])
        worst = max(worst, close(s_stack, ref_dist(m, Xt, runs[k]["val"], runs[k]["prm"], c)))
    assert worst <= TOL, f"Deep SVDD/SAD stage differs from reference by {worst:.3g}"
    assert any(o["n_lab"] > 0 for o in out) and all(o["n_lab"] == 0 for o in out if o["model"] == "SVDD")

    # independence: model run0 alone (K=1), in reversed order, and with the other runs' seeds/data changed
    alone = run_b2.stage_ae(Xt, runs[:1], rep, "log", epochs=E, batch=B)[0]
    rev = run_b2.stage_ae(Xt, runs[::-1], rep, "log", epochs=E, batch=B)[-1]
    other = [runs[0]] + [dict(r, seed=r["seed"] + 77, fit=r["fit"][: len(r["fit"]) // 2]) for r in runs[1:]]
    chg = run_b2.stage_ae(Xt, other, rep, "log", epochs=E, batch=B)[0]
    for variant in (alone, rev, chg):
        assert max(close(variant["state"][n], ae[0]["state"][n]) for n in variant["state"]) <= TOL
        assert close(variant["curve"], ae[0]["curve"]) <= TOL and close(variant["c"], ae[0]["c"]) <= TOL


# ------------------------------------------------------------------ (i)
def test_i_center_push_and_sad_loss():
    c = deep.push_center(torch.tensor([0.05, -0.05, 0.0, 0.2, -0.3, 0.0999, -0.1]))
    assert torch.equal(c, torch.tensor([0.1, -0.1, 0.1, 0.2, -0.3, 0.1, -0.1]))

    class Fixed:
        def __init__(self, phi):
            self.phi = phi

        def encode(self, x, train, active):
            return self.phi

    phi = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.1, -0.1]],
                        [[3.0, 0.0], [0.0, 4.0], [0.1, 0.1]]], dtype=torch.float64)   # K=2, B=3, rep=2
    c = torch.tensor([[[0.1, 0.1]], [[0.0, 0.0]]], dtype=torch.float64)
    pos = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=torch.uint8)          # N=4 rows
    idx = torch.tensor([[0, 1, 2], [0, 1, 3]])
    eta = torch.tensor([[2.0], [0.5]], dtype=torch.float64)
    loss = deep.sad_loss(c, pos, eta)(Fixed(phi), None, idx, torch.tensor([True, True]))
    d0 = [0.02, 1.62, 0.04]          # model 0: row 1 labelled
    d1 = [9.0, 16.0, 0.02]           # model 1: rows 0 and 3 labelled
    want0 = (d0[0] + 2.0 / (d0[1] + 1e-6) + d0[2]) / 3
    want1 = (0.5 / (d1[0] + 1e-6) + d1[1] + 0.5 / (d1[2] + 1e-6)) / 3
    assert abs(float(loss[0]) - want0) < 1e-12 and abs(float(loss[1]) - want1) < 1e-12


# ------------------------------------------------------------------ (h), (k)
@pytest.mark.parametrize("W", [7, 14])
def test_h_k_windows(W):
    codes, days, X = synth(seed=3)
    idx, mask = deep.windows(codes, days, W)
    N = len(codes)
    assert idx.shape == mask.shape == (N, W) and mask[:, -1].all() and (idx[:, -1] == np.arange(N)).all()
    first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    n_prev = np.arange(N) - first[np.searchsorted(first, np.arange(N), side="right") - 1]
    # (k) every position, valid or not, points at a row of the same user
    assert (codes[idx] == codes[:, None]).all()
    # (h) past only, consecutive previous active days, padding exactly where history is missing
    assert (days[idx][mask] <= np.repeat(days, W).reshape(N, W)[mask]).all()
    assert (mask.sum(1) == np.minimum(n_prev + 1, W)).all()
    assert (~mask[:, : W - 1] == (np.arange(W - 1)[None, :] < (W - 1 - n_prev)[:, None])).all()
    ok = mask[:, :-1] & mask[:, 1:]
    assert (np.diff(idx, axis=1)[ok] == 1).all()

    # future data does not change the score of day d (a TAE with random weights)
    torch.manual_seed(0)
    model = deep.TAE(X.shape[1]).double()
    Xs = torch.as_tensor(X, dtype=torch.float64)
    cut = 200
    X2 = X.copy()
    X2[days > cut] = np.random.default_rng(9).normal(size=(int((days > cut).sum()), X.shape[1]))
    ti, tm = torch.as_tensor(idx), torch.as_tensor(mask, dtype=torch.float64)
    rows = torch.as_tensor(np.flatnonzero(days <= cut))
    s1 = deep.tae_scores(model, Xs, ti, tm, rows)
    s2 = deep.tae_scores(model, torch.as_tensor(X2), ti, tm, rows)
    assert torch.equal(s1, s2)
    assert not torch.equal(deep.tae_scores(model, Xs, ti, tm, torch.arange(N)), deep.tae_scores(model, torch.as_tensor(X2), ti, tm, torch.arange(N)))


def test_k_windows_cross_user_boundary_masked():
    codes = np.array([0, 0, 1, 1, 1, 2])
    days = np.array([5, 9, 1, 2, 40, 3])
    idx, mask = deep.windows(codes, days, 3)
    assert mask.tolist() == [[False, False, True], [False, True, True], [False, False, True],
                             [False, True, True], [True, True, True], [False, False, True]]
    assert idx.tolist() == [[0, 0, 0], [1, 0, 1], [2, 2, 2], [3, 2, 3], [2, 3, 4], [5, 5, 5]]
    with pytest.raises(AssertionError):
        deep.windows(np.array([0, 0]), np.array([3, 3]), 3)


# ------------------------------------------------------------------ (j)
def test_j_label_gate(stack_setup):
    codes, days, X, Xt, runs, y = stack_setup
    run = runs[0]
    lab = pl.DataFrame({"user": codes.astype(str), "day": days, "y_win": y})
    gate = deep.LabelGate(lab, run["fit"], run["val"], run["test"])
    # training stages take no labels: only arrays of fit-positive rows reach them
    for fn in (run_b2.stage_ae, run_b2.stage_deep, run_b2.stage_tae, run_b2.deep_models):
        assert not set(inspect.signature(fn).parameters) & {"lab", "y", "labels", "gate", "y_win", "lab_v", "lab_t"}
    # only select_and_score opens label frames
    for name, fn in inspect.getmembers(run_b2, inspect.isfunction):
        if fn.__module__ == "run_b2" and name != "select_and_score":
            src = inspect.getsource(fn)
            assert ".frame(" not in src and "y_win" not in src, name
    with pytest.raises(PermissionError):
        gate.frame("val")
    with pytest.raises(PermissionError):
        gate.frame("test")
    with pytest.raises(AssertionError):
        gate.unlock("test")

    ae = run_b2.stage_ae(Xt, [run], 8, "log", epochs=1, batch=64)
    pos = gate.fit_positives()
    assert np.isin(pos, run["fit"]).all() and set(pos) == set(run["fit"][y[run["fit"]] == 1]) and len(pos) > 0
    assert not np.isin(pos, np.r_[run["val"], run["test"]]).any()
    models = run_b2.deep_models([run], {run["name"]: ae[0]}, {run["name"]: pos}, (0.5, 1.0, 2.0))
    bad = [dict(models[1], pos=np.r_[pos, run["val"][:1]])]
    with pytest.raises(AssertionError):
        run_b2.stage_deep(Xt, bad, "log", epochs=1, batch=64)
    out = run_b2.stage_deep(Xt, models, "log", epochs=1, batch=64)
    gate.mark("training done")
    assert gate.events == ["read fit positives", "training done"]

    calls = []
    base = run_b2.mlp_scorer(Xt, "log")

    def scorer(cand, rows):
        part = "val" if np.array_equal(rows, run["val"]) else "test" if np.array_equal(rows, run["test"]) else "?"
        assert part != "test" or "unlock test" in gate.events, "test rows scored before test labels were opened"
        calls.append((cand["tag"], part))
        return base(cand, rows)

    cands = [o | {"prm": run["prm"], "tag": f"eta{o['eta']:g}"} for o in out if o["model"] == "SAD"]
    sel = run_b2.select_and_score(cands, scorer, gate, by_val_pr_auc=True)
    assert [p for _, p in calls].count("test") == 1 and [p for _, p in calls].count("val") == 3
    assert calls[-1] == (cands[sel["chosen"]]["tag"], "test")
    ev = gate.events
    assert ev.index("training done") < ev.index("unlock val") < ev.index("read val") < ev.index("unlock test") < ev.index("read test")
    assert ev[-1].startswith("test scored") and sum(e.startswith("test scored") for e in ev) == 1
    assert len(sel["val_crit"]) == 3


# ------------------------------------------------------------------ (f)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
def test_f_gpu_bitwise_repeat():
    if not (ROOT / "data" / "userday" / "r4.2.parquet").exists():
        pytest.skip("features not built")
    D = run_b2.load_data("4.2", "PBU", "GD")
    X = run_b2.to_dev(D.Xnp)
    run = D.runs[0]
    run["prm"] = deep.prep_params(D.Xnp, run["fit"])
    val = torch.as_tensor(run["val"], device=X.device)
    prm = deep.stack_params([run["prm"]], X.dtype, X.device)

    def once():
        ae = run_b2.stage_ae(X, [run], 16, "log", epochs=2)[0]
        svdd = run_b2.stage_deep(X, run_b2.deep_models([run], {run["name"]: ae}, None, ()), "log", epochs=2)[0]
        net = deep.StackMLP.from_states([svdd["state"]], False, X.dtype, X.device)
        s = deep.apply_k(net, 0, X, val, prm, "log", "dist", svdd["c"].to(X.device)).cpu()
        tae = run_b2.stage_tae(X, D.codes, D.days, run, 7, "log", epochs=1)
        model = deep.TAE(X.shape[1]).to(X.device)
        model.load_state_dict(tae["state"])
        widx, wmask = deep.windows(D.codes, D.days, 7)
        st = deep.tae_scores(model, run_b2.prep_full(X, run["prm"], "log"), torch.as_tensor(widx, device=X.device),
                             torch.as_tensor(wmask, dtype=X.dtype, device=X.device), val).cpu()
        return ae, svdd, s, tae, st

    a, b = once(), once()
    diff = {"ae_weights": max(close(a[0]["state"][n], b[0]["state"][n]) for n in a[0]["state"]),
            "ae_val_mse": abs(a[0]["val_mse"] - b[0]["val_mse"]),
            "svdd_weights": max(close(a[1]["state"][n], b[1]["state"][n]) for n in a[1]["state"]),
            "svdd_val_scores": close(a[2], b[2]),
            "tae_weights": max(close(a[3]["state"][n], b[3]["state"][n]) for n in a[3]["state"]),
            "tae_val_scores": close(a[4], b[4])}
    bitwise = {"svdd_val_scores": torch.equal(a[2], b[2]), "tae_val_scores": torch.equal(a[4], b[4]),
               "ae_weights": all(torch.equal(a[0]["state"][n], b[0]["state"][n]) for n in a[0]["state"]),
               "tae_weights": all(torch.equal(a[3]["state"][n], b[3]["state"][n]) for n in a[3]["state"])}
    (ROOT / "results" / "phaseB").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phaseB" / "b2_determinism.json").write_text(json.dumps(
        {"max_abs_diff": diff, "bitwise": bitwise, "versions": run_b2.versions()}, indent=1))
    assert all(bitwise.values()), f"not bitwise reproducible (tái lập xấp xỉ): {diff}"

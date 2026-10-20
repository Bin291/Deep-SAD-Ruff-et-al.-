"""Benchmark r4.2: methods of other authors on the user-day data, under the protocol of the user's
report (C:/Users/binh5/Downloads/BAO_CAO_TONG_KET_TOAN_BO.md, sections 3.6, 7.1, 12.1).

  python src/bench_r42.py smoke [MODELS]   2 seeds, few epochs, train rows only (no test labels read)
  python src/bench_r42.py run [MODELS]     5 seeds, full schedules; saves test scores, reads no labels
                                           MODELS = comma list of PAE, PVAE, RUFF (default all)
  python src/bench_r42.py report           reads the test labels once, metrics per model and seed

Protocol (reproduced and checked against the report: train 189 237 / 460 positive, test 141 140 / 479
positive = 40/431/8 by scenario, 229 test days, 75 user-days = 27 positive dropped):
  unit      user-day (data/userday/r4.2.parquet, built from HF cert_r4.2, sha256 == lfs.oid)
  label     y_event (the user-day has at least one malicious event; 966 in all)
  split     time cut 2010-10-01; an insider with malicious days on both sides goes to the side with
            more of them (tie -> train) and his user-days on the other side are dropped
  features  G+D set of phase B (88 columns; history features use only past days, so no leakage)
  prep      signed log1p, then z-score with mean / std of the normal train rows (report 3.5)
  metrics   ROC-AUC, AUC-PR (average precision), rec@k = recall in the top k x 229 test rows,
            S1/S2/S3@25 = recall of each scenario's positive test rows inside the top 25 x 229

Models (all hyperparameters fixed here before any test label is read; nothing is tuned):
  PAE   Pantelidis et al. 2021 AE  d-100(tanh)-50(relu)-50(relu) | 50-100(relu)-d (linear), MSE,
        Adam 1e-3, 30 epochs, batch 256, normal train rows; score = reconstruction MSE
  PVAE  Pantelidis VAE  d-100(tanh)-50(relu)-latent 2 | 2-50-100(relu)-d (linear), Gaussian
        (sum of squared errors) + KL, NAdam 1e-3, 1000 epochs, batch 128; score = MSE of decode(mu)
  SVDD  Ruff Deep SVDD as in docs/reference/deepsad_bn2.py: bias-free MLP d-64-32, BN(affine=False,
        eps 1e-4) + LeakyReLU(0.1); AE pretrain (loss summed over features) then SVDD, each with
        Adam 1e-4 x 50 epochs then a fresh Adam 1e-5 x 100, batch 200, weight decay 1e-6;
        c = mean phi of the normal train rows, |c_j| < 0.1 pushed to +-0.1; score = ||phi - c||^2
  SAD   Ruff Deep SAD, same, eta = 1, the 460 positive train user-days labelled (as "460 mau that")
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

import deep_b2 as deep
import userday
from baselines_b import model_sets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bench_r42"
LOG = ROOT / "logs" / "bench_r42.log"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUT = dt.date(2010, 10, 1)
SEEDS = (0, 1, 2, 3, 4)
KS = (5, 10, 25, 50)
REP = 32


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ data and split
def load():
    f = pl.read_parquet(ROOT / "data" / "userday" / "r4.2.parquet")
    lab = pl.read_parquet(ROOT / "data" / "userday" / "labels_r4.2.parquet")
    assert f.select("user", "day").equals(lab.select("user", "day"))
    cols = model_sets("4.2", userday.groups(f, "4.2"))["GD"]
    return f.select(cols).to_numpy().astype(np.float64), lab, cols


def split(lab):
    """Report 3.6. Returns (train rows, test rows, dropped rows, dropped positives)."""
    is_test = (lab["day"] >= CUT).to_numpy()
    y, users = lab["y_event"].to_numpy().astype(bool), lab["user"].to_numpy()
    drop = np.zeros(len(lab), bool)
    for u in np.unique(users[y]):
        m = users == u
        et, etr = int((y & m & is_test).sum()), int((y & m & ~is_test).sum())
        if et and etr:
            drop |= m & (is_test if et <= etr else ~is_test)
    train, test = np.where(~is_test & ~drop)[0], np.where(is_test & ~drop)[0]
    return train, test, int(drop.sum()), int(y[drop].sum())


def preprocess(X, normal_train):
    s = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = s[normal_train].mean(0), s[normal_train].std(0)
    keep = sd >= 1e-6
    return ((s[:, keep] - mu[keep]) / sd[keep]).astype(np.float32), int((~keep).sum())


# ------------------------------------------------------------------ Pantelidis AE / VAE, S seeds stacked
class SLinear(nn.Module):
    """S independent Linear layers (Glorot uniform, zero bias as Keras), weights drawn from gens[s]."""

    def __init__(self, gens, n_in, n_out):
        super().__init__()
        lim = (6.0 / (n_in + n_out)) ** 0.5
        self.w = nn.Parameter(torch.stack([(torch.rand(n_in, n_out, generator=g) * 2 - 1) * lim for g in gens]))
        self.b = nn.Parameter(torch.zeros(len(gens), 1, n_out))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class PAE(nn.Module):
    def __init__(self, gens, d):
        super().__init__()
        self.e1, self.e2, self.e3 = SLinear(gens, d, 100), SLinear(gens, 100, 50), SLinear(gens, 50, 50)
        self.d1, self.d2, self.d3 = SLinear(gens, 50, 50), SLinear(gens, 50, 100), SLinear(gens, 100, d)

    def recon(self, x):
        z = torch.relu(self.e3(torch.relu(self.e2(torch.tanh(self.e1(x))))))
        return self.d3(torch.relu(self.d2(torch.relu(self.d1(z)))))

    def loss(self, x, gen):
        return ((self.recon(x) - x) ** 2).mean((1, 2))

    def score(self, x):
        return ((self.recon(x) - x) ** 2).mean(-1)


class PVAE(nn.Module):
    def __init__(self, gens, d, latent=2):
        super().__init__()
        self.e1, self.e2 = SLinear(gens, d, 100), SLinear(gens, 100, 50)
        self.mu, self.lv = SLinear(gens, 50, latent), SLinear(gens, 50, latent)
        self.d1, self.d2, self.d3 = SLinear(gens, latent, 50), SLinear(gens, 50, 100), SLinear(gens, 100, d)

    def _h(self, x):
        return torch.relu(self.e2(torch.tanh(self.e1(x))))

    def dec(self, z):
        return self.d3(torch.relu(self.d2(torch.relu(self.d1(z)))))

    def loss(self, x, gen):
        h = self._h(x)
        mu, lv = self.mu(h), self.lv(h).clamp(-20, 20)
        z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, generator=gen, device=mu.device)
        sse = ((self.dec(z) - x) ** 2).sum(-1)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(-1)
        return (sse + kl).mean(1)

    def score(self, x):
        return ((self.dec(self.mu(self._h(x))) - x) ** 2).mean(-1)


def train_pantelidis(model, opt, X, rows, gens, epochs, batch, noise_gen):
    """Each seed s draws its own permutation of rows from gens[s] every epoch (drop_last)."""
    S, n = len(gens), len(rows)
    steps = n // batch
    curve = []
    for _ in range(epochs):
        P = torch.stack([rows[torch.randperm(n, generator=g)[: steps * batch].to(rows.device)] for g in gens])
        acc = torch.zeros(S, device=X.device)
        for i in range(steps):
            loss = model.loss(X[P[:, i * batch:(i + 1) * batch]], noise_gen)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            acc += loss.detach()
        curve.append((acc / steps).cpu().numpy())
    return np.array(curve)


@torch.no_grad()
def score_pantelidis(model, X, rows, chunk=1 << 15):
    S = model.e1.w.shape[0]
    return torch.cat([model.score(X[rows[i:i + chunk]].unsqueeze(0).expand(S, -1, -1))
                      for i in range(0, len(rows), chunk)], 1).cpu().numpy()


# ------------------------------------------------------------------ Ruff Deep SVDD / SAD (deep_b2 stack)
def ae_loss_sum(net, x, idx, active):
    """Reference line 88: squared error summed over features, mean over the batch."""
    return ((net.decode(net.encode(x, True, active), True, active) - x) ** 2).sum(-1).mean(1)


def two_phase(net, K, X, prm, fits, gens, loss_fn, epochs):
    """Reference run_phase: Adam(1e-4) x e1, then a fresh Adam(1e-5) x e2, weight decay 1e-6, batch 200."""
    curves = []
    for lr, ep in ((1e-4, epochs[0]), (1e-5, epochs[1])):
        curves.append(deep.train_stack(net, deep.StackAdam(net.params(), K, lr, 1e-6), X, prm, fits, gens, ep,
                                       loss_fn, "none", batch=200))
    return np.concatenate(curves)


def run_ruff(X, normal, pos_rows, score_rows, seeds, epochs):
    S, N, d, dev = len(seeds), X.shape[0], X.shape[1], X.device
    one = {"mean": np.zeros(d), "std": np.ones(d), "keep": np.ones(d)}
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    net = deep.StackMLP(S, d, REP, gens, True, X.dtype, dev)
    prm = deep.stack_params([one] * S, X.dtype, dev)
    t0 = time.time()
    ae_curve = two_phase(net, S, X, prm, [normal] * S, gens, ae_loss_sum, epochs)
    log(f"Ruff AE pretrain: loss {ae_curve[0].mean():.3f} -> {ae_curve[-1].mean():.3f} (sum over features), {time.time() - t0:.0f}s")
    cs = [deep.center(net, k, X, normal, prm, "none") for k in range(S)]
    states = [net.state(k) for k in range(S)]
    # models 0..S-1 = Deep SVDD (normal rows); S..2S-1 = Deep SAD (normal rows + labelled positives)
    K = 2 * S
    deep_net = deep.StackMLP.from_states(states + states, False, X.dtype, dev)
    c = torch.stack(cs + cs).to(dev, X.dtype).unsqueeze(1)
    pos = torch.zeros(K, N, dtype=torch.uint8, device=dev)
    pos[S:, pos_rows] = 1
    eta = torch.ones(K, 1, dtype=X.dtype, device=dev)
    fits = [normal] * S + [torch.cat([normal, pos_rows])] * S
    dgens = ([torch.Generator().manual_seed(10_000 + s) for s in seeds] +
             [torch.Generator().manual_seed(20_000 + s) for s in seeds])
    prm2 = deep.stack_params([one] * K, X.dtype, dev)
    t0 = time.time()
    curve = two_phase(deep_net, K, X, prm2, fits, dgens, deep.sad_loss(c, pos, eta), epochs)
    log(f"Ruff SVDD loss {curve[0, :S].mean():.4f} -> {curve[-1, :S].mean():.6f}; SAD {curve[0, S:].mean():.4f} -> "
        f"{curve[-1, S:].mean():.6f}; nan {bool(np.isnan(curve).any())}; {time.time() - t0:.0f}s")
    out = {"SVDD": [], "SAD": []}
    for k in range(K):
        out["SVDD" if k < S else "SAD"].append(deep.apply_k(deep_net, k, X, score_rows, prm2, "none", "dist", c[k]).cpu().numpy())
    curves = {"AE": ae_curve.tolist(), "SVDD": curve[:, :S].tolist(), "SAD": curve[:, S:].tolist()}
    return {m: np.stack(v) for m, v in out.items()}, curves


# ------------------------------------------------------------------ metrics (the only place test labels are read)
def metrics(score, y, scen, n_days):
    order = np.argsort(-score, kind="stable")
    out = {"ROC-AUC": roc_auc_score(y, score), "AUC-PR": average_precision_score(y, score)}
    for k in KS:
        out[f"rec@{k}"] = y[order[: k * n_days]].sum() / y.sum()
    top = np.zeros(len(y), bool)
    top[order[: 25 * n_days]] = True
    for s in (1, 2, 3):
        m = (y == 1) & (scen == s)
        out[f"S{s}@25"] = (top & m).sum() / m.sum()
    return {k: float(v) for k, v in out.items()}


def report():
    """Reads the test labels (the only place) and scores every saved model once per seed."""
    _, lab, _ = load()
    train, test, _, _ = split(lab)
    y = lab["y_event"].to_numpy()[test]
    scen = lab["scenario"].fill_null(0).to_numpy()[test]
    n_days = len(np.unique(lab["day"].to_numpy()[test]))
    assert (len(test), int(y.sum()), n_days) == (141140, 479, 229), (len(test), y.sum(), n_days)
    rows = []
    for m in ("PAE", "PVAE", "SVDD", "SAD"):
        z = np.load(OUT / f"scores_{m}.npz")
        assert np.array_equal(z["test_rows"], test)
        for i, seed in enumerate(z["seeds"]):
            rows.append({"model": m, "seed": int(seed), **metrics(z["scores"][i], y, scen, n_days)})
    per = pl.DataFrame(rows)
    per.write_csv(OUT / "per_seed.csv")
    mcols = [c for c in per.columns if c not in ("model", "seed")]
    summ = per.group_by("model", maintain_order=True).agg([pl.col(c).mean() for c in mcols] +
                                                          [pl.col(c).std().alias(f"{c}_std") for c in mcols])
    summ.write_csv(OUT / "summary.csv")
    log("report written (test labels read once)")
    with pl.Config(tbl_cols=30, tbl_width_chars=250):
        print(summ.select(["model"] + mcols))


def main(mode, which=("PAE", "PVAE", "RUFF")):
    smoke = mode == "smoke"
    t_all = time.time()
    Xnp, lab, cols = load()
    train, test, n_drop, n_drop_pos = split(lab)
    y_all = lab["y_event"].to_numpy()
    normal_train = train[y_all[train] == 0]
    pos_train = train[y_all[train] == 1]
    Xp, n_const = preprocess(Xnp, normal_train)
    log(f"bench r4.2 {mode}: features {len(cols)} (dropped constant {n_const}), train {len(train)} "
        f"(normal {len(normal_train)}, positive {len(pos_train)}), test {len(test)}, dropped {n_drop} ({n_drop_pos} positive)")
    seeds = SEEDS[:2] if smoke else SEEDS
    X = torch.as_tensor(Xp, device=DEV)
    rows_n = torch.as_tensor(normal_train, device=DEV)
    score_rows = torch.as_tensor(normal_train[:5000] if smoke else test, device=DEV)
    scores, curves, secs = {}, {}, {}

    for name, cls, opt_fn, ep, batch in (("PAE", PAE, lambda p: torch.optim.Adam(p, lr=1e-3), 30, 256),
                                         ("PVAE", PVAE, lambda p: torch.optim.NAdam(p, lr=1e-3), 1000, 128)):
        if name not in which:
            continue
        ep = 2 if smoke else ep
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        model = cls(gens, X.shape[1]).to(DEV)
        t0 = time.time()
        cv = train_pantelidis(model, opt_fn(model.parameters()), X, rows_n,
                              [torch.Generator().manual_seed(100 + s) for s in seeds], ep, batch,
                              torch.Generator(device=DEV).manual_seed(7))
        secs[name] = time.time() - t0
        model.eval()
        scores[name] = score_pantelidis(model, X, score_rows)
        curves[name] = cv.tolist()
        log(f"{name}: loss {cv[0].mean():.4f} -> {cv[-1].mean():.4f}, nan {bool(np.isnan(cv).any())}, {secs[name]:.0f}s")

    if "RUFF" in which:
        t0 = time.time()
        ruff, rc = run_ruff(X, rows_n, torch.as_tensor(pos_train, device=DEV), score_rows, seeds, (1, 1) if smoke else (50, 100))
        secs["Ruff"] = time.time() - t0
        scores.update(ruff)
        curves.update({f"Ruff_{k}": v for k, v in rc.items()})

    OUT.mkdir(parents=True, exist_ok=True)
    if smoke:
        for m, s in scores.items():
            log(f"smoke {m}: score shape {s.shape}, nan {int(np.isnan(s).sum())}, median {np.median(s):.4g}")
        log(f"smoke done in {time.time() - t_all:.0f}s")
        return
    # test scores only (no labels); `report` reads the labels once
    for m, s in scores.items():
        np.savez_compressed(OUT / f"scores_{m}.npz", scores=s, seeds=np.array(seeds), test_rows=test)
    meta = {"features": cols, "n_const_dropped": n_const, "n_train": len(train), "n_train_normal": len(normal_train),
            "n_train_pos": len(pos_train), "n_test": len(test), "dropped": n_drop, "dropped_pos": n_drop_pos,
            "seeds": list(seeds), "seconds": secs, "curves": curves,
            "versions": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (OUT / f"meta_{'_'.join(which)}.json").write_text(json.dumps(meta, indent=1, default=float))
    log(f"train {which} done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        main(sys.argv[1], tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("PAE", "PVAE", "RUFF"))

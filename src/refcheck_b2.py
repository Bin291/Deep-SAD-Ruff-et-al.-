"""Cross-check of src/deep_b2.py against the user's reference docs/reference/deepsad_bn2.py.

  python src/refcheck_b2.py equiv    numeric identity checks (centre push, layers/BN, Adam, SAD loss) + the
                                     reference one_seed() end to end on a synthetic set (quick mode)
  python src/refcheck_b2.py proto    reference training protocol (run_phase: Adam 1e-4 x 50 then a fresh
                                     Adam 1e-5 x 100, batch 200 with the last partial batch, AE loss summed
                                     over features, AE + c on unlabelled rows only) on the B2 smoke fold
                                     (r4.2 PB-U run0, fit + val, our architecture and preprocessing),
                                     label-free val diagnostics only, as in the smoke run.

The reference file is imported unchanged; one_seed() itself cannot run on CERT (it makes its own
60:40 split), so `proto` repeats its lines 85-100 on our split. Writes results/phaseB/b2_refcheck_*.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import deep_b2 as deep
import run_b2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "reference"))
import deepsad_bn2 as ref  # noqa: E402

OUT = ROOT / "results" / "phaseB"


def ref_push(c):
    """Reference lines 92-93 verbatim."""
    c = c.clone()
    c[(c.abs() < 0.1) & (c < 0)] = -0.1
    c[(c.abs() < 0.1) & (c >= 0)] = 0.1
    return c


def copy_into_ref(net, enc, dec):
    """StackMLP (K=1) weights (in, out) -> nn.Linear weights (out, in) of the reference Sequential."""
    with torch.no_grad():
        enc[0].weight.copy_(net.p["e1"][0].T)
        enc[3].weight.copy_(net.p["e2"][0].T)
        dec[0].weight.copy_(net.p["d1"][0].T)
        dec[3].weight.copy_(net.p["d2"][0].T)


def cmd_equiv():
    torch.manual_seed(0)
    dt = torch.float64
    res = {}
    # 1. centre push (reference lines 92-93) on edge values and random vectors
    edge = torch.tensor([0.0, -0.0, 1e-9, -1e-9, 0.05, -0.05, 0.0999, -0.0999, 0.1, -0.1, 0.3, -0.3], dtype=dt)
    rnd = torch.randn(100000, dtype=dt) * 0.2
    res["push_center_max_abs_diff"] = max(float((ref_push(v) - deep.push_center(v)).abs().max()) for v in (edge, rnd))

    # 2. layers and BN: reference mlp([d, 64, rep]) / mlp([rep, 64, d]) vs StackMLP K=1, same weights
    d, rep, n = 88, 16, 256
    net = deep.StackMLP(1, d, rep, [torch.Generator().manual_seed(1)], True, dt)
    enc, dec = ref.mlp([d, deep.HIDDEN, rep]).to(dt), ref.mlp([rep, deep.HIDDEN, d]).to(dt)
    res["ref_layers"] = [str(m) for m in list(enc) + list(dec)]
    copy_into_ref(net, enc, dec)
    x = torch.randn(n, d, dtype=dt) * 2
    act = torch.tensor([True])
    out_ref = dec(enc(x))
    out_my = net.decode(net.encode(x[None], True, act), True, act)[0]
    res["train_forward_max_abs_diff"] = float((out_ref - out_my).abs().max())
    res["bn_running_mean_diff"] = float((enc[1].running_mean - net.bn["e"][0][0, 0]).abs().max())
    res["bn_running_var_diff"] = float((enc[1].running_var - net.bn["e"][1][0, 0]).abs().max())
    enc.eval(), dec.eval()
    res["eval_forward_max_abs_diff"] = float((dec(enc(x)) - net.decode(net.encode(x[None]))[0]).abs().max())
    enc.train(), dec.train()

    # 3. Adam with L2 weight decay: 20 steps of torch.optim.Adam vs StackAdam on the same loss
    wd = 1e-6
    params = list(enc.parameters()) + list(dec.parameters())
    opt_r = torch.optim.Adam(params, lr=1e-3, weight_decay=wd)
    opt_m = deep.StackAdam(net.params(), 1, 1e-3, wd)
    for _ in range(20):
        xb = torch.randn(n, d, dtype=dt)
        opt_r.zero_grad()
        ((dec(enc(xb)) - xb) ** 2).mean().backward()
        opt_r.step()
        deep.ae_loss(net, xb[None], None, act).sum().backward()
        opt_m.step(act)
    w_ref = [enc[0].weight.T, enc[3].weight.T, dec[0].weight.T, dec[3].weight.T]
    res["adam_20_steps_weight_max_abs_diff"] = max(float((a - b[0]).abs().max()) for a, b in zip(w_ref, net.params()))
    # loss scale: the reference sums over features, we average (factor d)
    xb = torch.randn(n, d, dtype=dt)
    with torch.no_grad():
        l_sum = float(((dec(enc(xb)) - xb) ** 2).sum(1).mean())
        l_mean = float(((dec(enc(xb)) - xb) ** 2).mean())
    res["ae_loss_ref_over_ours"] = l_sum / l_mean

    # 4. SAD loss: reference formula (line 98) vs deep_b2.sad_loss, eta 0.5/1/2
    enc.eval()
    with torch.no_grad():
        phi = enc(x)
    c = deep.push_center(phi.mean(0))
    s = torch.zeros(n, dtype=dt)
    s[torch.randperm(n)[:20]] = -1
    lab = (s == -1).to(torch.uint8)[None]
    idx = torch.arange(n)[None]
    diffs = []
    for eta in (0.5, 1.0, 2.0):
        dd = ((phi - c) ** 2).sum(1)
        l_ref = torch.where(s == 0, dd, eta * (dd + 1e-6) ** s).mean()
        l_my = deep.sad_loss(c, lab, torch.tensor([[eta]], dtype=dt))(_Fixed(phi), None, idx, None)
        diffs.append(float((l_ref - l_my).abs().max()))
    res["sad_loss_max_abs_diff"] = max(diffs)

    # 5. the reference script end to end (quick mode) on a synthetic set: does it run here?
    rs = np.random.RandomState(0)
    Xs = np.vstack([rs.randn(1900, 20), rs.randn(100, 20) * 3 + 2]).astype(np.float32)
    ys = np.r_[np.zeros(1900, int), np.ones(100, int)]
    t0 = time.time()
    auc = ref.one_seed(Xs, ys, (32, 16, 8), 0, quick=True)
    res["ref_one_seed_quick_synthetic"] = {"auc": float(auc), "seconds": time.time() - t0}
    print(json.dumps(res, indent=1))
    (OUT / "b2_refcheck_equiv.json").write_text(json.dumps(res, indent=1))


class _Fixed:
    """Stand-in net whose encode() returns fixed phi (K=1), to test sad_loss alone."""

    def __init__(self, phi):
        self.phi = phi

    def encode(self, x, train, active):
        return self.phi[None]


# ------------------------------------------------------------------ reference protocol on the smoke fold
class Curve:
    """Wraps loss_fn / loader_fn of ref.run_phase to keep the mean loss of each epoch."""

    def __init__(self, loss_fn, loader_fn):
        self.loss_fn, self.loader_fn, self.ep = loss_fn, loader_fn, []

    def loss(self, *b):
        l = self.loss_fn(*b)
        self.ep[-1][0] += l.detach()
        self.ep[-1][1] += 1
        return l

    def loader(self):
        self.ep.append([0.0, 0])
        return self.loader_fn()

    def values(self):
        return [float(s) / k for s, k in self.ep]


def ref_pipeline(Xfit, Xval, semi, rep, seed, log_name):
    """Reference one_seed() lines 83-100 on our split (Xfit = fit rows, already preprocessed)."""
    dev = Xfit.device
    torch.manual_seed(seed)
    D = Xfit.shape[1]
    hidden = (deep.HIDDEN, rep)
    enc, dec = ref.mlp([D, *hidden]).to(dev), ref.mlp([*reversed(hidden), D]).to(dev)
    unl = Xfit[semi == 0]
    t0 = time.time()
    cv = Curve(lambda x: ((dec(enc(x)) - x) ** 2).sum(1).mean(), ref.batches((unl,)))
    ref.run_phase(list(enc.parameters()) + list(dec.parameters()), cv.loss, cv.loader, 50, 100)
    ae_s = time.time() - t0
    enc.eval(), dec.eval()
    with torch.no_grad():
        ae_val_mse = float(((dec(enc(Xval)) - Xval) ** 2).mean())
        raw_c = enc(unl).mean(0)
        c = raw_c.clone()
        c[(c.abs() < 0.1) & (c < 0)] = -0.1
        c[(c.abs() < 0.1) & (c >= 0)] = 0.1
        d_init = ((enc(Xval) - c) ** 2).sum(1).cpu().numpy()
    enc.train()
    run_b2.log(f"ref AE ({'unlabelled' if (semi != 0).any() else 'all fit'}): loss {cv.values()[0]:.3f} -> {cv.values()[-1]:.3f} "
               f"(sum over features), val MSE {ae_val_mse:.5f}, {ae_s:.0f}s", log_name)
    eps, eta = 1e-6, 1.0

    def sad_loss(x, s):
        d = ((enc(x) - c) ** 2).sum(1)
        return torch.where(s == 0, d, eta * (d + eps) ** s).mean()

    t0 = time.time()
    cs = Curve(sad_loss, ref.batches((Xfit, semi)))
    ref.run_phase(list(enc.parameters()), cs.loss, cs.loader, 50, 100)
    enc.eval()
    with torch.no_grad():
        phv = enc(Xval)
        dv = ((phv - c) ** 2).sum(1).cpu().numpy()
        dfit = ((enc(Xfit) - c) ** 2).sum(1).cpu().numpy()
    lab = (semi != 0).cpu().numpy()
    return {"ae": {"curve_sum_over_features": cv.values(), "val_mse": ae_val_mse, "seconds": ae_s, "n_rows": len(unl)},
            "center": {"pushed": int((raw_c.abs() < 0.1).sum()), "raw_c_abs_min": float(raw_c.abs().min())},
            "val_dist_at_init": run_b2.dist_stats(d_init), "curve": cs.values(), "seconds": time.time() - t0,
            "nan": bool(np.isnan(cs.values()).any() or np.isnan(cv.values()).any()),
            "val_dist": run_b2.dist_stats(dv), "val_phi": run_b2.phi_stats(phv.cpu(), c.cpu()),
            "fit_dist_unlabelled_mean": float(dfit[~lab].mean()),
            "fit_dist_labelled_mean": float(dfit[lab].mean()) if lab.any() else None}


def cmd_proto():
    name = "phaseB2_refcheck"
    D = run_b2.load_data("4.2", "PBU", "GD")
    run = D.runs[0]
    prm = deep.stack_params([deep.prep_params(D.Xnp, run["fit"])], torch.float32, run_b2.DEV)
    X = run_b2.to_dev(D.Xnp)
    Xfit = deep.prep(X[torch.as_tensor(run["fit"], device=X.device)][None], prm, "log")[0].contiguous()
    Xval = deep.prep(X[torch.as_tensor(run["val"], device=X.device)][None], prm, "log")[0].contiguous()
    gate = deep.LabelGate(D.lab, run["fit"], run["val"], run["test"])
    pos = gate.fit_positives()
    semi_sad = torch.zeros(len(run["fit"]), device=X.device)
    semi_sad[torch.as_tensor(np.isin(run["fit"], pos), device=X.device)] = -1
    res = {"run": run["name"], "seed": run["seed"], "rep": 16, "batch": ref.BS, "wd": ref.WD, "versions": run_b2.versions()}
    run_b2.log(f"refcheck proto: r4.2 PB-U {run['name']} fit {len(run['fit'])} val {len(run['val'])}, batch {ref.BS}", name)
    for model, semi in (("SVDD", torch.zeros_like(semi_sad)), ("SAD", semi_sad)):
        r = ref_pipeline(Xfit, Xval, semi, 16, run["seed"], name)
        res[model] = r
        run_b2.log(f"ref {model}: loss {r['curve'][0]:.4f} -> {r['curve'][-1]:.6f}, val dist median {r['val_dist']['median']:.4g} "
                   f"(init {r['val_dist_at_init']['median']:.4g}), phi std {r['val_phi']['phi_std_mean']:.4g}, {r['seconds']:.0f}s", name)
        (OUT / "b2_refcheck_proto.json").write_text(json.dumps(res, indent=1, default=float))
    res["gate_events"] = gate.events
    (OUT / "b2_refcheck_proto.json").write_text(json.dumps(res, indent=1, default=float))
    run_b2.log(f"refcheck proto done, gate events {gate.events}", name)


if __name__ == "__main__":
    {"equiv": cmd_equiv, "proto": cmd_proto}[sys.argv[1]]()

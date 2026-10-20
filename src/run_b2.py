"""B2 runner (docs/phaseB_spec.md B2-0.d, B2-0.e).

  python src/run_b2.py smoke                                      run 0 PB-U r4.2, one config per model, fit+val only
  python src/run_b2.py ae    REL PROTO --rep 16 [--fs GD] [--mode log|clip]   AE pretrain of every run (+ centre)
  python src/run_b2.py deep  REL PROTO [--fs GD] [--mode log] [--no-sad]       rep choice, Deep SVDD + Deep SAD
  python src/run_b2.py tae   REL PROTO --run I --W 7                           one Temporal AE
  python src/run_b2.py score REL PROTO --model SVDD|SAD|TAE [--fs] [--mode]    open val -> choose -> open test -> metrics

Training steps never see val/test labels: they get only the fit positives of a
LabelGate (Deep SAD). `score` opens val labels after every checkpoint of the group
exists, chooses eta (Deep SAD) on val PR-AUC, then opens test labels and scores the
test rows of the chosen configuration only (once per run).
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import deep_b2 as deep
import protocols_b as pb
import userday
from baselines_b import SCEN, load, model_sets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phaseB"
LOGS = ROOT / "logs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR, WD = 1e-4, 1e-6
EPOCHS = {"ae": 150, "deep": 150, "tae": 50}
REPS, WS = (16, 32), (7, 14)
ETAS = {"4.2": (0.5, 1.0, 2.0), "6.2": (1.0,)}
MAX_STACK = 40


def log(msg, name="phaseB2"):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{name}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def versions():
    v = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
         "deterministic": torch.are_deterministic_algorithms_enabled()}
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
    return v


# ------------------------------------------------------------------ data
def load_data(rel, proto, fs):
    f, lab = load(rel)
    g = userday.groups(f, rel)
    Xnp = userday.matrix(f, model_sets(rel, g)[fs])
    users = lab["user"].to_numpy()
    codes = np.unique(users, return_inverse=True)[1]
    days = lab["day"].to_numpy().astype("datetime64[D]").astype(np.int64)
    if proto == "PBU":
        runs = [dict(sp, name=f"run{i}", seed=1000 + i) for i, sp in enumerate(pb.pbu_splits(lab, rel))]
    else:
        sp = pb.pbt_split(lab, rel)
        runs = [dict(sp, name=f"seed{s}", seed=s) for s in range(5)]
    return SimpleNamespace(rel=rel, proto=proto, fs=fs, lab=lab, Xnp=Xnp, codes=codes, days=days, runs=runs)


def to_dev(Xnp, dtype=torch.float32, device=DEV):
    return torch.as_tensor(Xnp, dtype=dtype, device=device)


def ckpt_dir(rel, proto, fs, mode):
    return OUT / "b2_ckpt" / f"r{rel}_{proto}_{fs}_{mode}"


# ------------------------------------------------------------------ stages (no val/test labels)
def stage_ae(X, runs, rep, mode, epochs=EPOCHS["ae"], batch=deep.BATCH):
    """AE pretrain of every run (one stack), then centre c and val reconstruction MSE (label-free)."""
    K, dt, dev = len(runs), X.dtype, X.device
    gens = [torch.Generator().manual_seed(r["seed"]) for r in runs]
    net = deep.StackMLP(K, X.shape[1], rep, gens, True, dt, dev)
    prm = deep.stack_params([r["prm"] for r in runs], dt, dev)
    fits = [torch.as_tensor(r["fit"], device=dev) for r in runs]
    t0 = time.time()
    curve = deep.train_stack(net, deep.StackAdam(net.params(), K, LR, WD), X, prm, fits, gens, epochs, deep.ae_loss, mode, batch)
    sec = time.time() - t0
    out = []
    for k, r in enumerate(runs):
        val = torch.as_tensor(r["val"], device=dev)
        out.append({"run": r["name"], "rep": rep, "state": net.state(k), "c": deep.center(net, k, X, fits[k], prm, mode).cpu(),
                    "curve": curve[:, k].tolist(), "val_mse": float(deep.apply_k(net, k, X, val, prm, mode, "recon").mean()),
                    "seconds_stack": sec, "stack_K": K})
    return out


def pick_rep(mse):
    """mse: {rep: val MSE}. Lowest, but 16 when the relative gap is < 1%."""
    a, b = mse[16], mse[32]
    return 16 if abs(a - b) / min(a, b) < 0.01 else min(mse, key=mse.get)


def deep_models(runs, ae_sel, pos, etas):
    """One Deep SVDD per run, plus one Deep SAD per eta when pos[run] is given."""
    ms = []
    for r in runs:
        ms.append({"run": r["name"], "model": "SVDD", "eta": 0.0, "seed": r["seed"], "fit": r["fit"], "prm": r["prm"],
                   "ae": ae_sel[r["name"]], "pos": None})
        if pos is not None:
            ms += [{"run": r["name"], "model": "SAD", "eta": float(e), "seed": r["seed"], "fit": r["fit"], "prm": r["prm"],
                    "ae": ae_sel[r["name"]], "pos": pos[r["name"]]} for e in etas]
    return ms


def stage_deep(X, models, mode, epochs=EPOCHS["deep"], batch=deep.BATCH):
    """Deep SVDD / Deep SAD models of one rep in one stack. Encoder, BN statistics and c come from the
    run's pretrained AE; Deep SVDD = Deep SAD loss with no labelled rows. Returns per-model dicts."""
    K, dt, dev, N = len(models), X.dtype, X.device, X.shape[0]
    net = deep.StackMLP.from_states([m["ae"]["state"] for m in models], False, dt, dev)
    c = torch.stack([m["ae"]["c"] for m in models]).to(dev, dt).unsqueeze(1)
    pos = torch.zeros(K, N, dtype=torch.uint8, device=dev)
    for k, m in enumerate(models):
        if m["pos"] is not None:
            assert np.isin(m["pos"], m["fit"]).all(), "labelled rows outside the fit set"
            pos[k, torch.as_tensor(m["pos"], device=dev)] = 1
    eta = torch.tensor([[m["eta"]] for m in models], dtype=dt, device=dev)
    gens = [torch.Generator().manual_seed(m["seed"]) for m in models]
    prm = deep.stack_params([m["prm"] for m in models], dt, dev)
    fits = [torch.as_tensor(m["fit"], device=dev) for m in models]
    t0 = time.time()
    curve = deep.train_stack(net, deep.StackAdam(net.params(), K, LR, WD), X, prm, fits, gens, epochs,
                             deep.sad_loss(c, pos, eta), mode, batch)
    sec = time.time() - t0
    out = []
    for k, m in enumerate(models):
        n_lab = 0 if m["pos"] is None else len(m["pos"])
        out.append({"run": m["run"], "model": m["model"], "eta": m["eta"], "rep": m["ae"]["rep"], "state": net.state(k),
                    "c": c[k, 0].cpu(), "curve": curve[:, k].tolist(), "n_lab": n_lab, "gamma_l": n_lab / len(m["fit"]),
                    "seconds_stack": sec, "stack_K": K})
    return out


def prep_full(X, prm, mode):
    p = deep.stack_params([prm], X.dtype, X.device)
    return deep.prep(X.unsqueeze(0), p, mode)[0]


def stage_tae(X, codes, days, run, W, mode, epochs=EPOCHS["tae"], batch=deep.BATCH):
    dev = X.device
    Xs = prep_full(X, run["prm"], mode)
    widx, wmask = deep.windows(codes, days, W)
    widx, wmask = torch.as_tensor(widx, device=dev), torch.as_tensor(wmask, dtype=X.dtype, device=dev)
    torch.manual_seed(run["seed"])
    model = deep.TAE(X.shape[1]).to(device=dev, dtype=X.dtype)
    gen = torch.Generator().manual_seed(run["seed"])
    fit = torch.as_tensor(run["fit"], device=dev)
    t0 = time.time()
    curve = deep.train_tae(model, Xs, widx, wmask, fit, gen, epochs, batch=batch)
    sec = time.time() - t0
    val_mse = float(deep.tae_scores(model, Xs, widx, wmask, torch.as_tensor(run["val"], device=dev)).mean())
    return {"run": run["name"], "W": W, "state": {k: v.cpu() for k, v in model.state_dict().items()},
            "curve": curve.tolist(), "val_mse": val_mse, "seconds": sec}


# ------------------------------------------------------------------ scorers (eval mode)
def mlp_scorer(X, mode):
    def score(cand, rows):
        net = deep.StackMLP.from_states([cand["state"]], False, X.dtype, X.device)
        prm = deep.stack_params([cand["prm"]], X.dtype, X.device)
        c = cand["c"].to(X.device, X.dtype)
        return deep.apply_k(net, 0, X, torch.as_tensor(rows, device=X.device), prm, mode, "dist", c).cpu().numpy().astype(np.float64)
    return score


def tae_scorer(X, codes, days, mode):
    def score(cand, rows):
        widx, wmask = deep.windows(codes, days, cand["W"])
        model = deep.TAE(X.shape[1]).to(device=X.device, dtype=X.dtype)
        model.load_state_dict(cand["state"])
        Xs = prep_full(X, cand["prm"], mode)
        s = deep.tae_scores(model, Xs, torch.as_tensor(widx, device=X.device), torch.as_tensor(wmask, dtype=X.dtype, device=X.device),
                            torch.as_tensor(rows, device=X.device))
        return s.cpu().numpy().astype(np.float64)
    return score


def select_and_score(cands, score_fn, gate, by_val_pr_auc):
    """All cands of one run are trained. Opens val labels, picks the best val PR-AUC when asked
    (ties: eta closest to 1, then the smaller), opens test labels, scores test for the chosen one only."""
    gate.unlock("val")
    lab_v = gate.frame("val")
    sv = [score_fn(c, gate.parts["val"]) for c in cands]
    crit, best = None, 0
    if by_val_pr_auc and len(cands) > 1:
        yv = lab_v["y_win"].to_numpy()
        crit = [pb.ranking(s, yv)["pr_auc"] for s in sv]
        best = min(range(len(cands)), key=lambda i: (-crit[i], abs(cands[i]["eta"] - 1), cands[i]["eta"]))
    gate.unlock("test")
    lab_t = gate.frame("test")
    st = score_fn(cands[best], gate.parts["test"])
    gate.mark(f"test scored {cands[best].get('tag', best)}")
    return {"chosen": best, "val_crit": crit, "score_val": sv[best], "score_test": st, "lab_v": lab_v, "lab_t": lab_t}


# ------------------------------------------------------------------ commands
def cmd_ae(a):
    D = load_data(a.rel, a.proto, a.fs)
    X = to_dev(D.Xnp)
    runs = D.runs if a.runs is None else [D.runs[i] for i in a.runs]
    for r in runs:
        r["prm"] = deep.prep_params(D.Xnp, r["fit"])
    ck = ckpt_dir(a.rel, a.proto, a.fs, a.mode)
    ck.mkdir(parents=True, exist_ok=True)
    tag = f"r{a.rel} {a.proto} {a.fs} {a.mode} AE rep{a.rep}"
    log(f"{tag}: {len(runs)} runs stacked, {EPOCHS['ae']} epochs")
    for res, r in zip(stage_ae(X, runs, a.rep, a.mode), runs):
        torch.save(res | {"prm": r["prm"], "versions": versions()}, ck / f"ae_rep{a.rep}_{r['name']}.pt")
        log(f"{tag} {r['name']}: loss {res['curve'][0]:.4f} -> {res['curve'][-1]:.4f}, val MSE {res['val_mse']:.5f}, "
            f"{res['seconds_stack']:.0f}s (stack)")


def cmd_deep(a):
    D = load_data(a.rel, a.proto, a.fs)
    X = to_dev(D.Xnp)
    ck = ckpt_dir(a.rel, a.proto, a.fs, a.mode)
    ae_sel, choice = {}, {}
    for r in D.runs:
        ae = {rep: torch.load(ck / f"ae_rep{rep}_{r['name']}.pt", weights_only=False) for rep in REPS}
        rep = pick_rep({k: v["val_mse"] for k, v in ae.items()})
        ae_sel[r["name"]], choice[r["name"]] = ae[rep], {"rep": rep, "val_mse": {k: v["val_mse"] for k, v in ae.items()}}
        r["prm"] = ae[rep]["prm"]
    pos = None
    sad = not a.no_sad and not (a.rel == "6.2" and a.proto == "PBT")
    if sad:
        pos = {r["name"]: deep.LabelGate(D.lab, r["fit"], r["val"], r["test"]).fit_positives() for r in D.runs}
    models = deep_models(D.runs, ae_sel, pos, ETAS[a.rel])
    for rep in REPS:
        group = [m for m in models if m["ae"]["rep"] == rep]
        for i in range(0, len(group), MAX_STACK):
            part = group[i:i + MAX_STACK]
            tag = f"r{a.rel} {a.proto} {a.fs} {a.mode} deep rep{rep} [{i}:{i + len(part)}]"
            log(f"{tag}: {len(part)} models stacked, {EPOCHS['deep']} epochs")
            for res, m in zip(stage_deep(X, part, a.mode), part):
                name = f"SVDD_{m['run']}" if m["model"] == "SVDD" else f"SAD_eta{m['eta']:g}_{m['run']}"
                torch.save(res | {"prm": m["prm"], "rep_choice": choice[m["run"]], "versions": versions()}, ck / f"{name}.pt")
                log(f"{tag} {name}: loss {res['curve'][0]:.4f} -> {res['curve'][-1]:.4f}, gamma_l {res['gamma_l']:.5f}")


def cmd_tae(a):
    D = load_data(a.rel, a.proto, "GD")
    X = to_dev(D.Xnp)
    r = D.runs[a.run]
    r["prm"] = deep.prep_params(D.Xnp, r["fit"])
    ck = ckpt_dir(a.rel, a.proto, "GD", a.mode)
    ck.mkdir(parents=True, exist_ok=True)
    res = stage_tae(X, D.codes, D.days, r, a.W, a.mode)
    torch.save(res | {"prm": r["prm"], "versions": versions()}, ck / f"TAE_W{a.W}_{r['name']}.pt")
    log(f"r{a.rel} {a.proto} TAE W{a.W} {r['name']}: loss {res['curve'][0]:.4f} -> {res['curve'][-1]:.4f}, "
        f"val MSE {res['val_mse']:.5f}, {res['seconds']:.0f}s")


def cmd_score(a):
    fs = "GD" if a.model == "TAE" else a.fs
    D = load_data(a.rel, a.proto, fs)
    X = to_dev(D.Xnp)
    ck = ckpt_dir(a.rel, a.proto, fs, a.mode)
    ins = pb.insiders(a.rel)
    scorer = tae_scorer(X, D.codes, D.days, a.mode) if a.model == "TAE" else mlp_scorer(X, a.mode)
    out_runs, scores = [], {}
    for r in D.runs:
        if a.model == "SAD":
            cands = [torch.load(ck / f"SAD_eta{e:g}_{r['name']}.pt", weights_only=False) | {"tag": f"eta{e:g}"} for e in ETAS[a.rel]]
        elif a.model == "SVDD":
            cands = [torch.load(ck / f"SVDD_{r['name']}.pt", weights_only=False) | {"tag": "SVDD"}]
        else:
            tae = {W: torch.load(ck / f"TAE_W{W}_{r['name']}.pt", weights_only=False) for W in WS}
            a7, a14 = tae[7]["val_mse"], tae[14]["val_mse"]
            W = 7 if abs(a7 - a14) / min(a7, a14) < 0.01 else min(tae, key=lambda w: tae[w]["val_mse"])
            cands = [tae[W] | {"tag": f"W{W}", "val_mse_all": {w: t["val_mse"] for w, t in tae.items()}}]
        gate = deep.LabelGate(D.lab, r["fit"], r["val"], r["test"])
        sel = select_and_score(cands, scorer, gate, by_val_pr_auc=(a.model == "SAD"))
        ch = cands[sel["chosen"]]
        res = pb.evaluate(sel["score_test"], sel["lab_t"], sel["score_val"], sel["lab_v"], ins, SCEN[a.rel])
        info = {k: ch[k] for k in ("tag", "rep", "eta", "W", "gamma_l", "n_lab", "curve", "val_mse", "val_mse_all", "rep_choice", "versions")
                if k in ch}
        out_runs.append({"run": r["name"], "rep": r["rep"], "fold": r["fold"], "seed": r["seed"], "chosen": info,
                         "val_pr_auc_grid": sel["val_crit"], "gate_events": gate.events, "results": res})
        scores[f"{r['name']}_val"], scores[f"{r['name']}_test"] = sel["score_val"], sel["score_test"]
        log(f"r{a.rel} {a.proto} {a.model} {fs} {a.mode} {r['name']}: chose {ch['tag']}, test PR-AUC {res['pr_auc']:.4f}")
    name = f"b2_r{a.rel}_{a.proto}_{a.model}_{fs}" + ("" if a.mode == "log" else f"_{a.mode}")
    (OUT / f"{name}.json").write_text(json.dumps({"meta": {"release": a.rel, "protocol": a.proto, "model": a.model, "feature_set": fs,
                                                            "mode": a.mode, "versions": versions()}, "runs": out_runs}, default=str))
    (OUT / "scores").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "scores" / f"{name}.npz", **scores)


# ------------------------------------------------------------------ smoke run (B2-0.d d5)
def dist_stats(d):
    d = d.astype(np.float64)
    q = np.quantile(d, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {"mean": float(d.mean()), "std": float(d.std()), "min": float(q[0]), "p01": float(q[1]), "median": float(q[2]),
            "p99": float(q[3]), "max": float(q[4]), "n_unique": int(len(np.unique(d))), "nan": int(np.isnan(d).sum())}


def phi_stats(phi, c):
    """Spread of the representation: per-dimension std (collapse -> ~0) and mean |phi - c|."""
    sd = phi.std(0)
    return {"phi_std_mean": float(sd.mean()), "phi_std_min": float(sd.min()), "dims_std_lt_1e-3": int((sd < 1e-3).sum()),
            "mean_abs_phi_minus_c": float((phi - c).abs().mean())}


def timing_probe(X, run, mode, K, kind, epochs=1):
    """Throughput only: K copies of run 0 stacked for one epoch on fit rows; results discarded."""
    runs = [dict(run, seed=run["seed"] + 10_000 + k) for k in range(K)]
    if kind == "ae":
        r = stage_ae(X, runs, 16, mode, epochs)
    else:
        gens = [torch.Generator().manual_seed(x["seed"]) for x in runs]
        ae = stage_ae(X, runs[:1], 16, mode, 0)[0]
        ms = [{"run": x["name"], "model": "SAD", "eta": 1.0, "seed": x["seed"], "fit": x["fit"], "prm": x["prm"], "ae": ae,
               "pos": None} for x, _ in zip(runs, gens)]
        r = stage_deep(X, ms, mode, epochs)
    steps = len(run["fit"]) // deep.BATCH * epochs
    return {"K": K, "kind": kind, "seconds": r[0]["seconds_stack"], "ms_per_step": 1000 * r[0]["seconds_stack"] / steps}


def cmd_smoke(a):
    name = "phaseB2_smoke"
    t_all = time.time()
    D = load_data("4.2", "PBU", "GD")
    X = to_dev(D.Xnp)
    run = D.runs[0]
    run["prm"] = deep.prep_params(D.Xnp, run["fit"])
    gate = deep.LabelGate(D.lab, run["fit"], run["val"], run["test"])
    res = {"run": run["name"], "seed": run["seed"], "n_fit": len(run["fit"]), "n_val": len(run["val"]), "mode": "log",
           "dropped_cols": int((run["prm"]["keep"] == 0).sum()), "versions": versions()}
    dev = X.device
    val = torch.as_tensor(run["val"], device=dev)
    fit = torch.as_tensor(run["fit"], device=dev)
    log(f"smoke: r4.2 PB-U {run['name']} fit {len(fit)} val {len(val)}, dropped columns {res['dropped_cols']}", name)

    t0 = time.time()
    ae = stage_ae(X, [run], 16, "log")[0]
    res["ae"] = {"curve": ae["curve"], "val_mse": ae["val_mse"], "seconds": time.time() - t0,
                 "nan": bool(np.isnan(ae["curve"]).any())}
    c = ae["c"]
    net0 = deep.StackMLP.from_states([ae["state"]], False, X.dtype, dev)
    prm = deep.stack_params([run["prm"]], X.dtype, dev)
    raw_c = deep.apply_k(net0, 0, X, fit, prm, "log", "phi").mean(0).cpu()
    res["center"] = {"c": c.tolist(), "pushed": int((raw_c.abs() < deep.C_EPS).sum()), "raw_c_abs_min": float(raw_c.abs().min())}
    d_init = deep.apply_k(net0, 0, X, val, prm, "log", "dist", c.to(dev)).cpu().numpy()
    res["val_dist_at_init"] = dist_stats(d_init)
    log(f"smoke AE rep16: loss {ae['curve'][0]:.4f} -> {ae['curve'][-1]:.4f}, val MSE {ae['val_mse']:.5f}, "
        f"{res['ae']['seconds']:.0f}s; c pushed {res['center']['pushed']}/16", name)

    t0 = time.time()
    pos = gate.fit_positives()
    models = deep_models([run], {run["name"]: ae}, {run["name"]: pos}, (1.0,))
    out = stage_deep(X, models, "log")
    res["deep_seconds"] = time.time() - t0
    for m in out:
        net = deep.StackMLP.from_states([m["state"]], False, X.dtype, dev)
        cc = m["c"].to(dev)
        dv = deep.apply_k(net, 0, X, val, prm, "log", "dist", cc).cpu().numpy()
        phv = deep.apply_k(net, 0, X, val, prm, "log", "phi").cpu()
        dfit = deep.apply_k(net, 0, X, fit, prm, "log", "dist", cc).cpu().numpy()
        lab_fit = np.isin(run["fit"], pos)
        res[m["model"]] = {"curve": m["curve"], "nan": bool(np.isnan(m["curve"]).any()), "gamma_l": m["gamma_l"], "n_lab": m["n_lab"],
                           "val_dist": dist_stats(dv), "val_phi": phi_stats(phv, m["c"]),
                           "fit_dist_unlabelled_mean": float(dfit[~lab_fit].mean()),
                           "fit_dist_labelled_mean": float(dfit[lab_fit].mean()) if m["n_lab"] else None}
        log(f"smoke {m['model']}: loss {m['curve'][0]:.4f} -> {m['curve'][-1]:.4f}, val dist median {res[m['model']]['val_dist']['median']:.4g} "
            f"(init {res['val_dist_at_init']['median']:.4g}), phi std {res[m['model']]['val_phi']['phi_std_mean']:.4g}", name)

    t0 = time.time()
    tae = stage_tae(X, D.codes, D.days, run, 7, "log")
    res["TAE"] = {"W": 7, "curve": tae["curve"], "val_mse": tae["val_mse"], "seconds": tae["seconds"],
                  "nan": bool(np.isnan(tae["curve"]).any())}
    log(f"smoke TAE W7: loss {tae['curve'][0]:.4f} -> {tae['curve'][-1]:.4f}, val last-day MSE {tae['val_mse']:.5f}, {tae['seconds']:.0f}s", name)

    res["timing_probe"] = [timing_probe(X, run, "log", K, kind) for kind in ("ae", "sad") for K in (10, 20, 40)]
    for p in res["timing_probe"]:
        log(f"smoke timing {p['kind']} K={p['K']}: {p['ms_per_step']:.2f} ms/step", name)
    res["gate_events"] = gate.events
    res["seconds_total"] = time.time() - t_all
    res["gpu_peak_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    (OUT / "b2_smoke.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"smoke done in {res['seconds_total']:.0f}s, gate events {gate.events}", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "ae", "deep", "tae", "score"])
    ap.add_argument("rel", nargs="?")
    ap.add_argument("proto", nargs="?", choices=["PBU", "PBT"])
    ap.add_argument("--fs", default="GD", choices=["GD", "GDR"])
    ap.add_argument("--mode", default="log", choices=["log", "clip"])
    ap.add_argument("--rep", type=int, choices=REPS)
    ap.add_argument("--runs", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--run", type=int)
    ap.add_argument("--W", type=int, choices=WS)
    ap.add_argument("--model", choices=["SVDD", "SAD", "TAE"])
    ap.add_argument("--no-sad", action="store_true")
    a = ap.parse_args()
    {"smoke": cmd_smoke, "ae": cmd_ae, "deep": cmd_deep, "tae": cmd_tae, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()

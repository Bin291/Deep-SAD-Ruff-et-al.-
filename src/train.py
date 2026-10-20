"""Step 5: train + evaluate one configuration (10 runs side by side).

  python src/train.py --rel 4.2 --labelset main --feat F38 --proto P1 --model AE --reading R1 [--epochs-scale 0.1]

Writes results/runs/<config>.json (config, per-run data info, loss curves,
metrics at T1/T2 in both metric families, AUCs, baselines, timings).
Protocol details are fixed in docs/step5_preregistration.md.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

import data5
from metrics import evaluate, flag_all, flag_nobody, threshold_t1, threshold_t2
from models import EPOCH_ROWS, FINETUNE_EPOCHS, SPEC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "runs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)


def config_name(c):
    return f"r{c['rel']}_{c['labelset']}_{c['feat']}_{c['proto']}_{c['model']}_{c['reading']}"


def to_dev(a):
    return torch.as_tensor(np.stack(a), dtype=torch.float32, device=DEV)


def train_unsup(model, W, X, epochs, spec, gen):
    """Minibatches drawn with replacement from cell weights W (S, C)."""
    params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = spec["opt"](params)
    steps = math.ceil(EPOCH_ROWS / spec["batch"])
    curve = []
    for _ in range(epochs):
        acc = torch.zeros(W.shape[0], device=DEV)
        for _ in range(steps):
            idx = torch.multinomial(W, spec["batch"], replacement=True, generator=gen)
            loss = model.recon_loss(X[idx], gen)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            acc += loss.detach()
        curve.append((acc / steps).cpu().numpy())
    return np.array(curve)


def train_sup(model, Wpos, Wneg, X, epochs, spec, gen):
    """Supervised fine-tune of encoder + head. A row is drawn uniformly (cell by
    total weight, then label by the cell's positive share): same as row sampling."""
    params = [p for n, p in model.named_parameters() if not n.startswith("d")]  # encoder + head, no decoder
    opt = spec["opt"](params)
    W = Wpos + Wneg
    ppos = torch.where(W > 0, Wpos / W.clamp_min(1e-12), torch.zeros_like(W))
    steps = math.ceil(EPOCH_ROWS / spec["batch"])
    curve = []
    for _ in range(epochs):
        acc = torch.zeros(W.shape[0], device=DEV)
        for _ in range(steps):
            idx = torch.multinomial(W, spec["batch"], replacement=True, generator=gen)
            y = torch.bernoulli(torch.gather(ppos, 1, idx), generator=gen)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logit(X[idx]), y, reduction="none").mean(1)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            acc += loss.detach()
        curve.append((acc / steps).cpu().numpy())
    return np.array(curve)


def fit_and_score(c, runs, feat, epochs_scale, seed_name):
    """Train the S runs (possibly several configs stacked) -> (S, C) cell scores + loss curves."""
    spec = SPEC[c["model"]]
    S = len(runs)
    X1 = torch.as_tensor(data5.cell_features(feat), device=DEV)
    X = X1.expand(S, -1, -1)
    seed = zlib.crc32(seed_name.encode())
    gen_init = torch.Generator().manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    model = spec["cls"](S, X1.shape[1], gen_init).to(DEV)
    fpos, fneg = to_dev([r.fit_pos for r in runs]), to_dev([r.fit_neg for r in runs])
    ep = max(1, round(spec["epochs"] * epochs_scale))
    curves = {}
    # training gathers X1[idx] with idx (S, B) -> (S, B, d); scoring uses all cells X (S, C, d)
    if c["reading"] == "R1":
        curves["recon"] = train_unsup(model, fneg, X1, ep, spec, gen)   # normal rows only
        score = model.score(X)
    else:
        curves["pretrain"] = train_unsup(model, fpos + fneg, X1, ep, spec, gen)  # all rows, no labels
        curves["finetune"] = train_sup(model, fpos, fneg, X1, max(1, round(FINETUNE_EPOCHS * epochs_scale)), spec, gen)
        with torch.no_grad():
            score = torch.sigmoid(model.logit(X))
    return score.double().cpu().numpy(), curves


def eval_rows(runs, score):
    out = []
    for r, s in zip(runs, score):
        tr_pos, tr_neg = r.fit_pos + r.val_pos, r.fit_neg + r.val_neg
        thr = {"T1": threshold_t1(s, tr_pos, tr_neg), "T2": threshold_t2(s, r.val_pos, r.val_neg)}
        out.append({"info": r.info, "test": evaluate(s, r.test_pos, r.test_neg, thr),
                    "flag_nobody": flag_nobody(r.test_pos, r.test_neg), "flag_all": flag_all(r.test_pos, r.test_neg)})
    return out


def eval_p3(d, splits, runs, score):
    out = []
    for split, r, s in zip(splits, runs, score):
        agg = data5.p3_aggregate(d, s)
        res = {"info": r.info}
        for unit in ["ud", "user"]:
            owner = d.ud_user if unit == "ud" else np.arange(len(d.users))
            lab = d.ud_label if unit == "ud" else d.is_ins
            m = {k: np.isin(owner, np.r_[split["fit"], split["val"]] if k == "train" else split[k]) for k in ["train", "val", "test"]}
            pos = {k: lab[v].astype(float) for k, v in m.items()}
            neg = {k: 1 - pos[k] for k in m}
            for rule in ["max", "mean"]:
                sc = agg[rule][unit]
                sk = {k: sc[v] for k, v in m.items()}
                thr = {"T1": threshold_t1(sk["train"], pos["train"], neg["train"]),
                       "T2": threshold_t2(sk["val"], pos["val"], neg["val"])}
                res[f"{unit}_{rule}"] = evaluate(sk["test"], pos["test"], neg["test"], thr)
            res[f"{unit}_flag_nobody"] = flag_nobody(pos["test"], neg["test"])
            res[f"{unit}_flag_all"] = flag_all(pos["test"], neg["test"])
        out.append(res)
    return out


def build_runs(c, cache):
    rel, ls, feat, proto = c["rel"], c["labelset"], c["feat"], c["proto"]
    if proto == "P1":
        return data5.p1_runs(rel, ls, feat), None, None
    counts = cache.setdefault(("counts", rel), None)
    if counts is None:
        counts = cache[("counts", rel)] = data5.load_counts(rel)
    if proto == "P1-bal":
        return data5.p1_bal_runs(rel, ls, feat, counts), None, None
    if proto == "P2":
        return data5.p2_runs(rel, ls, feat, counts), None, None
    key = ("p3", rel, ls, feat)
    if key not in cache:
        cache[key] = data5.p3_data(rel, ls, feat, counts)
    d = cache[key]
    splits = data5.p3_folds(d, rel)
    return [data5.p3_run_weights(d, sp) for sp in splits], d, splits


def run_group(configs, cache=None, epochs_scale=1.0, out_dir=OUT):
    """Train configs sharing (model, reading, feat) as one stacked batch of
    independent runs (kernel-launch bound: 160 runs cost about as much as 10),
    then evaluate and save each config separately."""
    assert len({(c["model"], c["reading"], c["feat"]) for c in configs}) == 1
    cache = {} if cache is None else cache
    group = f"{configs[0]['model']}_{configs[0]['reading']}_{configs[0]['feat']}"
    t0 = time.time()
    built = [build_runs(c, cache) for c in configs]
    t_data = time.time() - t0
    all_runs = [r for runs, _, _ in built for r in runs]
    t1 = time.time()
    score, curves = fit_and_score(configs[0], all_runs, configs[0]["feat"], epochs_scale,
                                  seed_name=group + "|" + ",".join(config_name(c) for c in configs))
    t_train = time.time() - t1
    outs, start = [], 0
    for c, (runs, d, splits) in zip(configs, built):
        sl = slice(start, start + len(runs))
        start += len(runs)
        sc, cv = score[sl], {k: v[:, sl] for k, v in curves.items()}
        finite = bool(np.isfinite(sc).all() and all(np.isfinite(v).all() for v in cv.values()))
        res = eval_rows(runs, sc) if c["proto"] != "P3" else eval_p3(d, splits, runs, sc)
        out = {"config": c | {"epochs_scale": epochs_scale}, "name": config_name(c), "finite": finite,
               "group": group, "group_size_runs": len(all_runs), "group_configs": [config_name(x) for x in configs],
               "seconds": {"data_group": round(t_data, 1), "train_group": round(t_train, 1)}, "device": str(DEV),
               "loss_first_last": {k: {"first": v[0].tolist(), "last": v[-1].tolist()} for k, v in cv.items()},
               "loss_curve_mean": {k: v.mean(1).tolist() for k, v in cv.items()},
               "runs": res}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{config_name(c)}.json").write_text(json.dumps(out, default=float))
        outs.append(out)
    return outs


def run_config(c, cache=None, epochs_scale=1.0, out_dir=OUT):
    return run_group([c], cache, epochs_scale, out_dir)[0]


def all_groups():
    """Every config of docs/step5_preregistration.md, grouped for stacked training."""
    groups = []
    for model in ["AE", "VAE"]:
        for reading in ["R1", "R2"]:
            f38 = [{"rel": "4.2", "labelset": ls, "proto": p} for ls in ["main", "V30a", "V30b"]
                   for p in ["P1", "P1-bal", "P2", "P3"]]
            f38 += [{"rel": "6.2", "labelset": "main", "proto": p} for p in ["P1", "P1-bal", "P2", "P3"]]
            f50 = [{"rel": r, "labelset": "main", "proto": p} for r in ["4.2", "6.2"] for p in ["P1", "P2", "P3"]]
            for feat, cs in [("F38", f38), ("F50", f50)]:
                groups.append([c | {"feat": feat, "model": model, "reading": reading} for c in cs])
    return groups


def main_all(epochs_scale, out_dir, skip_existing=True):
    """Run every group; P3 data and counts are cached across groups of a release."""
    cache = {}
    for g in all_groups():
        names = [config_name(c) for c in g]
        if skip_existing and all((out_dir / f"{n}.json").exists() for n in names):
            print("skip", names[0].split("_", 3)[-1], flush=True)
            continue
        t = time.time()
        outs = run_group(g, cache, epochs_scale, out_dir)
        bad = [o["name"] for o in outs if not o["finite"]]
        print(f"group {outs[0]['group']}: {len(g)} configs, {outs[0]['group_size_runs']} runs, "
              f"{time.time() - t:.0f}s, non-finite: {bad or 'none'}", flush=True)
        for o in outs:
            for k, v in o["loss_curve_mean"].items():
                print(f"   {o['name']:40s} {k:8s} loss {v[0]:.4f} -> {v[-1]:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every pre-registered config")
    for k in ["rel", "labelset", "feat", "proto", "model", "reading"]:
        ap.add_argument(f"--{k}")
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.all:
        main_all(a.epochs_scale, Path(a.out))
        raise SystemExit
    if a.all is False and a.rel is None:
        ap.error("give --all or a single config")
    c = {k: getattr(a, k) for k in ["rel", "labelset", "feat", "proto", "model", "reading"]}
    r = run_config(c, epochs_scale=a.epochs_scale, out_dir=Path(a.out))
    print(r["name"], "finite", r["finite"], r["seconds"])
    for k, v in r["loss_curve_mean"].items():
        print(f"  {k}: first {v[0]:.4f} last {v[-1]:.4f}")
    if c["proto"] != "P3":
        for m in ["roc_auc", "pr_auc"]:
            print(f"  {m}: " + " ".join(f"{x['test'][m]:.3f}" for x in r["runs"]))
        print("  T2 F1: " + " ".join(f"{x['test']['T2']['f1']:.3f}" for x in r["runs"]))

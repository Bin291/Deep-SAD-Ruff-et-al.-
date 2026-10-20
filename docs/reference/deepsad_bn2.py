"""Deep SAD reproduction on classic AD benchmarks (Ruff et al., ICLR 2020, Appendix B/D/E).

Protocol (from the paper):
  - random 60:40 train/test split, stratified (keeps the outlier proportion)
  - gamma_l = 0.01 labeled anomalies (k_l = 1 -> no distinct classes here), gamma_p = 0 (clean unlabeled)
  - standardize features (zero mean / unit variance) as the only preprocessing
  - MLP encoder, no bias terms, leaky-ReLU (unbounded) -> prevents hypersphere collapse
  - AE pretraining (decoder = mirrored encoder), then Deep SAD, eta=1, lambda(weight decay)=1e-6
  - Adam; searching phase lr=1e-4 x 50 epochs, fine-tuning phase lr=1e-5 x 100 epochs, batch 200
  - c = mean of initial forward pass (excluding labeled anomalies)
  - anomaly score = ||phi(x) - c||, metric = AUC on the test split, 10 seeds
Not stated in the paper (my choices, flagged in report): hidden activation slope 0.1 (as CNN section),
batch-norm omitted (affine-free BN would add no bias; kept off for simplicity), AE epochs = 150 same schedule.
"""
import os, sys, json, time
WD = float(os.environ.get("WD", "1e-6")); BS = int(os.environ.get("BS", "200")); TAG = os.environ.get("TAG", "")
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARCH = {  # Appendix D: 3-layer MLP
    "arrhythmia": (128, 64, 32),
    "cardio": (32, 16, 8),
    "satellite": (32, 16, 8),
    "satimage-2": (32, 16, 8),
    "shuttle": (32, 16, 8),
    "thyroid": (32, 16, 4),
}
FILES = {"arrhythmia": "3_arrhythmia", "cardio": "6_cardio", "satellite": "30_satellite", "satimage-2": "31_satimage-2",
         "shuttle": "32_shuttle", "thyroid": "38_thyroid"}


def mlp(dims, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.BatchNorm1d(dims[i + 1], eps=1e-4, affine=False))
            layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)


def run_phase(params, loss_fn, loader_fn, n_search=50, n_tune=100):
    for lr, n_ep in ((1e-4, n_search), (1e-5, n_tune)):
        opt = torch.optim.Adam(params, lr=lr, weight_decay=WD)
        for _ in range(n_ep):
            for batch in loader_fn():
                opt.zero_grad()
                loss_fn(*batch).backward()
                opt.step()


def batches(tensors, bs=None):
    bs = bs or BS
    n = tensors[0].shape[0]
    def gen():
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            yield tuple(t[idx] for t in tensors)
    return gen


def one_seed(X, y, hidden, seed, gamma_l=0.01, eta=1.0, quick=False):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    # semi-supervised labels: +1 unlabeled(normal assumed), -1 labeled anomaly; 0 = unlabeled
    semi = np.zeros(len(ytr), dtype=np.float32)
    anom_idx = np.where(ytr == 1)[0]
    n_lab = max(1, int(round(gamma_l * len(ytr))))
    lab = np.random.RandomState(seed).choice(anom_idx, size=min(n_lab, len(anom_idx)), replace=False)
    semi[lab] = -1
    # clean unlabeled set (gamma_p = 0): drop the remaining training anomalies from the unlabeled pool
    keep = (ytr == 0) | (semi == -1)
    Xtr, semi = Xtr[keep], semi[keep]
    Xt = torch.tensor(Xtr, dtype=torch.float32); St = torch.tensor(semi)
    D = Xt.shape[1]
    enc = mlp([D, *hidden]); dec = mlp([*reversed(hidden), D])
    ns, nt = (2, 3) if quick else (50, 100)
    # --- AE pretraining on all training data (as in the reference setup, labeled anomalies excluded)
    unl = Xt[St == 0]
    ae_params = list(enc.parameters()) + list(dec.parameters())
    run_phase(ae_params, lambda x: ((dec(enc(x)) - x) ** 2).sum(1).mean(), batches((unl,)), ns, nt)
    enc.eval()
    with torch.no_grad():
        c = enc(unl).mean(0)
        c[(c.abs() < 0.1) & (c < 0)] = -0.1
        c[(c.abs() < 0.1) & (c >= 0)] = 0.1
    enc.train()
    eps = 1e-6
    def sad_loss(x, s):
        d = ((enc(x) - c) ** 2).sum(1)
        l = torch.where(s == 0, d, eta * (d + eps) ** s)  # unlabeled: d ; labeled: eta*(d)^(s) -> s=-1 => 1/d
        return l.mean()
    run_phase(list(enc.parameters()), sad_loss, batches((Xt, St)), ns, nt)
    enc.eval()
    with torch.no_grad():
        sc_te = ((enc(torch.tensor(Xte, dtype=torch.float32)) - c) ** 2).sum(1).sqrt().numpy()
    return roc_auc_score(yte, sc_te)


if __name__ == "__main__":
    name = sys.argv[1]; seeds = int(sys.argv[2]); quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
    d = np.load(f"/home/claude/deepsad/data/{FILES[name]}.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(int)
    torch.set_num_threads(2)
    t0 = time.time(); aucs = []
    for s in range(seeds):
        aucs.append(one_seed(X, y, ARCH[name], s, quick=quick))
        print(name, "seed", s, round(100 * aucs[-1], 2), f"{time.time()-t0:.0f}s", flush=True)
    res = {"dataset": name, "mean": 100 * float(np.mean(aucs)), "std": 100 * float(np.std(aucs)), "aucs": aucs}
    print(json.dumps(res))
    json.dump(res, open(f"/home/claude/deepsad/resultBN{TAG}_{name}{'_quick' if quick else ''}.json", "w"))

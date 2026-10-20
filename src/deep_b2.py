"""B2 deep models (docs/phaseB_spec.md §8, B2-0.d, B2-0.e).

StackMLP   K independent bias-free MLPs trained side by side: encoder d->64->rep and,
           for the AE pretrain, decoder rep->64->d. Weights are (K, in, out) and
           multiplied with bmm; BatchNorm1d(affine=False, eps 1e-4, momentum 0.1) is
           written out per model (batch statistics in training, running statistics
           in eval); LeakyReLU(0.1) after BN.
StackAdam  torch.optim.Adam (L2 weight decay) applied element-wise with a per-model
           step count, so a model whose epoch has fewer batches is frozen exactly
           (no Adam, no BN-statistics update) for the remaining steps.
           Each model has its own fit rows, its own preprocessing (mean / std / kept
           columns of its run) and its own CPU generator, so the K trainings are
           independent (test g).
TAE        GRU sequence AE over day d and the W-1 previous active days of the user.
LabelGate  y_win behind a gate: fit rows readable; val / test only after unlock (test j).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

BN_EPS, BN_MOM, SLOPE = 1e-4, 0.1, 0.1
C_EPS, SAD_EPS = 0.1, 1e-6
BATCH, HIDDEN = 256, 64
CHUNK = 1 << 16


# ------------------------------------------------------------------ preprocessing (d1)
def prep_params(X, fit):
    """Scaler of one run, fitted on its fit rows (ddof 0). Columns with std < 1e-6 are dropped (0)."""
    Xf = X[fit].astype(np.float64)
    mean, std = Xf.mean(0), Xf.std(0)
    keep = std >= 1e-6
    return {"mean": mean, "std": np.where(keep, std, 1.0), "keep": keep.astype(np.float64)}


def stack_params(prms, dtype, device):
    """List of K prep_params -> dict of (K, 1, d) tensors."""
    return {k: torch.as_tensor(np.stack([p[k] for p in prms])[:, None, :], dtype=dtype, device=device)
            for k in ("mean", "std", "keep")}


def prep(x, prm, mode, sl=slice(None)):
    """Raw rows -> standardised, dropped columns 0, then signed log1p ('log') or clip to ±10 ('clip')."""
    s = (x - prm["mean"][sl]) / prm["std"][sl] * prm["keep"][sl]
    if mode == "log":
        return torch.sign(s) * torch.log1p(s.abs())
    if mode == "clip":
        return s.clamp(-10.0, 10.0)
    if mode == "none":  # X already preprocessed (src/bench_r42.py)
        return s
    raise ValueError(mode)


# ------------------------------------------------------------------ stacked MLP (d2, d3, d6)
class StackMLP:
    def __init__(self, K, d, rep, gens=None, decoder=True, dtype=torch.float32, device="cpu"):
        """Weights U(-1/sqrt(in), 1/sqrt(in)) (the nn.Linear default), drawn from gens[k] for model k
        in the order e1, e2[, d1, d2]. gens=None leaves zeros (to be filled from a checkpoint)."""
        self.K, self.d, self.rep = K, d, rep
        shapes = [("e1", d, HIDDEN), ("e2", HIDDEN, rep)] + ([("d1", rep, HIDDEN), ("d2", HIDDEN, d)] if decoder else [])
        draws = {n: [] for n, _, _ in shapes}
        for k in range(K):
            for n, i, o in shapes:
                draws[n].append((torch.rand(i, o, generator=gens[k], dtype=torch.float64) * 2 - 1) / math.sqrt(i)
                                if gens is not None else torch.zeros(i, o, dtype=torch.float64))
        self.p = {n: torch.stack(draws[n]).to(device=device, dtype=dtype).requires_grad_() for n, _, _ in shapes}
        self.bn = {key: (torch.zeros(K, 1, HIDDEN, dtype=dtype, device=device), torch.ones(K, 1, HIDDEN, dtype=dtype, device=device))
                   for key in (["e", "d"] if decoder else ["e"])}

    def params(self):
        return list(self.p.values())

    def _bn(self, h, key, train, active, sl):
        rm, rv = self.bn[key]
        if not train:
            return (h - rm[sl]) / torch.sqrt(rv[sl] + BN_EPS)
        m = h.mean(1, keepdim=True)
        v = h.var(1, unbiased=False, keepdim=True)
        with torch.no_grad():
            n = h.shape[1]
            a = active.view(-1, 1, 1)
            rm.copy_(torch.where(a, (1 - BN_MOM) * rm + BN_MOM * m, rm))
            rv.copy_(torch.where(a, (1 - BN_MOM) * rv + BN_MOM * v * (n / (n - 1)), rv))
        return (h - m) / torch.sqrt(v + BN_EPS)

    def encode(self, x, train=False, active=None, sl=slice(None)):
        h = F.leaky_relu(self._bn(torch.bmm(x, self.p["e1"][sl]), "e", train, active, sl), SLOPE)
        return torch.bmm(h, self.p["e2"][sl])

    def decode(self, z, train=False, active=None, sl=slice(None)):
        h = F.leaky_relu(self._bn(torch.bmm(z, self.p["d1"][sl]), "d", train, active, sl), SLOPE)
        return torch.bmm(h, self.p["d2"][sl])

    def state(self, k):
        """Encoder (+ decoder) of model k as CPU tensors."""
        out = {n: w[k].detach().cpu() for n, w in self.p.items()}
        for key, (rm, rv) in self.bn.items():
            out[f"bn_{key}_mean"], out[f"bn_{key}_var"] = rm[k].cpu(), rv[k].cpu()
        return out

    @classmethod
    def from_states(cls, states, decoder, dtype, device):
        """Stack K saved states (e.g. pretrained encoders) into a new StackMLP."""
        d, rep = states[0]["e1"].shape[0], states[0]["e2"].shape[1]
        net = cls(len(states), d, rep, None, decoder, dtype, device)
        with torch.no_grad():
            for n, w in net.p.items():
                w.copy_(torch.stack([s[n] for s in states]))
            for key, (rm, rv) in net.bn.items():
                rm.copy_(torch.stack([s[f"bn_{key}_mean"] for s in states]))
                rv.copy_(torch.stack([s[f"bn_{key}_var"] for s in states]))
        return net


class StackAdam:
    """Adam exactly as torch.optim.Adam(lr, weight_decay=L2, betas, eps, amsgrad=False), per model step count."""

    def __init__(self, params, K, lr, wd, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.wd, (self.b1, self.b2), self.eps = params, lr, wd, betas, eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = torch.zeros(K, 1, 1, dtype=params[0].dtype, device=params[0].device)

    @torch.no_grad()
    def step(self, active):
        a = active.view(-1, 1, 1)
        self.t += a
        t = self.t.clamp(min=1)
        bc1, bc2s = 1 - self.b1 ** t, torch.sqrt(1 - self.b2 ** t)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad + self.wd * p
            m.copy_(torch.where(a, m.lerp(g, 1 - self.b1), m))
            v.copy_(torch.where(a, v * self.b2 + (1 - self.b2) * g * g, v))
            p.sub_(torch.where(a, (self.lr / bc1) * m / (v.sqrt() / bc2s + self.eps), torch.zeros_like(p)))
            p.grad = None


def ae_loss(net, x, idx, active):
    """(K,) mean squared reconstruction error over batch and features."""
    return ((net.decode(net.encode(x, True, active), True, active) - x) ** 2).mean((1, 2))


def sad_loss(c, pos, eta):
    """Deep SAD loss (Ruff 2020) per model: unlabelled rows ||phi-c||^2, labelled positives
    eta (||phi-c||^2 + eps)^-1, batch mean. pos: (K, N) uint8, 1 = labelled positive of model k
    (all 0 for Deep SVDD, which makes the loss the Deep SVDD loss). eta: (K, 1)."""
    def f(net, x, idx, active):
        dist = ((net.encode(x, True, active) - c) ** 2).sum(-1)
        lab = torch.gather(pos, 1, idx).bool()
        return torch.where(lab, eta / (dist + SAD_EPS), dist).mean(1)
    return f


def train_stack(net, opt, X, prm, fits, gens, epochs, loss_fn, mode, batch=BATCH):
    """fits: K LongTensors of rows of X (on X.device). Each epoch model k draws
    randperm(len(fits[k])) from gens[k] and takes len // batch full batches (drop_last);
    for steps beyond its own count model k is frozen. Returns per-epoch mean loss (epochs, K)."""
    K, dev = len(fits), X.device
    n = [len(f) for f in fits]
    steps = torch.tensor([m // batch for m in n], device=dev)
    S = int(steps.max())
    curve = []
    for _ in range(epochs):
        P = torch.zeros(K, S * batch, dtype=torch.long, device=dev)
        for k in range(K):
            perm = torch.randperm(n[k], generator=gens[k])[: (n[k] // batch) * batch].to(dev)
            P[k, : len(perm)] = fits[k][perm]
        acc = torch.zeros(K, dtype=X.dtype, device=dev)
        for s in range(S):
            active = s < steps
            idx = P[:, s * batch:(s + 1) * batch]
            loss = loss_fn(net, prep(X[idx], prm, mode), idx, active)
            (loss * active).sum().backward()
            opt.step(active)
            acc += loss.detach() * active
        curve.append((acc / steps).cpu().numpy())
    return np.array(curve).reshape(epochs, K)


@torch.no_grad()
def apply_k(net, k, X, rows, prm, mode, what, c=None):
    """Eval-mode outputs of model k on rows: 'phi' (n, rep), 'dist' ||phi-c||^2, 'recon' MSE per row."""
    sl = slice(k, k + 1)
    out = []
    for i in range(0, len(rows), CHUNK):
        x = prep(X[rows[i:i + CHUNK]].unsqueeze(0), prm, mode, sl)
        z = net.encode(x, sl=sl)
        if what == "phi":
            out.append(z[0])
        elif what == "dist":
            out.append(((z[0] - c) ** 2).sum(-1))
        else:
            out.append(((net.decode(z, sl=sl) - x) ** 2).mean(-1)[0])
    return torch.cat(out)


def push_center(c):
    """|c_j| < 0.1 -> 0.1 sign(c_j); c_j = 0 -> +0.1 (B2-0.e item 1)."""
    c = c.clone()
    small = c.abs() < C_EPS
    c[small] = C_EPS * torch.where(c[small] < 0, -1.0, 1.0).to(c.dtype)  # 0.1 in c's dtype, not float32
    return c


def center(net, k, X, fit, prm, mode):
    """Mean of phi over the fit rows of model k (eval mode), then push_center."""
    return push_center(apply_k(net, k, X, fit, prm, mode, "phi").mean(0))


# ------------------------------------------------------------------ temporal AE (d4)
class TAE(nn.Module):
    def __init__(self, d, h=HIDDEN):
        super().__init__()
        self.enc = nn.GRU(d, h, batch_first=True)
        self.dec = nn.GRU(h, h, batch_first=True)
        self.out = nn.Linear(h, d)

    def forward(self, x):
        _, z = self.enc(x)
        y, _ = self.dec(z[-1].unsqueeze(1).expand(-1, x.shape[1], -1).contiguous())
        return self.out(y)


def windows(user_codes, days, W):
    """Rows sorted by (user, day). Window of row i = [i-W+1 .. i] restricted to the same user:
    position W-1 is day d itself, position W-1-j its j-th previous active day. Positions without
    such a day get mask 0 and point at row i (their input is zeroed). Returns idx (N, W), mask (N, W)."""
    u, dd = np.asarray(user_codes), np.asarray(days)
    assert np.all((u[1:] > u[:-1]) | ((u[1:] == u[:-1]) & (dd[1:] > dd[:-1]))), "rows not sorted by (user, day)"
    i = np.arange(len(u))[:, None]
    src = i - np.arange(W - 1, -1, -1)[None, :]
    ok = src >= 0
    ok &= u[np.where(ok, src, 0)] == u[:, None]
    return np.where(ok, src, i), ok


def tae_batch(Xs, widx, wmask, rows):
    m = wmask[rows].unsqueeze(-1)
    return Xs[widx[rows]] * m, m


def train_tae(model, Xs, widx, wmask, fit, gen, epochs, lr=1e-3, batch=BATCH, clip=1.0):
    """Masked MSE over valid positions and features; Adam(lr, wd 0); grad-norm clip; drop_last."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    steps = len(fit) // batch
    curve = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(fit), generator=gen)[: steps * batch].to(Xs.device)
        acc = torch.zeros((), dtype=Xs.dtype, device=Xs.device)
        for s in range(steps):
            x, m = tae_batch(Xs, widx, wmask, fit[perm[s * batch:(s + 1) * batch]])
            loss = (((model(x) - x) ** 2) * m).sum() / (m.sum() * x.shape[-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            acc += loss.detach()
        curve.append(float(acc / steps))
    return np.array(curve)


@torch.no_grad()
def tae_scores(model, Xs, widx, wmask, rows):
    """MSE over features at the last window position (day d itself), eval mode."""
    model.eval()
    out = []
    for i in range(0, len(rows), CHUNK):
        x, _ = tae_batch(Xs, widx, wmask, rows[i:i + CHUNK])
        out.append(((model(x)[:, -1] - x[:, -1]) ** 2).mean(-1))
    return torch.cat(out)


# ------------------------------------------------------------------ label gate (test j)
class LabelGate:
    """y_win of one split behind a gate: fit positives always; val / test frames only after unlock."""

    def __init__(self, lab, fit, val, test):
        self._lab, self._y = lab, lab["y_win"].to_numpy()
        self.parts = {"fit": fit, "val": val, "test": test}
        self.unlocked, self.events = set(), []

    def fit_positives(self):
        f = self.parts["fit"]
        self.events.append("read fit positives")
        return f[self._y[f] == 1]

    def mark(self, event):
        self.events.append(event)

    def unlock(self, part):
        assert part in ("val", "test") and (part == "val" or "val" in self.unlocked), "unlock val before test"
        self.events.append(f"unlock {part}")
        self.unlocked.add(part)

    def frame(self, part):
        if part not in self.unlocked:
            raise PermissionError(f"labels of {part} are still locked")
        self.events.append(f"read {part}")
        return self._lab[self.parts[part]]

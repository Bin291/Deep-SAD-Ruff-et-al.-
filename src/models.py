"""AE and VAE for S independent runs trained side by side.

Each layer holds S weight matrices (S, in, out); inputs are (S, B, in). The
loss is the sum over runs of each run's mean loss, so gradients never mix
across runs, and Adam/NAdam act element-wise, so this is exactly S separate
trainings sharing one GPU kernel launch. Architecture per the paper (input d,
100, 50, bottleneck 50 for AE; latent 2 for VAE); see
docs/step5_preregistration.md for the choices the paper leaves open.
"""
import math

import torch
from torch import nn


class BLinear(nn.Module):
    def __init__(self, S, n_in, n_out, gen):
        super().__init__()
        lim = math.sqrt(6.0 / (n_in + n_out))  # Glorot uniform (Keras default), zero bias
        self.w = nn.Parameter((torch.rand(S, n_in, n_out, generator=gen) * 2 - 1) * lim)
        self.b = nn.Parameter(torch.zeros(S, 1, n_out))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class AE(nn.Module):
    def __init__(self, S, d, gen):
        super().__init__()
        self.e1, self.e2, self.e3 = BLinear(S, d, 100, gen), BLinear(S, 100, 50, gen), BLinear(S, 50, 50, gen)
        self.d1, self.d2, self.d3 = BLinear(S, 50, 50, gen), BLinear(S, 50, 100, gen), BLinear(S, 100, d, gen)
        self.head = BLinear(S, 50, 1, gen)
        self.z_dim = 50

    def encode(self, x):
        return torch.relu(self.e3(torch.relu(self.e2(torch.tanh(self.e1(x))))))

    def decode(self, z):
        return torch.sigmoid(self.d3(torch.relu(self.d2(torch.relu(self.d1(z))))))

    def recon_loss(self, x, gen=None):
        """(S,) per-run mean MSE; the training objective is its sum."""
        return ((self.decode(self.encode(x)) - x) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def score(self, x):
        """R1 anomaly score per input: MSE averaged over dimensions. x: (S, C, d) -> (S, C)."""
        return ((self.decode(self.encode(x)) - x) ** 2).mean(-1)

    def logit(self, x):
        return self.head(self.encode(x)).squeeze(-1)


class VAE(nn.Module):
    def __init__(self, S, d, gen, latent=2):
        super().__init__()
        self.e1, self.e2 = BLinear(S, d, 100, gen), BLinear(S, 100, 50, gen)
        self.mu, self.lv = BLinear(S, 50, latent, gen), BLinear(S, 50, latent, gen)
        self.d1, self.d2, self.d3 = BLinear(S, latent, 50, gen), BLinear(S, 50, 100, gen), BLinear(S, 100, d, gen)
        self.head = BLinear(S, latent, 1, gen)
        self.z_dim = latent

    def _h(self, x):
        return torch.relu(self.e2(torch.tanh(self.e1(x))))

    def encode(self, x):
        return self.mu(self._h(x))

    def decode_logits(self, z):
        return self.d3(torch.relu(self.d2(torch.relu(self.d1(z)))))

    def recon_loss(self, x, gen=None):
        """(S,) negative ELBO per run: BCE summed over dims + KL, batch mean."""
        h = self._h(x)
        mu, lv = self.mu(h), self.lv(h).clamp(-20, 20)
        eps = torch.randn(mu.shape, generator=gen, device=mu.device)
        z = mu + torch.exp(0.5 * lv) * eps
        bce = nn.functional.binary_cross_entropy_with_logits(self.decode_logits(z), x, reduction="none").sum(-1)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(-1)
        return (bce + kl).mean(1)

    @torch.no_grad()
    def score(self, x):
        """R1 anomaly score: BCE summed over dims between x and decoder(mu)."""
        return nn.functional.binary_cross_entropy_with_logits(
            self.decode_logits(self.encode(x)), x, reduction="none").sum(-1)

    def logit(self, x):
        return self.head(self.encode(x)).squeeze(-1)


SPEC = {
    "AE": {"cls": AE, "epochs": 30, "batch": 256, "opt": lambda p: torch.optim.Adam(p, lr=1e-3)},
    "VAE": {"cls": VAE, "epochs": 1000, "batch": 128, "opt": lambda p: torch.optim.NAdam(p, lr=1e-3)},
}
FINETUNE_EPOCHS = 30   # R2 supervised stage [our choice]
EPOCH_ROWS = 15_000    # rows drawn per epoch, all protocols [our choice, = P1 fit size]

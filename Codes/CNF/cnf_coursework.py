"""
CNF Coursework  –  all four sections in one script.

Sections
--------
1. CNF model definition and training utility.
2. Fit a small (null) and large (alternative) nested CNF; compare NLL.
3. Likelihood ratio test via Wilks' theorem.
4. Q–Q plots of latent coordinates for the larger model.

Run:
    python cnf_coursework.py
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from data_utils_2d import make_two_moons_loaders, standard_normal_logprob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SECTION 1 – CNF model definition
# ============================================================

class ODEFunc(nn.Module):
    """Neural vector field  f_theta(t, x)."""

    def __init__(self, dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_col = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_col], dim=1))


def hutchinson_trace(dxdt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Hutchinson estimator of trace(df/dx) using Rademacher noise."""
    eps = (torch.randint(0, 2, x.shape, device=x.device).float() * 2.0 - 1.0)
    jtv = torch.autograd.grad(
        dxdt, x, grad_outputs=eps,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return (jtv * eps).sum(dim=1)


class CNFBlock(nn.Module):
    """Augmented ODE: d/dt [x, log p] = [f, -tr(df/dx)]."""

    def __init__(self, odefunc: ODEFunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, t, states):
        x, logp = states
        x = x.requires_grad_(True)
        dxdt = self.odefunc(t, x)
        return dxdt, -hutchinson_trace(dxdt, x)


class CNFModel(nn.Module):
    """Continuous normalising flow with adjoint-based training."""

    def __init__(self, dim: int = 2, hidden_dim: int = 32,
                 t0: float = 0.0, t1: float = 1.0):
        super().__init__()
        self.t0, self.t1 = t0, t1
        self.dim = dim
        self.block = CNFBlock(ODEFunc(dim=dim, hidden_dim=hidden_dim))

    def _integrate(self, x0, logp0, t_start, t_end):
        ts = torch.tensor([t_start, t_end], device=x0.device)
        z, lp = odeint(
            self.block, (x0, logp0), ts,
            rtol=1e-4, atol=1e-4, method="dopri5",
            adjoint_params=tuple(self.block.parameters()),
        )
        return z[-1], lp[-1]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        logp0 = torch.zeros(x.shape[0], device=x.device)
        z, delta_logp = self._integrate(x, logp0, self.t1, self.t0)
        return standard_normal_logprob(z) + delta_logp   # Δlogp sign from augmented ODE

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.dim, device=device)
        lp = torch.zeros(n, device=device)
        x, _ = self._integrate(z, lp, self.t0, self.t1)
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map data to latent space (for diagnostics)."""
        with torch.enable_grad():
            x = x.requires_grad_(True)
            lp = torch.zeros(x.shape[0], device=x.device)
            z, _ = self._integrate(x, lp, self.t1, self.t0)
        return z.detach()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train(model: CNFModel, train_loader, n_epochs: int = 100, lr: float = 1e-3,
          print_every: int = 10) -> list[float]:
    """Train model via MLE; return per-epoch mean training NLL."""
    opt = optim.Adam(model.parameters(), lr=lr)
    nll_history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = -model.log_prob(xb).mean()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        nll_history.append(total / count)
        if epoch % print_every == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train NLL {nll_history[-1]:.4f}")
    return nll_history


@torch.no_grad()
def mean_nll(model: CNFModel, loader) -> float:
    """Mean negative log-likelihood on a DataLoader."""
    model.eval()
    total, count = 0.0, 0
    with torch.enable_grad():          # needed for Hutchinson trace inside log_prob
        for (xb,) in loader:
            xb = xb.to(device)
            total += (-model.log_prob(xb)).sum().item()
            count += xb.shape[0]
    return total / count


# ============================================================
# SECTION 2 – Fit nested CNFs and compare NLL
# ============================================================

def section2(train_loader, test_loader, epochs: int = 100):
    print("\n" + "="*60)
    print("SECTION 2 – Nested CNF comparison")
    print("="*60)

    results = {}

    for name, hdim in [("small (H=16)", 16), ("large (H=64)", 64)]:
        print(f"\nTraining {name} model  ({epochs} epochs)...")
        model = CNFModel(dim=2, hidden_dim=hdim).to(device)
        train(model, train_loader, n_epochs=epochs)

        tr_nll = mean_nll(model, train_loader)
        te_nll = mean_nll(model, test_loader)
        print(f"  → Train NLL: {tr_nll:.4f}   Test NLL: {te_nll:.4f}   Params: {model.n_params()}")

        results[name] = {
            "model":    model,
            "hdim":     hdim,
            "train_nll": tr_nll,
            "test_nll":  te_nll,
            "n_params":  model.n_params(),
        }

    return results


# ============================================================
# SECTION 3 – Likelihood ratio test (Wilks' theorem)
# ============================================================

def section3(results: dict, n_test: int):
    """
    Λ = 2(ℓ_1 − ℓ_0)  where  ℓ = −n · mean_NLL.
    Under H0, Λ ~ chi²(df)  with  df = n_params_large − n_params_small.
    """
    print("\n" + "="*60)
    print("SECTION 3 – Likelihood ratio test")
    print("="*60)

    r0 = results["small (H=16)"]
    r1 = results["large (H=64)"]

    # Total log-likelihoods on the test set.
    ell0 = -r0["test_nll"] * n_test
    ell1 = -r1["test_nll"] * n_test

    Lambda = 2.0 * (ell1 - ell0)
    df     = r1["n_params"] - r0["n_params"]
    p_val  = 1.0 - stats.chi2.cdf(Lambda, df=df)

    print(f"  ℓ_small  = {ell0:.2f}  (mean NLL {r0['test_nll']:.4f})")
    print(f"  ℓ_large  = {ell1:.2f}  (mean NLL {r1['test_nll']:.4f})")
    print(f"  Λ        = {Lambda:.2f}")
    print(f"  df       = {df}  (parameter difference)")
    print(f"  p-value  = {p_val:.4e}")
    print()
    if p_val < 0.05:
        print("  → Reject H0 at 5% level: the larger model fits significantly better.")
    else:
        print("  → Fail to reject H0: no significant improvement from extra capacity.")

    return Lambda, df, p_val


# ============================================================
# SECTION 4 – Q–Q plots of latent coordinates
# ============================================================

def section4(results: dict, x_test: torch.Tensor):
    print("\n" + "="*60)
    print("SECTION 4 – Q–Q plots of latent coordinates")
    print("="*60)

    for name, r in results.items():
        model = r["model"].eval()
        z = model.encode(x_test.to(device)).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.suptitle(f"Q–Q plots  –  {name}", fontsize=12)

        for ax, (coord, label) in zip(axes, enumerate(["z₁", "z₂"])):
            stats.probplot(z[:, coord], dist="norm", plot=ax)
            ax.set_title(f"Latent coordinate {label}")
            ax.get_lines()[0].set(markersize=2, alpha=0.5)

        plt.tight_layout()
        fname = f"section4_qq_{name.split()[0]}.png"
        plt.savefig(fname, dpi=150)
        print(f"  Saved {fname}")
        plt.show()


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    train_loader, test_loader, x_train, x_test = make_two_moons_loaders(
        n_samples=2000, batch_size=256
    )

    # Section 2: train both models.
    results = section2(train_loader, test_loader, epochs=100)

    # Section 3: LRT.
    section3(results, n_test=len(x_test))

    # Section 4: Q–Q plots.
    section4(results, x_test)

    # Bonus: quick visual of samples from the large model.
    large_model = results["large (H=64)"]["model"].eval()
    samples = large_model.sample(800).cpu()
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=5, alpha=0.6)
    plt.title("Training data")
    plt.axis("equal")
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=5, alpha=0.6, color="coral")
    plt.title("Large CNF samples")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("section2_samples.png", dpi=150)
    plt.show()

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import math


def make_two_moons_loaders(
    n_samples: int = 2000,
    batch_size: int = 256,
    test_split: float = 0.2,
    noise: float = 0.05,
    seed: int = 42,
):
    """Return train/test DataLoaders and raw tensors for the two-moons dataset."""
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # Standardise to zero mean, unit variance.
    X = (X - X.mean(0)) / X.std(0)

    n_test = int(n_samples * test_split)
    x_train = torch.tensor(X[n_test:], dtype=torch.float32)
    x_test  = torch.tensor(X[:n_test],  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(TensorDataset(x_test),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, x_train, x_test


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Log-density of N(0, I) evaluated at each row of z.  Shape: (batch,)"""
    d = z.shape[1]
    return -0.5 * (d * math.log(2.0 * math.pi) + (z ** 2).sum(dim=1))

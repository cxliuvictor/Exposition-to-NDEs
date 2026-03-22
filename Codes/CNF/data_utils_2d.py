import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import math


def make_two_moons_loaders(
    n_samples: int = 2000,
    batch_size: int = 256,
    val_split: float = 0.15,
    test_split: float = 0.15,
    noise: float = 0.05,
    seed: int = 42,
):
    """
    Return train/val/test DataLoaders and raw tensors for the two-moons dataset.

    Default split: 70% train, 15% val (early stopping), 15% test (LRT).
    Keeping val and test separate avoids the circularity of stopping on test NLL
    then reporting the same test NLL for the likelihood ratio test.
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # Standardise to zero mean, unit variance.
    X = (X - X.mean(0)) / X.std(0)

    n_test = int(n_samples * test_split)
    n_val  = int(n_samples * val_split)

    x_test  = torch.tensor(X[:n_test],              dtype=torch.float32)
    x_val   = torch.tensor(X[n_test:n_test + n_val], dtype=torch.float32)
    x_train = torch.tensor(X[n_test + n_val:],       dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(x_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(x_test),  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_train, x_val, x_test


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Log-density of N(0, I) evaluated at each row of z.  Shape: (batch,)"""
    d = z.shape[1]
    return -0.5 * (d * math.log(2.0 * math.pi) + (z ** 2).sum(dim=1))
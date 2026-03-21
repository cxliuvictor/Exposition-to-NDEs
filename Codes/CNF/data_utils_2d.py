# Import NumPy for numerical array work.
import numpy as np

# Import PyTorch.
import torch

# Import TensorDataset and DataLoader for batching data.
from torch.utils.data import TensorDataset, DataLoader

# Import a simple synthetic dataset generator.
from sklearn.datasets import make_moons

# Import a train/test split helper.
from sklearn.model_selection import train_test_split


def make_two_moons_loaders(
    n_samples: int = 4000,
    noise: float = 0.08,
    test_size: float = 0.25,
    batch_size: int = 256,
    random_state: int = 0,
):
    """
    Generate a simple 2D two-moons dataset, standardise it,
    and return PyTorch DataLoaders for training and testing.

    Parameters
    ----------
    n_samples : int
        Total number of data points to generate.
    noise : float
        Noise level used by sklearn's make_moons.
    test_size : float
        Fraction of data reserved for testing.
    batch_size : int
        Batch size used by the DataLoaders.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_loader : DataLoader
        Training set loader.
    test_loader : DataLoader
        Test set loader.
    x_train_tensor : torch.Tensor
        Full training tensor of shape [n_train, 2].
    x_test_tensor : torch.Tensor
        Full test tensor of shape [n_test, 2].
    """
    # Create the two-moons dataset in NumPy form.
    x, _ = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    # Split the data into train and test parts.
    x_train, x_test = train_test_split(
        x,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Compute the training mean for standardisation.
    train_mean = x_train.mean(axis=0, keepdims=True)

    # Compute the training standard deviation for standardisation.
    train_std = x_train.std(axis=0, keepdims=True)

    # Add a tiny epsilon so division is always safe.
    train_std = train_std + 1e-8

    # Standardise the training data using training statistics.
    x_train = (x_train - train_mean) / train_std

    # Standardise the test data using the same training statistics.
    x_test = (x_test - train_mean) / train_std

    # Convert the training array to a float32 PyTorch tensor.
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

    # Convert the test array to a float32 PyTorch tensor.
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Wrap the training tensor in a TensorDataset.
    train_dataset = TensorDataset(x_train_tensor)

    # Wrap the test tensor in a TensorDataset.
    test_dataset = TensorDataset(x_test_tensor)

    # Build a training DataLoader with shuffling.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Build a test DataLoader without shuffling.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Return everything needed downstream.
    return train_loader, test_loader, x_train_tensor, x_test_tensor


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-density of each row of z under N(0, I).

    Parameters
    ----------
    z : torch.Tensor
        Tensor of shape [batch, dim].

    Returns
    -------
    torch.Tensor
        Log-density for each sample, shape [batch].
    """
    # Get the latent dimension.
    dim = z.shape[1]

    # Compute the constant part of the Gaussian log-density.
    log_constant = -0.5 * dim * np.log(2.0 * np.pi)

    # Compute the quadratic term for each row.
    quadratic_term = -0.5 * torch.sum(z ** 2, dim=1)

    # Return the row-wise log-density.
    return log_constant + quadratic_term

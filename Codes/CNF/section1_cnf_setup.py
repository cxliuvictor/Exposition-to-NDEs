# Import PyTorch.
import torch

# Import neural-network tools from PyTorch.
import torch.nn as nn

# Import the Adam optimiser.
import torch.optim as optim

# Import the adjoint ODE solver from torchdiffeq.
from torchdiffeq import odeint_adjoint as odeint

# Import matplotlib for plots.
import matplotlib.pyplot as plt

# Import the shared data utilities.
from data_utils_2d import make_two_moons_loaders, standard_normal_logprob


# Choose GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODEFunc(nn.Module):
    """
    Neural vector field f_theta(t, x) for the CNF dynamics.
    """

    def __init__(self, dim: int = 2, hidden_dim: int = 16):
        # Initialise the parent nn.Module.
        super().__init__()

        # Store the state dimension.
        self.dim = dim

        # Build a small MLP taking [x, t] as input and returning dx/dt.
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the vector field at time t and state x.
        """
        # Create a column filled with the current scalar time.
        t_column = torch.ones(x.shape[0], 1, device=x.device) * t

        # Concatenate state and time.
        tx = torch.cat([x, t_column], dim=1)

        # Return the neural vector field output.
        return self.net(tx)


def hutchinson_trace(dxdt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Estimate trace(df/dx) using Hutchinson's estimator with Rademacher noise.
    """
    # Draw Rademacher noise with entries in {-1, +1}.
    eps = torch.randint(low=0, high=2, size=x.shape, device=x.device).float() * 2.0 - 1.0

    # Compute J^T eps via reverse-mode autodiff.
    j_transpose_eps = torch.autograd.grad(
        outputs=dxdt,
        inputs=x,
        grad_outputs=eps,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute eps^T J eps row by row.
    trace_estimate = torch.sum(j_transpose_eps * eps, dim=1)

    # Return one trace estimate per sample.
    return trace_estimate


class CNFBlock(nn.Module):
    """
    Augmented ODE system for (x, logp).
    """

    def __init__(self, odefunc: ODEFunc):
        # Initialise the parent class.
        super().__init__()

        # Store the vector field module.
        self.odefunc = odefunc

    def forward(self, t: torch.Tensor, states):
        """
        Return derivatives for the augmented system:
            dx/dt = f(t, x)
            d log p / dt = - trace(df/dx)
        """
        # Unpack the augmented state.
        x, logp = states

        # Ensure x tracks gradients because we need df/dx.
        x = x.requires_grad_(True)

        # Evaluate the vector field.
        dxdt = self.odefunc(t, x)

        # Estimate the Jacobian trace.
        trace_estimate = hutchinson_trace(dxdt, x)

        # Apply the instantaneous change-of-variables formula.
        dlogpdt = -trace_estimate

        # Return the two derivatives.
        return dxdt, dlogpdt


class CNFModel(nn.Module):
    """
    Simple continuous normalising flow model.
    """

    def __init__(self, dim: int = 2, hidden_dim: int = 16, t0: float = 0.0, t1: float = 1.0):
        # Initialise the parent class.
        super().__init__()

        # Store the latent/data dimension.
        self.dim = dim

        # Store the start time.
        self.t0 = t0

        # Store the end time.
        self.t1 = t1

        # Build the vector field.
        odefunc = ODEFunc(dim=dim, hidden_dim=hidden_dim)

        # Wrap it in the augmented CNF block.
        self.block = CNFBlock(odefunc)

    def forward_transform(self, z0: torch.Tensor, logp0: torch.Tensor):
        """
        Push latent samples forwards from t0 to t1.
        """
        # Create the two integration times.
        integration_times = torch.tensor([self.t0, self.t1], device=z0.device)

        # Solve the augmented ODE forwards.
        z_traj, logp_traj = odeint(
            self.block,
            (z0, logp0),
            integration_times,
            rtol=1e-4,
            atol=1e-4,
            method="dopri5",
        )

        # Return the final state at time t1.
        return z_traj[-1], logp_traj[-1]

    def inverse_transform(self, x1: torch.Tensor, logp1: torch.Tensor):
        """
        Map observed data backwards from t1 to t0.
        """
        # Create the reversed integration times.
        integration_times = torch.tensor([self.t1, self.t0], device=x1.device)

        # Solve the augmented ODE backwards.
        z_traj, logp_traj = odeint(
            self.block,
            (x1, logp1),
            integration_times,
            rtol=1e-4,
            atol=1e-4,
            method="dopri5",
        )

        # Return the latent state at time t0.
        return z_traj[-1], logp_traj[-1]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_theta(x) using the backward solve and base Gaussian density.
        """
        # Get the batch size.
        batch_size = x.shape[0]

        # Start with zero log-density correction at the data end.
        logp_x = torch.zeros(batch_size, device=x.device)

        # Backwards-map the data to the base space.
        z0, logp0 = self.inverse_transform(x, logp_x)

        # Evaluate the base Gaussian log-density.
        base_logp = standard_normal_logprob(z0)

        # Combine base density and volume-change term.
        return base_logp - logp0

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples by pushing Gaussian latent points forwards.
        """
        # Turn off gradient tracking for sampling.
        with torch.no_grad():
            # Draw latent samples from N(0, I).
            z0 = torch.randn(n_samples, self.dim, device=device)

            # Start the log-density accumulator at zero.
            logp0 = torch.zeros(n_samples, device=device)

            # Push them forward through the flow.
            x1, _ = self.forward_transform(z0, logp0)

        # Return the generated samples.
        return x1


def mean_nll(model: CNFModel, data_loader) -> float:
    """
    Compute the mean negative log-likelihood on a dataset.
    """
    # Put the model in evaluation mode.
    model.eval()

    # Initialise the running total of negative log-likelihood.
    total_nll = 0.0

    # Initialise the running sample count.
    total_count = 0

    # We still need gradients inside log_prob because the trace uses autodiff.
    with torch.enable_grad():
        # Loop over all batches.
        for (x_batch,) in data_loader:
            # Move the batch to the active device.
            x_batch = x_batch.to(device)

            # Compute log-likelihoods.
            logp = model.log_prob(x_batch)

            # Accumulate total negative log-likelihood.
            total_nll += (-logp).sum().item()

            # Accumulate batch size.
            total_count += x_batch.shape[0]

    # Return the dataset average.
    return total_nll / total_count


def train_cnf(
    hidden_dim: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    model_path: str = "cnf_section1.pt",
):
    """
    Train a single CNF on the 2D two-moons dataset.
    """
    # Load training and test data.
    train_loader, test_loader, x_train_tensor, x_test_tensor = make_two_moons_loaders(
        batch_size=batch_size
    )

    # Create the CNF model.
    model = CNFModel(dim=2, hidden_dim=hidden_dim).to(device)

    # Create the Adam optimiser.
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # Start the training loop.
    for epoch in range(1, epochs + 1):
        # Put the model in training mode.
        model.train()

        # Loop over mini-batches.
        for (x_batch,) in train_loader:
            # Move the batch to the correct device.
            x_batch = x_batch.to(device)

            # Clear the previous gradients.
            optimiser.zero_grad()

            # Compute batch log-likelihoods.
            logp = model.log_prob(x_batch)

            # Convert to the loss used for MLE training.
            loss = -logp.mean()

            # Differentiate the loss.
            loss.backward()

            # Update parameters.
            optimiser.step()

        # Print progress every 10 epochs.
        if epoch % 10 == 0 or epoch == 1:
            # Compute training NLL.
            train_nll = mean_nll(model, train_loader)

            # Compute test NLL.
            test_nll = mean_nll(model, test_loader)

            # Print both metrics.
            print(f"Epoch {epoch:03d} | Train NLL: {train_nll:.4f} | Test NLL: {test_nll:.4f}")

    # Save the trained weights.
    torch.save(model.state_dict(), model_path)

    # Put the model in evaluation mode.
    model.eval()

    # Draw generated samples from the fitted model.
    generated = model.sample(1000).cpu()

    # Plot the real training data and generated samples side by side.
    plt.figure(figsize=(10, 4))

    # Plot the real data.
    plt.subplot(1, 2, 1)
    plt.scatter(x_train_tensor[:, 0].numpy(), x_train_tensor[:, 1].numpy(), s=6)
    plt.title("Training data")
    plt.axis("equal")

    # Plot the generated data.
    plt.subplot(1, 2, 2)
    plt.scatter(generated[:, 0].numpy(), generated[:, 1].numpy(), s=6)
    plt.title("CNF samples")
    plt.axis("equal")

    # Make spacing tidy.
    plt.tight_layout()

    # Save the figure.
    plt.savefig("section1_cnf_samples.png", dpi=200)

    # Show the figure.
    plt.show()

    # Return the trained model.
    return model


if __name__ == "__main__":
    # Train the Section 1 model with a small hidden width.
    train_cnf(hidden_dim=16, epochs=100, lr=1e-3, model_path="cnf_section1.pt")

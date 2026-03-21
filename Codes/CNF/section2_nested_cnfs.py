# Import PyTorch.
import torch

# Import the Adam optimiser.
import torch.optim as optim

# Import matplotlib for plots.
import matplotlib.pyplot as plt

# Import shared data utilities.
from data_utils_2d import make_two_moons_loaders

# Import the CNF model and helper from Section 1.
from section1_cnf_setup import CNFModel, mean_nll, device


def train_one_model(
    hidden_dim: int,
    train_loader,
    test_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: str = "cnf_model.pt",
):
    """
    Train one CNF model and return its final train/test NLLs.
    """
    # Create the model with the chosen hidden width.
    model = CNFModel(dim=2, hidden_dim=hidden_dim).to(device)

    # Create the optimiser.
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # Loop over epochs.
    for epoch in range(1, epochs + 1):
        # Put the model in training mode.
        model.train()

        # Loop over mini-batches.
        for (x_batch,) in train_loader:
            # Move the batch to the active device.
            x_batch = x_batch.to(device)

            # Reset the gradients.
            optimiser.zero_grad()

            # Compute the batch log-likelihood.
            logp = model.log_prob(x_batch)

            # Turn it into the loss.
            loss = -logp.mean()

            # Backpropagate.
            loss.backward()

            # Update the parameters.
            optimiser.step()

    # Compute final training NLL.
    train_nll = mean_nll(model, train_loader)

    # Compute final test NLL.
    test_nll = mean_nll(model, test_loader)

    # Save the model weights.
    torch.save(model.state_dict(), save_path)

    # Return the trained model and its metrics.
    return model, train_nll, test_nll


def run_section2():
    """
    Fit two nested-style CNFs: a small one and a large one.
    """
    # Load the dataset.
    train_loader, test_loader, x_train_tensor, x_test_tensor = make_two_moons_loaders(batch_size=256)

    # Train the smaller model.
    small_model, small_train_nll, small_test_nll = train_one_model(
        hidden_dim=16,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        lr=1e-3,
        save_path="cnf_small.pt",
    )

    # Train the larger model.
    large_model, large_train_nll, large_test_nll = train_one_model(
        hidden_dim=32,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        lr=1e-3,
        save_path="cnf_large.pt",
    )

    # Print a concise comparison table.
    print("\nSection 2 results")
    print("-" * 50)
    print(f"CNF-small | hidden_dim = 16 | Train NLL = {small_train_nll:.4f} | Test NLL = {small_test_nll:.4f}")
    print(f"CNF-large | hidden_dim = 32 | Train NLL = {large_train_nll:.4f} | Test NLL = {large_test_nll:.4f}")

    # Generate samples from the small model.
    small_samples = small_model.sample(1000).cpu()

    # Generate samples from the large model.
    large_samples = large_model.sample(1000).cpu()

    # Plot the two generated sample clouds.
    plt.figure(figsize=(10, 4))

    # Plot samples from the small model.
    plt.subplot(1, 2, 1)
    plt.scatter(small_samples[:, 0].numpy(), small_samples[:, 1].numpy(), s=6)
    plt.title("CNF-small samples")
    plt.axis("equal")

    # Plot samples from the large model.
    plt.subplot(1, 2, 2)
    plt.scatter(large_samples[:, 0].numpy(), large_samples[:, 1].numpy(), s=6)
    plt.title("CNF-large samples")
    plt.axis("equal")

    # Make spacing tidy.
    plt.tight_layout()

    # Save the figure.
    plt.savefig("section2_nested_cnf_samples.png", dpi=200)

    # Show the figure.
    plt.show()


if __name__ == "__main__":
    # Run Section 2.
    run_section2()

# Import PyTorch.
import torch

# Import the chi-squared distribution for the Wilks approximation.
from scipy.stats import chi2

# Import the dataset loader.
from data_utils_2d import make_two_moons_loaders

# Import the CNF model class and device.
from section1_cnf_setup import CNFModel, device


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    # Sum the sizes of all trainable tensors.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_log_likelihood(model, data_loader):
    """
    Compute the total log-likelihood on a dataset.
    """
    # Put the model in evaluation mode.
    model.eval()

    # Start the running total of log-likelihood.
    total_ll = 0.0

    # Start the running sample count.
    total_n = 0

    # We still need gradients because log_prob uses Jacobian traces.
    with torch.enable_grad():
        # Loop over all test batches.
        for (x_batch,) in data_loader:
            # Move data to the active device.
            x_batch = x_batch.to(device)

            # Compute batch log-likelihoods.
            logp = model.log_prob(x_batch)

            # Add the batch total to the running sum.
            total_ll += logp.sum().item()

            # Add the batch size.
            total_n += x_batch.shape[0]

    # Return the full total and the sample count.
    return total_ll, total_n


def run_wilks_test():
    """
    Load the two trained CNFs and perform a Wilks-style likelihood ratio test.
    """
    # Recreate the same data split used in Section 2.
    train_loader, test_loader, x_train_tensor, x_test_tensor = make_two_moons_loaders(batch_size=256)

    # Create the small model architecture.
    small_model = CNFModel(dim=2, hidden_dim=16).to(device)

    # Load the trained small-model weights.
    small_model.load_state_dict(torch.load("cnf_small.pt", map_location=device))

    # Create the large model architecture.
    large_model = CNFModel(dim=2, hidden_dim=32).to(device)

    # Load the trained large-model weights.
    large_model.load_state_dict(torch.load("cnf_large.pt", map_location=device))

    # Compute total test log-likelihood for the small model.
    ll_small, n_test = total_log_likelihood(small_model, test_loader)

    # Compute total test log-likelihood for the large model.
    ll_large, _ = total_log_likelihood(large_model, test_loader)

    # Form the likelihood-ratio statistic.
    Lambda = 2.0 * (ll_large - ll_small)

    # Use the difference in trainable parameter counts as the degrees of freedom.
    df = count_parameters(large_model) - count_parameters(small_model)

    # Compute the chi-squared upper-tail p-value.
    p_value = 1.0 - chi2.cdf(Lambda, df=df)

    # Print the results cleanly.
    print("\nSection 3: Wilks-style likelihood ratio test")
    print("-" * 60)
    print(f"Small model total test log-likelihood : {ll_small:.4f}")
    print(f"Large model total test log-likelihood : {ll_large:.4f}")
    print(f"Lambda = 2(ell_1 - ell_0)            : {Lambda:.4f}")
    print(f"Degrees of freedom                   : {df}")
    print(f"Approximate p-value                  : {p_value:.6g}")

    # Print a plain-English conclusion.
    if p_value < 0.05:
        print("Conclusion: reject H0 at the 5% level; the larger CNF appears to improve fit.")
    else:
        print("Conclusion: do not reject H0 at the 5% level; the smaller CNF may already be sufficient.")


if __name__ == "__main__":
    # Run the Wilks-style test.
    run_wilks_test()

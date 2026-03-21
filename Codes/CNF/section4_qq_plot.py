# Import PyTorch.
import torch

# Import matplotlib for plotting.
import matplotlib.pyplot as plt

# Import scipy's probability plot helper.
from scipy import stats

# Import the dataset loader.
from data_utils_2d import make_two_moons_loaders

# Import the CNF model class and device.
from section1_cnf_setup import CNFModel, device


def extract_latent_codes(model, data_loader):
    """
    Map observed data backwards into the latent Gaussian space.
    """
    # Put the model into evaluation mode.
    model.eval()

    # Create a list to hold latent batches.
    latent_batches = []

    # We still need gradients because inverse_transform uses trace estimation.
    with torch.enable_grad():
        # Loop through all batches.
        for (x_batch,) in data_loader:
            # Move the batch to the active device.
            x_batch = x_batch.to(device)

            # Start the end-time log-density accumulator at zero.
            logp_x = torch.zeros(x_batch.shape[0], device=device)

            # Map data points backwards to latent space.
            z0, _ = model.inverse_transform(x_batch, logp_x)

            # Store the batch on CPU for later plotting.
            latent_batches.append(z0.detach().cpu())

    # Concatenate all latent batches into one tensor.
    return torch.cat(latent_batches, dim=0)


def qq_plot_one_coordinate(z: torch.Tensor, coord: int = 0, save_path: str = "section4_qq.png"):
    """
    Create a Q-Q plot comparing one latent coordinate against N(0,1).
    """
    # Extract the chosen coordinate as a NumPy array.
    values = z[:, coord].numpy()

    # Create a square figure.
    plt.figure(figsize=(5, 5))

    # Build the Q-Q plot against the standard normal.
    stats.probplot(values, dist="norm", plot=plt)

    # Add a title.
    plt.title(f"Q-Q plot for latent coordinate z[:, {coord}]")

    # Make spacing tidy.
    plt.tight_layout()

    # Save the plot to file.
    plt.savefig(save_path, dpi=200)

    # Show the plot.
    plt.show()


def run_section4():
    """
    Load the larger CNF and create Q-Q plots for one or two latent coordinates.
    """
    # Recreate the same data split used previously.
    train_loader, test_loader, x_train_tensor, x_test_tensor = make_two_moons_loaders(batch_size=256)

    # Create the larger model architecture.
    model = CNFModel(dim=2, hidden_dim=32).to(device)

    # Load the trained weights for the larger model.
    model.load_state_dict(torch.load("cnf_large.pt", map_location=device))

    # Extract latent codes for the test set.
    z_test = extract_latent_codes(model, test_loader)

    # Make a Q-Q plot for the first latent coordinate.
    qq_plot_one_coordinate(z_test, coord=0, save_path="section4_qq_coord0.png")

    # Make a Q-Q plot for the second latent coordinate as well.
    qq_plot_one_coordinate(z_test, coord=1, save_path="section4_qq_coord1.png")


if __name__ == "__main__":
    # Run Section 4.
    run_section4()

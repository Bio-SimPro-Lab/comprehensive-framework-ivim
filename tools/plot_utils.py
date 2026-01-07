import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gmm(pi, mu, sigma, ground_truth, saving_path, x_range=None, num_points=1000):
    """
    Plots a 1D Gaussian Mixture Model with specified parameters.

    Parameters:
    - pi: list or array-like of mixing coefficients (length K)
    - mu: list or array-like of means (length K)
    - sigma: list or array-like of standard deviations (length K)
    - x_range: tuple (x_min, x_max) specifying the range of x values; if None, it's computed automatically
    - num_points: number of points to compute in the x-axis
    - ground_truth: list or array-like of ground truth x-values to mark with vertical dashed lines
    """
    pi = np.array(pi)
    mu = np.array(mu)
    sigma = np.array(sigma)
    K = len(pi)

    # Validate inputs
    if not (len(mu) == len(sigma) == K):
        raise ValueError("The lengths of pi, mu, and sigma must be equal.")

    # Determine x-axis range if not provided
    if x_range is None:
        x_min = min(mu - 4 * sigma)
        x_max = max(mu + 4 * sigma)
    else:
        x_min, x_max = x_range

    x = np.linspace(x_min, x_max, num_points)

    # Compute individual Gaussian components
    components = [pi[k] * norm.pdf(x, mu[k], sigma[k]) for k in range(K)]

    # Compute the GMM PDF as the sum of components
    gmm_pdf = np.sum(components, axis=0)

    # Plot the GMM and its components
    plt.figure(figsize=(8, 5))
    plt.plot(x, gmm_pdf, label='GMM', color='black', linewidth=2)
    for k, comp in enumerate(components):
        plt.plot(x, comp, label=f'Component {k + 1}', linestyle='--')

    # Plot ground truth lines if provided
    if ground_truth is not None:
        for gt in np.atleast_1d(ground_truth):
            plt.axvline(gt, color='red', linestyle='dashed', linewidth=1.5, label='Ground Truth')

    plt.title('Gaussian Mixture Model')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.grid(True)

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.savefig(saving_path)

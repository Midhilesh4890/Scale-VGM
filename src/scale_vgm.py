import logging
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ScaleVGM:
    """
    Implements Variational Gaussian Mixture (VGM) with Mode-Specific Normalization
    as per the CTGAN paper for scalable datasets.
    """

    def __init__(self, n_components=10, random_state=42, eps=0.005):
        """
        Initialize the ScaleVGM class.

        Args:
            n_components (int): Number of BGMM components (modes).
            random_state (int): Random seed for reproducibility.
            eps (float): Minimum weight threshold for valid modes.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.eps = eps
        self.bgmm = None  # Placeholder for the Bayesian Gaussian Mixture Model

    def fit(self, data):
        """
        Fit a Bayesian Gaussian Mixture Model (BGMM) to the data.

        Args:
            data (numpy.ndarray): 1D array of continuous data (e.g., `Amount` column).
        """
        logging.info("Fitting Bayesian Gaussian Mixture Model (BGMM).")
        self.bgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=100,
            random_state=self.random_state,
        )
        self.bgmm.fit(data.reshape(-1, 1))
        logging.info("BGMM fitting completed.")

    def transform(self, data):
        """
        Apply Mode-Specific Normalization to the data.

        Args:
            data (numpy.ndarray): 1D array of continuous data.

        Returns:
            tuple: (normalized_data, mode_assignments)
        """
        if self.bgmm is None:
            raise ValueError(
                "BGMM model has not been fitted. Call 'fit' first.")

        logging.info("Applying Mode-Specific Normalization.")
        means = self.bgmm.means_.flatten()
        stds = np.sqrt(self.bgmm.covariances_.flatten())
        modes = self.bgmm.predict(data.reshape(-1, 1))  # Cluster assignments

        normalized_data = np.zeros_like(data)
        for mode in np.unique(modes):
            indices = modes == mode
            normalized_data[indices] = (
                data[indices] - means[mode]) / (4 * stds[mode])
            normalized_data[indices] = np.clip(
                normalized_data[indices], -0.99, 0.99)

        return normalized_data, modes

    def inverse_transform(self, normalized_data, modes):
        """
        Revert normalized data back to its original scale.

        Args:
            normalized_data (numpy.ndarray): Normalized data.
            modes (numpy.ndarray): Mode assignments for each data point.

        Returns:
            numpy.ndarray: Reconstructed original data.
        """
        if self.bgmm is None:
            raise ValueError(
                "BGMM model has not been fitted. Call 'fit' first.")

        logging.info("Reverting normalized data to original scale.")
        means = self.bgmm.means_.flatten()
        stds = np.sqrt(self.bgmm.covariances_.flatten())

        original_data = np.zeros_like(normalized_data)
        for mode in np.unique(modes):
            indices = modes == mode
            original_data[indices] = normalized_data[indices] * \
                (4 * stds[mode]) + means[mode]

        return original_data

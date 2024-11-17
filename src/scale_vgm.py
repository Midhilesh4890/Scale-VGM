import logging
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class ScaleVGM:
    """
    A class to apply Bayesian Gaussian Mixture Model (BGMM) scaling.
    It normalizes data based on the clusters detected by BGMM.

    Attributes:
        n_components (int): Number of mixture components for BGMM.
        random_state (int): Random seed for reproducibility.
        eps (float): Small value to avoid division by zero.
        bgmm (BayesianGaussianMixture): Instance of the BGMM model.

    Methods:
        fit(data): Fits the BGMM model to the provided data.
        transform(data): Transforms and normalizes the data using the fitted BGMM.
    """

    def __init__(self, n_components=10, random_state=42, eps=0.005):
        """
        Initializes the ScaleVGM class with specified parameters.

        Args:
            n_components (int): Number of mixture components for BGMM.
            random_state (int): Random seed for reproducibility.
            eps (float): Small value to avoid division by zero.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.eps = eps
        self.bgmm = None  # Placeholder for the Bayesian Gaussian Mixture Model

    def fit(self, data):
        """
        Fits the BGMM model to the provided data.

        Args:
            data (numpy.ndarray): 1D array of data to fit the model.
        
        Raises:
            ValueError: If the data is not a 1D numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")

        logging.info("Fitting Bayesian Gaussian Mixture Model (BGMM).")
        self.bgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=200,
            random_state=self.random_state,
        )
        # Fit the BGMM model to the input data
        self.bgmm.fit(data.reshape(-1, 1))
        logging.info("BGMM fitting completed.")

    def transform(self, data):
        """
        Normalizes the data using the fitted BGMM model.

        Args:
            data (numpy.ndarray): 1D array of data to be normalized.

        Returns:
            numpy.ndarray: Normalized data.
            numpy.ndarray: Cluster modes for each data point.
        
        Raises:
            ValueError: If the BGMM model has not been fitted.
        """
        if self.bgmm is None:
            raise ValueError(
                "BGMM model has not been fitted. Call 'fit' first.")

        logging.info("Transforming data using the BGMM model.")
        # Extract cluster-specific means and standard deviations
        means = self.bgmm.means_.flatten()
        stds = np.sqrt(self.bgmm.covariances_.flatten())

        # Predict the cluster assignment for each data point
        modes = self.bgmm.predict(data.reshape(-1, 1))

        # Initialize an array for normalized data
        normalized_data = np.zeros_like(data)

        # Normalize each cluster's data
        for mode in np.unique(modes):
            indices = modes == mode
            normalized_data[indices] = (
                data[indices] - means[mode]) / (4 * stds[mode])
            # Clip the normalized values to avoid extreme outliers
            normalized_data[indices] = np.clip(
                normalized_data[indices], -0.99, 0.99)

        logging.info("Data transformation completed.")
        return normalized_data, modes

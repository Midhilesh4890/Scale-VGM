import logging
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import Client
import dask.dataframe as dd

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


class DataProcessor:
    """
    Handles scalable data processing for large datasets using Dask.
    """

    def __init__(self, input_path, output_path, transformer):
        """
        Initialize the DataProcessor.

        Args:
            input_path (str): Path to the input Parquet files (1 billion rows dataset).
            output_path (str): Path to save the transformed dataset.
            transformer (ScaleVGM): Instance of the ScaleVGM class.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.transformer = transformer

    def process_data(self):
        """
        Process the dataset in chunks, transform it using VGM, and save the output.
        """
        logging.info("Loading dataset using Dask.")
        # Load Parquet files as Dask DataFrame
        ddf = dd.read_parquet(self.input_path)

        # Apply transformation to all partitions
        logging.info("Applying VGM transformation to all data partitions.")
        transformed_ddf = ddf.map_partitions(self._transform_partition)

        # Save the transformed dataset back to Parquet
        logging.info(f"Saving transformed dataset to {self.output_path}.")
        transformed_ddf.to_parquet(
            self.output_path, engine="pyarrow", write_index=False)
        logging.info("Transformation completed and data saved.")

    def _transform_partition(self, partition):
        """
        Transform a single partition using the ScaleVGM transformer.

        Args:
            partition (pandas.DataFrame): A single partition of the dataset.

        Returns:
            pandas.DataFrame: Transformed partition with normalized data.
        """
        logging.info("Transforming a data partition.")

        # Apply the ScaleVGM transform to the "Amount" column
        normalized_data, modes = self.transformer.transform(
            partition["Amount"].values)

        # Add the normalized column to the partition
        partition["Amount_Transformed"] = normalized_data
        # Optional: Add mode assignments for debugging
        partition["Mode_Assignment"] = modes

        return partition


if __name__ == "__main__":
    # Configure Dask for multiprocessing
    client = Client(processes=True, n_workers=4, threads_per_worker=1)
    print(client.dashboard_link)

    # Paths
    INPUT_PATH = "data/scaled_data_1b/*.parquet"  # Input path for scaled data
    OUTPUT_PATH = "data/transformed_data_1b/"    # Output path for transformed data

    # Step 1: Load a sample for BGMM fitting
    logging.info("Loading a sample of the data for BGMM fitting.")
    sample_data = pd.read_parquet(INPUT_PATH).head(100000)["Amount"].values

    # Step 2: Fit the ScaleVGM transformer
    logging.info("Fitting the ScaleVGM transformer.")
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 3: Process the dataset with the fitted transformer
    logging.info(
        "Processing the dataset with the fitted ScaleVGM transformer.")
    processor = DataProcessor(input_path=INPUT_PATH,
                              output_path=OUTPUT_PATH, transformer=scale_vgm)
    processor.process_data()

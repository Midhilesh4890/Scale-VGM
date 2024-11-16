import logging
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from multiprocessing import Pool, cpu_count
from glob import glob

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


def process_partition(file_path, transformer, output_path):
    """
    Process a single Parquet file: load, transform, and save.

    Args:
        file_path (str): Path to the input Parquet file.
        transformer (ScaleVGM): Instance of ScaleVGM for transformation.
        output_path (str): Directory to save the transformed partition.

    Returns:
        str: Path to the saved transformed file.
    """
    logging.info(f"Processing file: {file_path}")

    # Load partition
    partition = pd.read_parquet(file_path)

    # Transform the partition
    normalized_data, modes = transformer.transform(partition["Amount"].values)
    partition["Amount_Transformed"] = normalized_data
    # Optional: Add mode assignments for debugging
    partition["Mode_Assignment"] = modes

    # Save the transformed partition
    output_file = f"{output_path}/transformed_{file_path.split('/')[-1]}"
    partition.to_parquet(output_file, index=False, engine="pyarrow")
    logging.info(f"Saved transformed file to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Paths
    INPUT_PATH = "data/scaled_data_1b/"  # Directory with input Parquet files
    # Directory for transformed Parquet files
    OUTPUT_PATH = "data/transformed_data_1b/"

    # Step 1: Fit the ScaleVGM transformer using a sample
    logging.info("Loading a sample of the data for BGMM fitting.")
    # Use the first file as a sample
    sample_file = glob(f"{INPUT_PATH}/*.parquet")[0]
    sample_data = pd.read_parquet(
        sample_file)["Amount"].values[:100000]  # Load first 100,000 rows
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 2: Discover all Parquet files for processing
    logging.info("Discovering input Parquet files.")
    files = glob(f"{INPUT_PATH}/*.parquet")
    logging.info(f"Discovered {len(files)} files for processing.")

    # Step 3: Process all files in parallel using multiprocessing
    logging.info(f"Starting multiprocessing with {cpu_count()} workers.")
    with Pool(processes=cpu_count()) as pool:
        results = [
            pool.apply_async(process_partition, (file, scale_vgm, OUTPUT_PATH))
            for file in files
        ]
        outputs = [res.get() for res in results]

    logging.info("All files processed successfully.")

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from multiprocessing import Pool, cpu_count
from glob import glob
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ScaleVGM:
    """
    Implements Variational Gaussian Mixture (VGM) with Mode-Specific Normalization
    as per the CTGAN paper for scalable datasets.
    """

    def __init__(self, n_components=10, random_state=42, eps=0.005):
        self.n_components = n_components
        self.random_state = random_state
        self.eps = eps
        self.bgmm = None

    def fit(self, data):
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
        if self.bgmm is None:
            raise ValueError(
                "BGMM model has not been fitted. Call 'fit' first.")
        means = self.bgmm.means_.flatten()
        stds = np.sqrt(self.bgmm.covariances_.flatten())
        modes = self.bgmm.predict(data.reshape(-1, 1))
        normalized_data = np.zeros_like(data)
        for mode in np.unique(modes):
            indices = modes == mode
            normalized_data[indices] = (
                data[indices] - means[mode]) / (4 * stds[mode])
            normalized_data[indices] = np.clip(
                normalized_data[indices], -0.99, 0.99)
        return normalized_data, modes


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
    start_time = time.time()
    logging.info(f"Started processing file: {file_path}")

    # Load partition
    partition = pd.read_parquet(file_path)

    # Transform the partition
    normalized_data, modes = transformer.transform(partition["Amount"].values)
    partition["Amount_Transformed"] = normalized_data
    partition["Mode_Assignment"] = modes

    # Save the transformed partition
    output_file = os.path.join(
        output_path, f"transformed_{os.path.basename(file_path)}")
    partition.to_parquet(output_file, index=False, engine="pyarrow")
    elapsed_time = time.time() - start_time
    logging.info(
        f"Finished processing file: {file_path} in {elapsed_time:.2f} seconds.")
    return output_file


if __name__ == "__main__":
    # Paths
    INPUT_PATH = "data/scaled_data_1b/"  # Directory with input Parquet files
    # Directory for transformed Parquet files
    OUTPUT_PATH = "data/python_transformed_data_1b/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Step 1: Fit the ScaleVGM transformer using a sample
    logging.info("Loading a sample of the data for BGMM fitting.")
    sample_file = glob(f"{INPUT_PATH}/*.parquet")[0]
    sample_data = pd.read_parquet(sample_file)["Amount"].values[:100000]
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 2: Discover all Parquet files for processing
    logging.info("Discovering input Parquet files.")
    files = glob(f"{INPUT_PATH}/*.parquet")
    logging.info(f"Discovered {len(files)} files for processing.")

    # Step 3: Process all files in parallel using multiprocessing
    logging.info(f"Starting multiprocessing with {cpu_count()} workers.")
    total_start_time = time.time()

    with Pool(processes=cpu_count()) as pool:
        results = [
            pool.apply_async(process_partition, (file, scale_vgm, OUTPUT_PATH))
            for file in files
        ]
        outputs = [res.get() for res in results]

    total_elapsed_time = time.time() - total_start_time
    logging.info(
        f"All files processed successfully in {total_elapsed_time:.2f} seconds.")

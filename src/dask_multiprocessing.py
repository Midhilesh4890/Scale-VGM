import glob
import logging
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from dask.distributed import Client
import dask.dataframe as dd
import os
import time

# Configure logging for the script to monitor execution
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class ScaleVGM:
    """
    A class to apply Bayesian Gaussian Mixture Model (BGMM) scaling.
    It normalizes data based on the clusters detected by BGMM.
    """

    def __init__(self, n_components=10, random_state=42, eps=0.005):
        self.n_components = n_components  # Number of mixture components
        self.random_state = random_state  # Seed for reproducibility
        self.eps = eps  # Small value to avoid division by zero
        self.bgmm = None  # Placeholder for the BGMM model

    def fit(self, data):
        """
        Fit the BGMM model to the input data.
        :param data: 1D numpy array of data to fit the model.
        """
        logging.info("Fitting Bayesian Gaussian Mixture Model (BGMM).")
        # Initialize BGMM with specified parameters
        self.bgmm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=200,
            random_state=self.random_state,
        )
        # Fit the BGMM model on the reshaped data
        self.bgmm.fit(data.reshape(-1, 1))
        logging.info("BGMM fitting completed.")

    def transform(self, data):
        """
        Normalize data using the fitted BGMM model.
        :param data: 1D numpy array of data to be normalized.
        :return: Normalized data and cluster modes.
        """
        if self.bgmm is None:
            raise ValueError(
                "BGMM model has not been fitted. Call 'fit' first.")

        # Extract mean and standard deviation for each cluster
        means = self.bgmm.means_.flatten()
        stds = np.sqrt(self.bgmm.covariances_.flatten())
        # Predict cluster assignments for the data
        modes = self.bgmm.predict(data.reshape(-1, 1))

        # Normalize data based on cluster parameters
        normalized_data = np.zeros_like(data)
        for mode in np.unique(modes):
            indices = modes == mode
            normalized_data[indices] = (
                data[indices] - means[mode]) / (4 * stds[mode])
            # Clip normalized values to avoid extreme outliers
            normalized_data[indices] = np.clip(
                normalized_data[indices], -0.99, 0.99)

        return normalized_data, modes


class DataProcessor:
    """
    A class to process large datasets using Dask and apply transformations
    with the ScaleVGM transformer.
    """

    def __init__(self, input_path, output_path, transformer):
        self.input_path = input_path  # Path to input Parquet files
        self.output_path = output_path  # Path to save transformed data
        self.transformer = transformer  # ScaleVGM transformer instance

    def process_data(self):
        """
        Process the input dataset in parallel using Dask, apply transformations,
        and save the output as Parquet files.
        """
        start_time = time.time()
        logging.info("Loading dataset using Dask.")

        # Load dataset using Dask for parallel processing
        ddf = dd.read_parquet(self.input_path, engine="pyarrow")
        logging.info("Repartitioning dataset for low-memory processing.")
        # Create smaller partitions
        ddf = ddf.repartition(partition_size="50MB")

        # Apply transformations using the ScaleVGM transformer
        logging.info("Applying VGM transformation to all partitions.")
        transformed_ddf = ddf.map_partitions(self._transform_partition)

        # Save the transformed dataset to the output path
        logging.info(f"Saving transformed dataset to {self.output_path}.")
        transformed_ddf.to_parquet(
            self.output_path, engine="pyarrow", write_index=False, compression="snappy"
        )
        logging.info("Transformation completed and data saved.")

        end_time = time.time()
        logging.info(
            f"Total processing time: {end_time - start_time:.2f} seconds.")

    def _transform_partition(self, partition):
        """
        Apply the ScaleVGM transformer to a single partition of the dataset.
        :param partition: A pandas DataFrame partition from Dask.
        :return: Transformed partition with added columns.
        """
        start_time = time.time()
        logging.info(
            f"Started processing a partition of size: {partition.shape[0]} rows.")

        try:
            # Transform the "Amount" column and create new columns
            normalized_data, modes = self.transformer.transform(
                partition["Amount"].values)
            partition["Amount_Transformed"] = normalized_data
            partition["Mode_Assignment"] = modes
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            raise

        elapsed_time = time.time() - start_time
        logging.info(
            f"Finished processing partition in {elapsed_time:.2f} seconds.")
        return partition


if __name__ == "__main__":
    # Configure Dask for low-memory systems
    client = Client(
        processes=True,
        n_workers=2,  # Limited to 2 workers for parallelism
        threads_per_worker=1,
        memory_limit="3GB",  # Restrict memory usage per worker
        local_directory="dask_temp",  # Disk spilling location for Dask
    )
    logging.info(f"Dask dashboard link: {client.dashboard_link}")

    # Paths for input and output data
    INPUT_PATH = "data/scaled_data_1b/*.parquet"
    OUTPUT_PATH = "data/transformed_data_1b/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Ensure output directory exists

    # Step 1: BGMM Fitting
    logging.info("Loading a sample of the data for BGMM fitting.")
    parquet_files = glob.glob(INPUT_PATH)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found at {INPUT_PATH}")

    # Load a small sample for fitting the BGMM
    sample_frames = [pd.read_parquet(file) for file in parquet_files[:2]]
    sample_data = pd.concat(sample_frames).sample(
        n=10000, random_state=42)["Amount"].values

    # Fit the ScaleVGM transformer
    logging.info("Fitting the ScaleVGM transformer.")
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 2: Data Transformation
    logging.info(
        "Processing the dataset with the fitted ScaleVGM transformer.")
    processor = DataProcessor(input_path=INPUT_PATH,
                              output_path=OUTPUT_PATH, transformer=scale_vgm)
    processor.process_data()

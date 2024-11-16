import dask.dataframe as dd
import numpy as np
import os
import logging
import time
import gc
from dask.distributed import Client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataScalerDask:
    """
    Class for scaling datasets to larger sizes using Dask Distributed, with memory management and progress logs.
    """

    def __init__(self, input_path, csv_output_path, parquet_output_path, target_rows):
        """
        Initialize the DataScalerDask.

        Parameters:
        - input_path (str): Path to the input dataset.
        - csv_output_path (str): Path to save the scaled dataset in CSV format.
        - parquet_output_path (str): Path to save the scaled dataset in Parquet format.
        - target_rows (int): Target number of rows in the scaled dataset.
        """
        self.input_path = input_path
        self.csv_output_path = csv_output_path
        self.parquet_output_path = parquet_output_path
        self.target_rows = target_rows

    def log_time(self, operation_name, start_time, completed_rows=None):
        """
        Logs the time taken for a specific operation with optional progress information.

        Parameters:
        - operation_name (str): Name of the operation.
        - start_time (float): Start time of the operation.
        - completed_rows (int): Number of rows processed so far (optional).
        """
        end_time = time.time()
        elapsed_time = end_time - start_time
        if completed_rows is not None:
            logging.info(
                f"{operation_name} completed {completed_rows:,} rows in {elapsed_time:.2f} seconds."
            )
        else:
            logging.info(
                f"{operation_name} completed in {elapsed_time:.2f} seconds.")

    def load_data(self):
        """
        Load the input dataset using Dask, and drop any unwanted index columns.

        Returns:
        - dask.DataFrame: Loaded dataset.
        """
        start_time = time.time()
        logging.info(f"Loading dataset from {self.input_path}...")
        # Avoid type conflicts
        data = dd.read_csv(self.input_path, assume_missing=True)

        # Drop 'Unnamed: 0' or any implicit index column
        if "Unnamed: 0" in data.columns:
            logging.info("Dropping unwanted column: 'Unnamed: 0'")
            data = data.drop(columns=["Unnamed: 0"])

        self.log_time("Dataset loading", start_time)
        return data

    def scale_data(self, data):
        """
        Scale the dataset to the target size.

        Parameters:
        - data (dask.DataFrame): Input dataset.

        Returns:
        - None: Saves the scaled dataset to the output paths.
        """
        start_time = time.time()
        logging.info(f"Scaling dataset to {self.target_rows:,} rows...")

        # Compute the number of replications required
        current_rows = data.shape[0].compute()
        replication_factor = (self.target_rows // current_rows) + 1
        logging.info(f"Replication factor: {replication_factor}")

        # Replicate the dataset and partition for parallelism
        replicated_data = dd.concat(
            [data] * replication_factor, interleave_partitions=True)

        # Ensure the dataset is sliced lazily to the target size
        replicated_data = replicated_data.head(self.target_rows, compute=False)

        # Add noise to numeric columns using map_partitions
        def add_noise(df, cols):
            """
            Add random noise to numeric columns in a Dask DataFrame.

            Parameters:
            - df (pd.DataFrame): A Pandas DataFrame (processed partition).
            - cols (list): List of column names to add noise to.

            Returns:
            - pd.DataFrame: DataFrame with noise added.
            """
            for col in cols:
                df[col] += np.random.normal(0, 0.01, size=len(df))
            return df

        numeric_columns = list(data.select_dtypes(include=["number"]).columns)
        if numeric_columns:
            logging.info(f"Adding noise to numeric columns: {numeric_columns}")
            replicated_data = replicated_data.map_partitions(
                add_noise, cols=numeric_columns)

        self.log_time("Data scaling", start_time)

        # Save partitioned Parquet files directly
        start_time = time.time()
        logging.info(
            f"Saving scaled dataset to {self.parquet_output_path} (Parquet format, partitioned)...")
        replicated_data.to_parquet(
            self.parquet_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            compute=True,
        )
        self.log_time("Parquet save", start_time)

        # Save to CSV
        start_time = time.time()
        logging.info(
            f"Saving scaled dataset to {self.csv_output_path} (CSV format, single file)...")
        replicated_data.to_csv(self.csv_output_path,
                               single_file=True, index=False, compute=True)
        self.log_time("CSV save", start_time)

    def run(self):
        """
        Execute the scaling pipeline with time tracking.
        """
        start_time = time.time()
        data = self.load_data()
        self.scale_data(data)
        self.log_time("Total pipeline", start_time)


if __name__ == "__main__":
    # Initialize Dask Distributed with optimized settings for your machine
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="1.5GB")
    logging.info("Dask Distributed Client initialized.")
    logging.info(client)

    # Define input and output paths
    INPUT_PATH = "data/New_Credit.csv"
    CSV_OUTPUT_PATH = "data/scaled_data_1b.csv"
    PARQUET_OUTPUT_PATH = "data/scaled_data_1b/"
    TARGET_ROWS = 1_000_000_000  # 1 billion rows

    # Initialize and run the scaling process
    scaler = DataScalerDask(
        input_path=INPUT_PATH,
        csv_output_path=CSV_OUTPUT_PATH,
        parquet_output_path=PARQUET_OUTPUT_PATH,
        target_rows=TARGET_ROWS,
    )
    scaler.run()

    # Shutdown the Dask client
    client.shutdown()

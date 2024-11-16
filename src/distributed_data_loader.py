import dask.dataframe as dd
import numpy as np
import os
import logging
import time
import gc
from multiprocessing import Pool, cpu_count
from typing import List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataScalerDask:
    """
    Class for scaling datasets to larger sizes using Dask and multiprocessing,
    with time tracking and memory management.
    """

    def __init__(self, input_path: str, parquet_output_path: str, target_rows: int) -> None:
        """
        Initialize the DataScalerDask.

        Parameters:
        - input_path (str): Path to the input dataset.
        - parquet_output_path (str): Path to save the scaled dataset in Parquet format.
        - target_rows (int): Target number of rows in the scaled dataset.
        """
        self.input_path = input_path
        self.parquet_output_path = parquet_output_path
        self.target_rows = target_rows

    def log_time(self, operation_name: str, start_time: float, completed_rows: Optional[int] = None) -> None:
        """
        Logs the time taken for a specific operation with optional progress information.

        Parameters:
        - operation_name (str): Name of the operation.
        - start_time (float): Start time of the operation.
        - completed_rows (Optional[int]): Number of rows processed so far (optional).
        """
        end_time = time.time()
        elapsed_time = end_time - start_time
        if completed_rows is not None:
            logging.info(
                f"{operation_name} completed {completed_rows:,} rows in {elapsed_time:.2f} seconds."
            )
        else:
            logging.info(
                f"{operation_name} completed in {elapsed_time:.2f} seconds."
            )

    def load_data(self) -> dd.DataFrame:
        """
        Load the input dataset using Dask, and drop any unwanted index columns.

        Returns:
        - dd.DataFrame: Loaded dataset.
        """
        start_time = time.time()
        logging.info(f"Loading dataset from {self.input_path}...")
        data: dd.DataFrame = dd.read_csv(self.input_path, assume_missing=True)

        # Drop 'Unnamed: 0' or any implicit index column
        if "Unnamed: 0" in data.columns:
            logging.info("Dropping unwanted column: 'Unnamed: 0'")
            data = data.drop(columns=["Unnamed: 0"])

        self.log_time("Dataset loading", start_time)
        return data

    @staticmethod
    def add_noise(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Add random noise to numeric columns in a Pandas DataFrame.

        Parameters:
        - df (pd.DataFrame): A Pandas DataFrame (processed partition).
        - cols (List[str]): List of column names to add noise to.

        Returns:
        - pd.DataFrame: DataFrame with noise added.
        """
        for col in cols:
            df[col] += np.random.normal(0, 0.01, size=len(df))
        return df

    def scale_data(self, data: dd.DataFrame) -> None:
        """
        Scale the dataset using multiprocessing to parallelize chunk processing.

        Parameters:
        - data (dd.DataFrame): The input Dask DataFrame to scale.
        """
        start_time = time.time()
        logging.info(f"Scaling dataset to {self.target_rows:,} rows...")

        # Convert Dask DataFrame to Pandas DataFrame for multiprocessing
        data = data.compute()
        current_rows = len(data)

        # Identify numeric columns
        numeric_columns = list(data.select_dtypes(include=["number"]).columns)

        # Prepare for multiprocessing
        rows_per_chunk = 10_000_000  # Adjust based on memory constraints
        total_chunks = (self.target_rows + rows_per_chunk -
                        1) // rows_per_chunk  # Ceiling division

        chunk_info_list = []
        rows_processed = 0
        for chunk_index in range(total_chunks):
            remaining_rows = self.target_rows - rows_processed
            current_chunk_size = min(rows_per_chunk, remaining_rows)

            chunk_info_list.append({
                "data": data,
                "numeric_columns": numeric_columns,
                "chunk_size": current_chunk_size,
                "parquet_output_path": self.parquet_output_path,
                "chunk_index": chunk_index,
            })

            rows_processed += current_chunk_size

        # Process chunks in parallel using multiprocessing
        with Pool(cpu_count()) as pool:
            chunk_paths = pool.map(self.process_chunk, chunk_info_list)

        self.log_time("Scaling with multiprocessing", start_time)

    @staticmethod
    def process_chunk(chunk_info):
        """
        Processes a single chunk: replicates data, adds noise, and saves it to a Parquet file.

        Parameters:
        - chunk_info (dict): Information for processing the chunk.

        Returns:
        - str: Path to the saved Parquet chunk.
        """
        data, numeric_columns, chunk_size, parquet_output_path, chunk_index = (
            chunk_info["data"],
            chunk_info["numeric_columns"],
            chunk_info["chunk_size"],
            chunk_info["parquet_output_path"],
            chunk_info["chunk_index"],
        )

        # Replicate data for the chunk
        current_rows = len(data)
        times_to_repeat_data = chunk_size // current_rows
        remainder_rows = chunk_size % current_rows

        replicated_data_list = []
        if times_to_repeat_data > 0:
            replicated_data_list.append(
                pd.concat([data] * times_to_repeat_data, ignore_index=True))
        if remainder_rows > 0:
            replicated_data_list.append(data.iloc[:remainder_rows])

        replicated_data = pd.concat(replicated_data_list, ignore_index=True)

        # Add noise to numeric columns
        if numeric_columns:
            replicated_data = DataScalerDask.add_noise(
                replicated_data, numeric_columns)

        # Save the chunk to Parquet
        chunk_parquet_path = os.path.join(
            parquet_output_path, f"chunk_{chunk_index}.parquet")
        replicated_data.to_parquet(
            chunk_parquet_path, engine="pyarrow", index=False)

        return chunk_parquet_path

    def run(self) -> None:
        """
        Execute the scaling pipeline with time tracking.
        """
        start_time = time.time()
        data = self.load_data()
        self.scale_data(data)
        self.log_time("Total pipeline", start_time)


if __name__ == "__main__":
    # Example usage
    INPUT_PATH = "data/New_Credit.csv"
    PARQUET_OUTPUT_PATH = "data/scaled_data_1b/"
    TARGET_ROWS = 1_000_000_000  # 1 billion rows

    scaler = DataScalerDask(
        input_path=INPUT_PATH,
        parquet_output_path=PARQUET_OUTPUT_PATH,
        target_rows=TARGET_ROWS,
    )
    scaler.run()

import dask.dataframe as dd
import numpy as np
import os
import logging
import time
import gc
from typing import List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataScalerDask:
    """
    Class for scaling datasets to larger sizes using Dask, with time tracking and memory management.
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
        # Avoid type conflicts
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
        start_time: float = time.time()
        logging.info(f"Scaling dataset to {self.target_rows:,} rows...")

        current_rows: int = data.shape[0].compute()
        total_rows_needed = self.target_rows
        rows_per_chunk: int = 10_000_000  # Adjust if necessary
        total_chunks: int = (total_rows_needed + rows_per_chunk -
                             1) // rows_per_chunk  # Ceiling division
        rows_processed: int = 0

        # Identify numeric columns once
        numeric_columns: List[str] = list(
            data.select_dtypes(include=["number"]).columns
        )

        for chunk in range(total_chunks):
            remaining_rows = total_rows_needed - rows_processed
            current_chunk_size = min(rows_per_chunk, remaining_rows)

            times_to_repeat_data = current_chunk_size // current_rows
            remainder_rows_in_chunk = current_chunk_size % current_rows

            # Replicate data as needed for the current chunk
            replicated_data_list = []
            if times_to_repeat_data > 0:
                replicated_data_list.append(
                    dd.concat([data] * times_to_repeat_data,
                              interleave_partitions=True)
                )
            if remainder_rows_in_chunk > 0:
                remainder_data = data.head(
                    remainder_rows_in_chunk, compute=False)
                remainder_data = dd.from_pandas(remainder_data, npartitions=1)
                replicated_data_list.append(remainder_data)

            if replicated_data_list:
                replicated_data = dd.concat(
                    replicated_data_list, interleave_partitions=True
                )
            else:
                # If no data to replicate, skip
                logging.warning(f"No data to replicate in chunk {chunk}.")
                continue

            # Add noise to numeric columns
            if numeric_columns:
                replicated_data = replicated_data.map_partitions(
                    self.add_noise, cols=numeric_columns
                )

            # Save the chunk incrementally to Parquet
            chunk_parquet_path: str = os.path.join(
                self.parquet_output_path, f"chunk_{chunk}.parquet"
            )
            replicated_data.to_parquet(
                chunk_parquet_path, engine="pyarrow", write_index=False
            )

            # Update progress
            rows_processed += current_chunk_size
            self.log_time(
                f"Chunk {chunk + 1}/{total_chunks}",
                start_time,
                completed_rows=rows_processed,
            )

            # Clean up memory
            del replicated_data
            gc.collect()

        self.log_time("Total scaling", start_time)

        logging.info("Data scaling and saving to Parquet completed.")

    def run(self) -> None:
        """
        Execute the scaling pipeline with time tracking.
        """
        start_time: float = time.time()
        data: dd.DataFrame = self.load_data()
        self.scale_data(data)
        self.log_time("Total pipeline", start_time)


if __name__ == "__main__":
    # Example usage
    INPUT_PATH: str = "data/New_Credit.csv"
    PARQUET_OUTPUT_PATH: str = "data/scaled_data_1b/"
    TARGET_ROWS: int = 1_000_000_000  # 1 billion rows

    scaler = DataScalerDask(
        input_path=INPUT_PATH,
        parquet_output_path=PARQUET_OUTPUT_PATH,
        target_rows=TARGET_ROWS,
    )
    scaler.run()

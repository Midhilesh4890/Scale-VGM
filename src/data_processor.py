import logging
import time
import dask.dataframe as dd

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataProcessor:
    """
    A class to process large datasets using Dask and apply transformations
    with the ScaleVGM transformer.

    Attributes:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the transformed dataset.
        transformer (ScaleVGM): An instance of the ScaleVGM transformer.

    Methods:
        process_data(): Processes the dataset in parallel and saves the results.
        _transform_partition(partition): Transforms a single partition of the dataset.
    """

    def __init__(self, input_path, output_path, transformer):
        """
        Initializes the DataProcessor with paths and the transformer.

        Args:
            input_path (str): Path to the input dataset (Parquet files).
            output_path (str): Path to save the transformed dataset.
            transformer (ScaleVGM): Instance of the ScaleVGM transformer.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.transformer = transformer

    def process_data(self):
        """
        Processes the input dataset using Dask, applies transformations,
        and saves the results as Parquet files.

        The input dataset is partitioned to optimize memory usage, and each partition
        is transformed using the provided ScaleVGM transformer.
        """
        start_time = time.time()
        logging.info("Loading dataset using Dask.")

        # Load the dataset using Dask for parallel processing
        ddf = dd.read_parquet(self.input_path, engine="pyarrow")
        # Repartition the dataset to reduce memory usage during processing
        ddf = ddf.repartition(partition_size="50MB")

        logging.info("Applying transformations to all partitions.")
        # Apply the transformer to each partition
        transformed_ddf = ddf.map_partitions(self._transform_partition)

        logging.info(f"Saving transformed dataset to {self.output_path}.")
        # Save the transformed dataset as Parquet files
        transformed_ddf.to_parquet(
            self.output_path, engine="pyarrow", write_index=False, compression="snappy"
        )

        logging.info("Data processing and saving completed.")
        logging.info(
            f"Total processing time: {time.time() - start_time:.2f} seconds.")

    def _transform_partition(self, partition):
        """
        Applies the ScaleVGM transformer to a single partition.

        Args:
            partition (pandas.DataFrame): A single partition of the dataset.

        Returns:
            pandas.DataFrame: Transformed partition with added columns.
        """
        start_time = time.time()
        logging.info(f"Processing a partition with {partition.shape[0]} rows.")

        try:
            # Transform the 'Amount' column
            normalized_data, modes = self.transformer.transform(
                partition["Amount"].values)
            # Add transformed columns to the partition
            partition["Amount_Transformed"] = normalized_data
            partition["Mode_Assignment"] = modes
        except Exception as e:
            logging.error(f"Error during partition transformation: {e}")
            raise

        logging.info(
            f"Partition processed in {time.time() - start_time:.2f} seconds.")
        return partition

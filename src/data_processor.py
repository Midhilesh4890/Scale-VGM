import logging
import dask.dataframe as dd

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

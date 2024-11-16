import dask.dataframe as dd


class DataProcessor:
    """
    A class to handle processing and statistics for a Parquet dataset using Dask.
    """

    def __init__(self, parquet_path: str):
        """
        Initialize the processor with the path to the Parquet dataset.
        :param parquet_path: Path to the partitioned Parquet directory
        """
        self.parquet_path = parquet_path
        self.df = self._load_data()

    def _load_data(self) -> dd.DataFrame:
        """
        Load the Parquet files into a Dask DataFrame.
        :return: Dask DataFrame
        """
        try:
            return dd.read_parquet(self.parquet_path)
        except Exception as e:
            raise ValueError(
                f"Error loading data from {self.parquet_path}: {e}")

    def get_total_rows(self) -> int:
        """
        Compute the total number of rows in the dataset.
        :return: Total row count
        """
        return self.df.shape[0].compute()

    def get_column_statistics(self) -> dict:
        """
        Compute basic statistics (mean, std, min, max) for all numeric columns.
        :return: Dictionary containing statistics
        """
        numeric_df = self.df.select_dtypes(include=['float', 'int'])
        stats = numeric_df.describe().compute()
        return stats.to_dict()

    def get_missing_values(self) -> dict:
        """
        Compute the total number of missing values for each column.
        :return: Dictionary with column names as keys and missing value counts as values
        """
        missing_values = self.df.isnull().sum().compute()
        return missing_values.to_dict()

    def get_memory_usage(self) -> float:
        """
        Estimate the memory usage of the Dask DataFrame.
        :return: Estimated memory usage in MB
        """
        return self.df.memory_usage(deep=True).sum().compute() / (1024 * 1024)

    def to_pandas(self) -> "pd.DataFrame":
        """
        Convert the Dask DataFrame to a Pandas DataFrame.
        :return: Pandas DataFrame
        """
        return self.df.compute()


class StatsReport:
    """
    A class to handle reporting of statistics for a Parquet dataset.
    """

    def __init__(self, processor: DataProcessor):
        """
        Initialize the reporter with a DataProcessor.
        :param processor: Instance of DataProcessor
        """
        self.processor = processor

    def generate_report(self):
        """
        Generate a comprehensive report of the dataset.
        """
        try:
            print("Dataset Statistics Report")
            print(f"Total Rows: {self.processor.get_total_rows()}")
            print(f"Memory Usage: {self.processor.get_memory_usage():.2f} MB")
            print("\nMissing Values:")
            for col, missing in self.processor.get_missing_values().items():
                print(f"  {col}: {missing} missing values")

            print("\nColumn Statistics:")
            column_stats = self.processor.get_column_statistics()
            for col, stats in column_stats.items():
                print(f"\n  {col}:")
                for stat, value in stats.items():
                    print(f"    {stat}: {value}")

        except Exception as e:
            print(f"Error generating report: {e}")


if __name__ == "__main__":
    # Path to the Parquet dataset
    parquet_path = "data/scaled_data_1b/"

    # Initialize the processor and reporter
    processor = DataProcessor(parquet_path)
    reporter = StatsReport(processor)

    # Generate and display the report
    reporter.generate_report()

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from dask.dataframe import from_pandas
from src.distributed_data_scaling import DataScalerDask  # Adjusted import
import os


@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data as a Pandas DataFrame.
    """
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4.0, 5.1, 6.2],
        "C": ["x", "y", "z"]
    })


@pytest.fixture
def dask_df(sample_data):
    """
    Fixture to provide sample data as a Dask DataFrame.
    """
    return from_pandas(sample_data, npartitions=1)


@pytest.fixture
def scaler():
    """
    Fixture to initialize the DataScalerDask instance with mocked paths.
    """
    input_path = "mock_input.csv"
    parquet_output_path = "test_data/output"
    target_rows = 10_000
    return DataScalerDask(input_path, parquet_output_path, target_rows)


@patch("dask.dataframe.read_csv")
def test_load_data(mock_read_csv, dask_df, scaler):
    """
    Test the load_data method to ensure it loads the dataset correctly.
    """
    mock_read_csv.return_value = dask_df
    result = scaler.load_data()
    mock_read_csv.assert_called_once_with(
        scaler.input_path, assume_missing=True)
    assert result.equals(dask_df)


def test_add_noise(sample_data):
    """
    Test the add_noise method to ensure noise is added to numeric columns.
    """
    numeric_columns = ["A", "B"]
    result = DataScalerDask.add_noise(sample_data.copy(), numeric_columns)
    for col in numeric_columns:
        # Ensure values are changed
        assert not result[col].equals(sample_data[col])


@patch("os.makedirs")
@patch("src.distributed_data_scaling.DataScalerDask.process_chunk")
def test_scale_data(mock_process_chunk, mock_makedirs, dask_df, scaler):
    """
    Test the scale_data method for correct chunk processing and directory creation.
    """
    mock_process_chunk.return_value = "test_path"
    mock_data = MagicMock()
    mock_data.compute.return_value = dask_df.compute()

    scaler.scale_data(mock_data)

    mock_makedirs.assert_called_once_with(
        scaler.parquet_output_path, exist_ok=True)
    mock_process_chunk.assert_called()


@patch("pandas.DataFrame.to_parquet")
def test_process_chunk(mock_to_parquet, sample_data):
    """
    Test the process_chunk method for replicating data and saving to Parquet.
    """
    chunk_info = {
        "data": sample_data,
        "numeric_columns": ["A", "B"],
        "chunk_size": 10,
        "parquet_output_path": "test_data/output",
        "chunk_index": 0,
    }

    DataScalerDask.process_chunk(chunk_info)
    mock_to_parquet.assert_called_once()


@patch("time.time")
def test_log_time(mock_time, scaler):
    """
    Test the log_time method to ensure proper time logging.
    """
    mock_time.side_effect = [100, 105]

    with patch("builtins.print") as mock_print:
        scaler.log_time("Test Operation", 100)
        mock_print.assert_called_with(
            "Test Operation completed in 5.00 seconds.")


@pytest.fixture(autouse=True)
def cleanup():
    """
    Clean up after tests by removing test directories.
    """
    yield
    if os.path.exists("test_data/output"):
        os.rmdir("test_data/output")

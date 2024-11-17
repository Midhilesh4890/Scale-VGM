import logging
import os
import glob
import pandas as pd
from dask.distributed import Client

# Import custom modules from the src directory
from src.scale_vgm import ScaleVGM
from src.data_processor import DataProcessor

if __name__ == "__main__":
    # Configure logging for the script to monitor execution
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
    sample_data = (
        pd.concat(sample_frames)
        .sample(n=10000, random_state=42)["Amount"]
        .values
    )

    # Initialize and fit the ScaleVGM transformer
    logging.info("Fitting the ScaleVGM transformer.")
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 2: Data Transformation
    logging.info(
        "Processing the dataset with the fitted ScaleVGM transformer."
    )
    processor = DataProcessor(
        input_path=INPUT_PATH, output_path=OUTPUT_PATH, transformer=scale_vgm
    )
    processor.process_data()

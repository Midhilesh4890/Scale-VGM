from dask.distributed import Client
from src.scale_vgm import ScaleVGM
from src.data_processor import DataProcessor
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Configure Dask for multiprocessing
    client = Client(processes=True, n_workers=4, threads_per_worker=1)
    print(client.dashboard_link)

    # Paths
    INPUT_PATH = "data/scaled_data_1b/*.parquet"  # Input path for scaled data
    OUTPUT_PATH = "data/transformed_data_1b/"    # Output path for transformed data

    # Step 1: Load a sample for BGMM fitting
    logging.info("Loading a sample of the data for BGMM fitting.")
    sample_data = pd.read_parquet(INPUT_PATH).head(100000)["Amount"].values

    # Step 2: Fit the ScaleVGM transformer
    logging.info("Fitting the ScaleVGM transformer.")
    scale_vgm = ScaleVGM(n_components=10)
    scale_vgm.fit(sample_data)

    # Step 3: Process the dataset with the fitted transformer
    logging.info(
        "Processing the dataset with the fitted ScaleVGM transformer.")
    processor = DataProcessor(input_path=INPUT_PATH,
                              output_path=OUTPUT_PATH, transformer=scale_vgm)
    processor.process_data()

# import pandas as pd 
# import pdb
# df = pd.read_csv('data/New_Credit.csv')
# print(df.columns)
# print(df.shape)

# pdb.set_trace()
import dask.dataframe as dd
# Path to the partitioned Parquet directory
parquet_path = "data/scaled_data_1b/"

# Read all partitioned Parquet files
df = dd.read_parquet(parquet_path)

# Perform operations or convert to Pandas
print(f"Total rows: {df.shape[0].compute()}")  # Compute total rows

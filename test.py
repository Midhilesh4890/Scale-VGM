# import pandas as pd 
# df = pd.read_csv('data/scaled_data_10m.csv')
# print(df.columns)
# print(df.shape)


import dask.dataframe as dd
# Path to the partitioned Parquet directory
parquet_path = "data/scaled_data_1b/"

# Read all partitioned Parquet files
df = dd.read_parquet(parquet_path)

# Perform operations or convert to Pandas
print(df.head())  # View the first few rows
print(f"Total rows: {df.shape[0].compute()}")  # Compute total rows

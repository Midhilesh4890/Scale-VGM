import pandas as pd 
df = pd.read_csv('Credit.csv', usecols=['Amount'])
print(df.columns)
print(df.shape)
df.to_csv('New_Credit.csv')
import pandas as pd
import numpy as np

# for reading excel files
import openpyxl

# load dataset into pandas dataframe
file_path = 'a1/real_estate_data.xlsx'
df = pd.read_excel(file_path)

# verify what the data looks like
print(df.head())
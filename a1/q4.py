import pandas as pd
import numpy as np

# for reading excel files
import openpyxl

# load dataset into pandas dataframe
file_path = 'a1/real_estate_data.xlsx'
df = pd.read_excel(file_path)

# Extract relevant columns
features = df[['X2 House Age', 'X3 Distance to the Nearest MRT Station', 'X4 Number of Convenience Stores']]
target = df['Y House Price of Unit Area']

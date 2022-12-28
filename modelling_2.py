#%%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd

df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")

df['Category'].info()
print(df['Category'].unique())
df.info()
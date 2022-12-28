#%%
from tabular_data import load_airbnb
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import numpy as np
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
df
# ohe = OneHotEncoder(drop='first')
# print(ohe)
# df['Category'].info()
#df['Category'].type()
#print(df['Category'].unique())
# print(df.groupby('Category').size())
#df.info()
#print(df.shape)
# %%
#sns.countplot(df['Category'],label="Count")
#plt.show()
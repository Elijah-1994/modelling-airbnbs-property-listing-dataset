#%%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
import inspect
#%%
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
X, y = load_airbnb(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#%%
class SGDRegressor:
    def __init__(self,n_features):
        np.random.seed(2)
        self.weights = np.random.randn(n_features, 1)
        self.bias = np.random.randn(1)
    
    def predict(self, X):
        y_prediction = X @ self.weights + self.bias
        return y_prediction
    
    def get_params(deep=True):
        return  dict(deep=deep)
        

model = SGDRegressor(n_features=9)
y_pred = model.predict(X_train) 
y_params = model.get_params
print("Predictions:\n", y_pred[:20]) 

#%%
from tabular_data import load_airbnb
from sklearn.datasets import make_regression
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics 
#import numpy as np
import pandas as pd
#%%
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
X,y = load_airbnb(df)
X, y = make_regression(n_samples=829, n_features=9)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
X = scale(X)
y = scale(y)
#%%
def calculate_loss(y_test, y_pred):
    mse = metrics.mean_squared_error(y_test, y_pred)
    return mse

if __name__ == '__main__':
    model = SGDRegressor()
    model = model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("R-squared:", score)
    y_pred = model.predict(X_test)
    print(y_pred[:5], "\n", y[:5])
    calculate_loss(y_test, y_pred)

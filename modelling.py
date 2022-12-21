#%%
from tabular_data import load_airbnb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics 
import numpy as np
import pandas as pd

#%%
def calculate_loss(y_test, y_pred):
    mse = metrics.mean_squared_error(y_test, y_pred)
    return mse

def MSE(targets, predicted):
    return np.mean(np.square(targets - predicted))

def RMSE(targets, predicted):
    return np.sqrt(MSE(targets, predicted))

def R2(targets, predicted):
    return 1 - (MSE(targets, predicted)/np.var(targets))

#%%
if __name__ == '__main__':
    np.random.seed(10)
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    X, y = make_regression(n_samples=829, n_features=9)
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
    X = scale(X)
    y = scale(y)
    model = SGDRegressor()
    model = model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("R-squared:", score)
    y_pred = model.predict(X_test)
    print(y_pred[:5], "\n", y[:5])
    calculate_loss(y_test, y_pred)
    print("MSE (scikit-learn):", mean_squared_error(y_test, y_pred))
    print("MSE (Python):", MSE(y_test, y_pred))
    print()
    print("RMSE (scikit-learn):", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 (Python):", R2(y_test, y_pred))
    print("R2 (scikit-learn):", r2_score(y_test, y_pred))

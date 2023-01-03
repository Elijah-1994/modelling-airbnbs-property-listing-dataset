#%%.
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import plot_confusion_matrix
import joblib
import json
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
df
#df.info()
# df['Location'].nunique()
# df['Location'].size()
# df['Location'].count()
# df['Location'].info()

if __name__ == '__main__':
    np.random.seed(2)
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    y = y.to_frame().reset_index(drop=True)
    ohe = OneHotEncoder(drop='first')
    y = ohe.fit_transform(y)
    #class_names = ohe.categories_
    print(ohe.categories_)
    print(y.shape)
    y = y.toarray()
    y = y[:,1]
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(log_reg.score(X_test, y_test))
    print(y[:500])
    print()
    print(y_pred[:500])
    title = "Confusion matrix"
    disp = plot_confusion_matrix(
    log_reg, X_train, y_train, cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()
    
  

 

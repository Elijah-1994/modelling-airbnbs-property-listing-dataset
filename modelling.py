#%%
from tabular_data import load_airbnb
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import numpy as np
import pandas as pd
import json
#%%
def tune_regression_model_hyperparameters(model,model_name,X_train, y_train):
    if model_name == 'SGDRegressor':
        params = {
        'alpha':[0.0001,0.0002,0.0003],
        'l1_ratio':[0.15,0.1,0.25], 
        'max_iter':[1000,1250,1500,1750,2000], 
        'tol':[0.001,0.02,0.003],
        'epsilon':[0.1,0.2,0.3,0.5,0.9],
        'eta0':[0.01,0.02,0.03,0.05,0.09],
        'power_t':[0.25,0.35,0.45],
                                    }
        clf = GridSearchCV(
        estimator='SGDRegressor',
        scoring='neg_mean_squared_error',
        param_grid=params,
        cv=5,
        n_jobs=1,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        return clf.best_params_
    
    elif model_name == 'DecisionTreeRegressor':
        params = {
            "splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "max_features":[1.0,"log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
                                                              }
        clf = GridSearchCV(
        estimator=model,
        scoring='neg_mean_squared_error',
        param_grid=params,
        cv=5,
        n_jobs=1,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        return clf.best_params_

def save_model(model,model_name,best_parameters,performance_metrics):
    if model_name == 'SGDRegressor':
        joblib.dump(model, "models/regression/model.joblib")
        with open("models/hyperparameters/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))  
    elif model_name == 'DecisionTreeRegressor':
        joblib.dump(model, "models/regression/DecisionTree/model.joblib")
        with open("models/hyperparameters/DecisionTree/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/DecisionTree/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))  

if __name__ == '__main__':
    np.random.seed(10)
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    n_samples, n_features = 830, 9
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    # model = SGDRegressor()
    # best_parameters = tune_regression_model_hyperparameters(model,X_train, y_train)
    #updated_model = SGDRegressor(**best_parameters)
    #model = updated_model.fit(X_train, y_train)
    # y_pred_test = model.predict(X_test)
    # y_pred_train = model.predict(X_train)
    # RMSE_test = mean_squared_error(y_test, y_pred_test, squared=False)
    # RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)
    # R2_test = r2_score(y_test, y_pred_test)
    # R2_train = r2_score(y_train, y_pred_train)
    # performance_metrics = {'RMSE_test':RMSE_test,'RMSE_train':RMSE_train,'R2_test':R2_test,'R2_train':R2_train}
    # save_model(updated_model,best_parameters,performance_metrics)
    model = DecisionTreeRegressor()
    best_parameters = tune_regression_model_hyperparameters(model,'DecisionTreeRegressor',X_train, y_train)
    updated_model = DecisionTreeRegressor(**best_parameters,random_state=0)
    model = updated_model.fit(X_test,y_test)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    RMSE_test = mean_squared_error(y_test, y_pred_test, squared=False)
    RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)
    R2_test = r2_score(y_test, y_pred_test)
    R2_train = r2_score(y_train, y_pred_train)
    performance_metrics = {'RMSE_test':RMSE_test,'RMSE_train':RMSE_train,'R2_test':R2_test,'R2_train':R2_train}
    score = model.score(X_test,y_test)
    save_model(updated_model,'DecisionTreeRegressor',best_parameters,performance_metrics)
    print(score)
    cross_val_score(model, X, y, cv=10)
    y_pred_train = model.predict(X_train)




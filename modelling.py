#%%.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tabular_data import load_airbnb
import joblib
import json
import numpy as np
import pandas as pd
import json
import warnings
#%%
def evaluate_all_models(X_train,X_test,X_validation,y_train,y_test,y_validation):
    warnings.filterwarnings('ignore')
    np.random.seed(2)
    models = [SGDRegressor(),DecisionTreeRegressor(),RandomForestRegressor(),GradientBoostingRegressor()]
    for model in models:
        model_name = str(model)
        best_parameters = tune_regression_model_hyperparameters(model,model_name,X_train, y_train)
        model= model_name.replace(')', '')
        model = f'{model}**{best_parameters}' +')'
        model= eval(model)
        model.fit(X_train, y_train)
        y_train_prediction = model.predict(X_train)
        y_validation_prediction = model.predict(X_validation)
        y_test_prediction = model.predict(X_test)
        train_loss = mean_squared_error(y_train, y_train_prediction, squared=False)
        validation_loss = mean_squared_error(y_validation,  y_validation_prediction, squared=False)
        test_loss = mean_squared_error(y_test, y_test_prediction, squared=False)
        R2_train = r2_score(y_train, y_train_prediction)
        R2_validation = r2_score(y_validation, y_validation_prediction)
        R2_test = r2_score(y_test, y_test_prediction)
        performance_metrics = {'RMSE_train':train_loss,'RMSE_validation':validation_loss,'RMSE_test':test_loss,
                               'R2_train':R2_train,'R2_test':R2_test,'R2_validation':R2_validation,}
        save_model(model,model_name,best_parameters,performance_metrics)
        
def tune_regression_model_hyperparameters(model,model_name,X_train, y_train):
    if model_name == 'SGDRegressor()':
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
        estimator=model,
        scoring='neg_mean_squared_error',
        param_grid=params,
        cv=5,
        n_jobs=1,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        return clf.best_params_
    
    elif model_name == 'DecisionTreeRegressor()':
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
        n_jobs=5,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        clf.best_params_
        return clf.best_params_

    elif model_name == 'RandomForestRegressor()':
        params = {
            "n_estimators" : [5,20,50,100],
            "max_features":[1.0,"log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
            "max_depth" : [1,3,5,7,9,11,12],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "bootstrap" : [True, False]
    
                                                              }
        clf = GridSearchCV(
        estimator=model,
        scoring='neg_mean_squared_error',
        param_grid=params,
        cv=10,
        n_jobs=5,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        clf.best_params_
        return clf.best_params_
    
    elif model_name == 'GradientBoostingRegressor()':
        params = {
            "learning_rate" : [0.1,1.0,3.0,6.0,9.0,12.0],
            "n_estimators" : [5,20,50,100],
            "subsample": [0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "max_depth" : [1,3,5,7,9,11,12],         
            "tol":[0.0001,0.0002,0.0003]
                                                              }
        clf = GridSearchCV(
        estimator=model,
        scoring='neg_mean_squared_error',
        param_grid=params,
        cv=10,
        n_jobs=1,
        verbose=1
                    )
        clf.fit(X_train, y_train)
        clf.best_params_
        return clf.best_params_

def save_model(model,model_name,best_parameters,performance_metrics):
    if model_name == 'SGDRegressor()':
        joblib.dump(model, "models/regression/SGDRegressor/model.joblib")
        with open("models/hyperparameters/SGDRegressor/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/SGDRegressor/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))  
    elif model_name == 'DecisionTreeRegressor()':
        joblib.dump(model, "models/regression/DecisionTreeRegressor/model.joblib")
        with open("models/hyperparameters/DecisionTreeRegressor/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/DecisionTreeRegressor/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str)) 
    elif model_name == 'RandomForestRegressor()':
        joblib.dump(model, "models/regression/RandomForestRegressor/model.joblib")
        with open("models/hyperparameters/RandomForestRegressor/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/RandomForestRegressor/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))  
    elif model_name == 'GradientBoostingRegressor()':
        joblib.dump(model, "models/regression/GradientBoostingRegressor/model.joblib")
        with open("models/hyperparameters/GradientBoostingRegressor/hyperparameters.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open("models/performance_metrics/GradientBoostingRegressor/performance_metrics.json", mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))   

def find_best_model ():
    RMSE_validations =[]
    with open("models/performance_metrics/SGDRegressor/performance_metrics.json", mode='r') as f:
        json_dict_SGDRegressor = json.load(f)
    SGDRegressor_RMSE_Validation = json_dict_SGDRegressor['RMSE_validation']
    RMSE_validations.append(SGDRegressor_RMSE_Validation)
    with open("models/performance_metrics/DecisionTreeRegressor/performance_metrics.json", mode='r') as f:
        json_dict_DecisionTreeRegressor = json.load(f)
    DecisionTreeRegressor_RMSE_Validation = json_dict_DecisionTreeRegressor['RMSE_validation']
    RMSE_validations.append(DecisionTreeRegressor_RMSE_Validation)
    with open("models/performance_metrics/RandomForestRegressor/performance_metrics.json", mode='r') as f:
        json_dict_RandomForestRegressor = json.load(f)
    RandomForestRegressor_RMSE_Validation = json_dict_RandomForestRegressor['RMSE_validation']
    RMSE_validations.append(RandomForestRegressor_RMSE_Validation)
    with open("models/performance_metrics/GradientBoostingRegressor/performance_metrics.json", mode='r') as f:
        json_dict_GradientBoostingRegressor = json.load(f)
    GradientBoostingRegressor_RMSE_Validation = json_dict_GradientBoostingRegressor['RMSE_validation']
    RMSE_validations.append(GradientBoostingRegressor_RMSE_Validation)
    best_model = min(RMSE_validations)
    if best_model == RMSE_validations[0]:
        model = joblib.load("models/regression/SGDRegressor/model.joblib", mmap_mode=None)
        with open("models/performance_metrics/SGDRegressor/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        return (model,performance_metrics)
    elif best_model == RMSE_validations[1]:
        model = joblib.load("models/regression/DecisionTreeRegressor/model.joblib", mmap_mode=None)
        with open("models/performance_metrics/DecisionTreeRegressor/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        return (model,performance_metrics)
    elif best_model == RMSE_validations[2]:
        model = joblib.load("models/regression/RandomForestRegressor/model.joblib", mmap_mode=None)
        with open("models/performance_metrics/RandomForestRegressor/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        return (model,performance_metrics)
    elif best_model == RMSE_validations[2]:
        model = joblib.load("models/regression/GradientBoostingRegressor/model.joblib", mmap_mode=None)
        with open("models/performance_metrics/GradientBoostingRegressor/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        return (model,performance_metrics)
    
if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    n_samples, n_features = 830, 9
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
    #evaluate_all_models(X_test,X_train,X_validation,y_test,y_train,y_validation)
    best_model = find_best_model()
    print(best_model)
 




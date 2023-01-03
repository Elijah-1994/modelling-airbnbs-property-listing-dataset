#%%.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from tabular_data import load_airbnb
import joblib
import json
import numpy as np
import pandas as pd
import json
import warnings
import os
np.random.seed(2)
warnings.filterwarnings('ignore')

def evaluate_all_models():
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(SGDRegressor(),X,y,X_test,y_test,{
    #     'learning_rate':['constant','optimal','invscaling','adaptive'],
    #     'penalty':['l2', 'l1', 'elasticnet','None'],
    #     'alpha':[0.0001,0.0002,0.0003],
    #     'l1_ratio':[0.15,0.1,0.25], 
    #     'max_iter':[1000,1250,1500,1750,2000], 
    #     'tol':[0.001,0.02,0.003],
    #     'epsilon':[0.1,0.2,0.3,0.5,0.9],
    #     'eta0':[0.01,0.02,0.03,0.05,0.09],
    #     'power_t':[0.25,0.35,0.45],
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/linear_regression/')
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(DecisionTreeRegressor(),X,y,X_test,y_test,{
        "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        "splitter":["best","random"],
        "max_depth" : [1,3,5,7,9,11,12],
        "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
        "min_samples_split": [0,1,2,3],
        "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
        "max_features":[1.0,"log2","sqrt",None],
        "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]
        ""
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/decision_tree/')
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(RandomForestRegressor(),X,y,X_test,y_test,{
        "n_estimators" : [5,20,50,100],
        "max_features":[1.0,"log2","sqrt",None],
        "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
        "max_depth" : [1,3,5,7,9,11,12],
        "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
        "bootstrap" : [True, False]
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/random_forest/')

def tune_regression_model_hyperparameters(model,X,y,X_test,y_test,*parameters):
    clf = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    n_jobs=-1,
    verbose=1,
                    )
    clf.fit(X,y)
    predictions = clf.predict(X_test)
    loss = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    performance_metrics = {'loss':loss,'r2':r2}
    return (clf,clf.best_params_,performance_metrics)
    
    # elif model_name == 'RandomForestRegressor()':
    #     params = {
    #         "n_estimators" : [5,20,50,100],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "bootstrap" : [True, False]
    
    #                                                           }
    #     clf = GridSearchCV(
    #     estimator=model,
    #     scoring='neg_mean_squared_error',
    #     param_grid=params,
    #     cv=10,
    #     n_jobs=5,
    #     verbose=1
    #                 )
    #     clf.fit(X_train, y_train)
    #     clf.best_params_
    #     return clf.best_params_
    
    # elif model_name == 'DecisionTreeRegressor()':
    #     params = {
    #         "splitter":["best","random"],
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
    #                                                           }
    #     clf = GridSearchCV(
    #     estimator=model,
    #     scoring='neg_mean_squared_error',
    #     param_grid=params,
    #     cv=5,
    #     n_jobs=5,
    #     verbose=1
    #                 )
    #     clf.fit(X_train, y_train)
    #     clf.best_params_
    #     return clf.best_params_

    # elif model_name == 'RandomForestRegressor()':
    #     params = {
    #         "criterion" : ['gini', 'entropy', 'log_loss'],
    #         "n_estimators" : [5,20,50,100],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "bootstrap" : [True, False]
    
    #                                                           }
    #     clf = GridSearchCV(
    #     estimator=model,
    #     scoring='neg_mean_squared_error',
    #     param_grid=params,
    #     cv=10,
    #     n_jobs=5,
    #     verbose=1
    #                 )
    #     clf.fit(X_train, y_train)
    #     clf.best_params_
    #     print(clf.best_params_)
    #     return clf.best_params_
    
    # elif model_name == 'GradientBoostingRegressor()':
    #     params = {
    #         "learning_rate" : [0.1,1.0,3.0,6.0,9.0,12.0],
    #         "n_estimators" : [5,20,50,100],
    #         "subsample": [0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_depth" : [1,3,5,7,9,11,12],         
    #         "tol":[0.0001,0.0002,0.0003]
    #                                                           }
    #     clf = GridSearchCV(
    #     estimator=model,
    #     scoring='neg_mean_squared_error',
    #     param_grid=params,
    #     cv=10,
    #     n_jobs=1,
    #     verbose=1
    #                 )
    #     clf.fit(X_train, y_train)
    #     clf.best_params_
    #     return clf.best_params_

def tune_classification_model_hyperparameters(model,X,y,X_test,y_test,*parameters):
        clf = GridSearchCV(
        estimator=model,
        scoring='precision',
        param_grid=parameters,
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit = True
                    )
        clf.fit(X,y)
        # model evaluation
        predictions = clf.predict(X_test)
        score = precision_score(y_test, predictions, average="macro")
        print (f"Printing the best params and score")
        print(clf.best_params_, score)
        return (clf,clf.best_params_,score)
    
  
    # elif model_name == 'RandomForestClassifier()':
    #     params = {
    #         "criterion":['gini', 'entropy', 'log_loss'],
    #         "n_estimators" : [5,20,50,100],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "bootstrap" : [True, False]
    
    # elif model_name == 'GradientBoostingClassifier()':
    #     params = {
    #         "learning_rate" : [0.1,1.0,3.0,6.0,9.0,12.0],
    #         "n_estimators" : [5,20,50,100],
    #         "subsample": [0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_depth" : [1,3,5,7,9,11,12],         
    #         "tol":[0.0001,0.0002,0.0003]
    #                                                           }
def save_model(model,best_parameters,performance_metrics,folder):
    parent_directory = folder
    print(parent_directory)
    model_path = 'models.joblib'
    model_path = os.path.join(parent_directory,model_path)
    joblib.dump(model, model_path )
    hyparemeters_path = 'hyperparameters.json' 
    hyparemeters_path = os.path.join(parent_directory,hyparemeters_path)
    performance_metrics_path = 'performance_metrics.json'
    performance_metrics_path = os.path.join(parent_directory,performance_metrics_path)
    with open(hyparemeters_path, mode="w", encoding= "utf-8") as file:
           file.write(json.dumps((best_parameters), default=str))   
    with open(performance_metrics_path, mode="w", encoding= "utf-8") as file:
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
    evaluate_all_models()

    # y = y.to_frame().reset_index(drop=True)
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # tune_classification_model_hyperparameters(LogisticRegression(),X,y,X_test,y_test,
    # {
    #         'penalty':['l2','none'],
    #         'tol':[0.0001,0.0002,0.0003],
    #         'C':[0.1,0.5,1.0,1.5],
    #         'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
    #         'max_iter':[50,75,100,125,150,175,200],
    
    #         }
    # tune_classification_model_hyperparameters(LogisticRegression(),X,y,X_test,y_test,
    # {
    #         'penalty':['l2','none'],
    #         'tol':[0.0001,0.0002,0.0003],
    #         'C':[0.1,0.5,1.0,1.5],
    #         'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
    #         'max_iter':[50,75,100,125,150,175,200],
    
    #         })
                                                              #)
    # tune_classification_model_hyperparameters(DecisionTreeClassifier(),X,y,X_test,y_test,
    #         {
    #         "splitter":["best","random"],
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "ccp_alpha":[0,1,2,3,4]
    #         }
    #y_train_prediction = model.predict(X_train)
    # y_pred = model.predict(X_test)
    # y_pred = model.predict(X_train)
    # accuracy = accuracy_score(y_train, y_pred)
    # precision= precision_score(y_train, y_pred, average="macro")
    # recall = recall_score(y_train, y_pred, average="macro")
    # F1 = f1_score(y_train, y_pred, average="macro")
    # performance_metrics = {'Accuracy':accuracy,'precision':precision,'recall':recall,
    #                            'F1':F1}
    # save_model(model,model_name,best_parameters,performance_metrics)
    #evaluate_all_models(X_test,X_train,X_validation,y_test,y_train,y_validation,task_folder='models/classification')
    # n_samples, n_features = 830, 9
    # rng = np.random.RandomState(0)
    # X = rng.randn(n_samples, n_features)
    # y = rng.randn(n_samples)
    # X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
    # print(y_train.shape)
    # print(X_train.shape)
    #evaluate_all_models(X_test,X_train,X_validation,y_test,y_train,y_validation)
    #best_model = find_best_model()
    #print(best_model)
# %%

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
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(DecisionTreeRegressor(),X,y,X_test,y_test,{
    #     "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    #     "splitter":["best","random"],
    #     "max_depth" : [1,3,5,7,9,11,12],
    #     "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #     "min_samples_split": [0,1,2,3],
    #     "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #     "max_features":[1.0,"log2","sqrt",None],
    #     "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90],
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/decision_tree/')
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(RandomForestRegressor(),X,y,X_test,y_test,{
    #     "min_samples_split": [0,1,2,3],
    #     "n_estimators" : [5,20,50,100],
    #     "max_features":[1.0,"log2","sqrt",None],
    #     "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #     "max_depth" : [1,3,5,7,9,11,12],
    #     "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #     "bootstrap" : [True, False],
        
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/random_forest/')
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(GradientBoostingRegressor(),X,y,X_test,y_test,{
    #      "max_depth" : [1,3,5,7,9,11,12],
    #      "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #      "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #      "max_features":[1.0,"log2","sqrt",None],
    #      "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/gradient_boosting/')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(LogisticRegression(),X,y,X_test,y_test,{
    #        'penalty':['l2','none'],
    #        'tol':[0.0001,0.0002,0.0003],
    #        'C':[0.1,0.5,1.0,1.5],
    #        'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
    #        'max_iter':[50,75,100,125,150,175,200],
    
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/logistic_regression/')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(DecisionTreeClassifier(),X,y,X_test,y_test,{
    #         "splitter":["best","random"],
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "ccp_alpha":[0,1,2,3,4]
    
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/decision_tree/')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(RandomForestClassifier(),X,y,X_test,y_test,{
    #         "max_depth" : [1,3,5,7,9,11,12],
    #         "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    #         "max_features":[1.0,"log2","sqrt",None],
    #         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
    #         "ccp_alpha":[0,1,2,3,4]
    
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/random_forest/')
    
    model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(GradientBoostingClassifier(),X,y,X_test,y_test,{
            "learning_rate" : [0.1,1.0,3.0,6.0,9.0,12.0],
            "n_estimators" : [5,20,50,100],
            "subsample": [0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "max_depth" : [1,3,5,7,9,11,12],         
            "tol":[0.0001,0.0002,0.0003]
    
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/classification/gradient_boosting/')
    
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
    
def tune_classification_model_hyperparameters(model,X,y,X_test,y_test,*parameters):
        clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=5,
        n_jobs=-1,
        verbose=1,
                    )
        clf.fit(X,y)
        # model evaluation
        predictions = clf.predict(X_test)
        performance_metrics = precision_score(y_test, predictions, average="macro")
        return (clf,clf.best_params_,performance_metrics)
    
                                                       
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

def find_best_model():
    loss =[]
    with open("models/regression/linear_regression/performance_metrics.json", mode='r') as f:
        json_dict_linear_regression = json.load(f)
    linear_regression_loss = json_dict_linear_regression['loss']
    loss.append(linear_regression_loss)
    with open("models/regression/decision_tree/performance_metrics.json", mode='r') as f:
        json_dict_decision_tree = json.load(f)
    decision_tree_loss = json_dict_decision_tree['loss']
    loss.append(decision_tree_loss)
    with open("models/regression/random_forest/performance_metrics.json", mode='r') as f:
        json_dict_random_forest = json.load(f)
    random_forest_loss = json_dict_random_forest['loss']
    loss.append(random_forest_loss)
    with open("models/regression/gradient_boosting/performance_metrics.json", mode='r') as f:
        json_dict_gradient_boosting = json.load(f)
    gradient_boosting_loss = json_dict_gradient_boosting['loss']
    loss.append(gradient_boosting_loss)
    best_model = min(loss)
    if best_model == loss[0]:
        model = joblib.load("models/regression/linear_regression/models.joblib", mmap_mode=None)
        with open("models/regression/linear_regression/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        with open("models/regression/linear_regression/hyperparameters.json", mode='r') as f:
            hyperparameters = json.load(f)
        return (model,hyperparameters,performance_metrics)
    elif best_model == loss[1]:
        model = joblib.load("models/regression/decision_tree/models.joblib", mmap_mode=None)
        with open("models/regression/decision_tree/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        with open("models/regression/decision_tree/hyperparameters.json", mode='r') as f:
            hyperparameters = json.load(f)
        return (model,hyperparameters,performance_metrics)
    elif best_model == loss[2]:
        model = joblib.load("models/regression/random_forest/models.joblib", mmap_mode=None)
        with open("models/regression/random_forest/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        with open("models/regression/random_forest/hyperparameters.json", mode='r') as f:
            hyperparameters = json.load(f)
        return (model,hyperparameters,performance_metrics)
    elif best_model == loss[3]:
        model = joblib.load("models/regression/gradient_boosting/models.joblib", mmap_mode=None)
        with open("models/regression/gradient_boosting/performance_metrics.json", mode='r') as f:
            performance_metrics= json.load(f)
        with open("models/regression/gradient_boosting/hyperparameters.json", mode='r') as f:
            hyperparameters = json.load(f)
        return (model,hyperparameters,performance_metrics)
    
if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    # n_samples, n_features = 830, 9
    # rng = np.random.RandomState(0)
    # X = rng.randn(n_samples, n_features)
    # y = rng.randn(n_samples)
    # X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    y = y.to_frame().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate_all_models()
    # best_model = find_best_model ()
    # print(best_model


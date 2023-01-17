#%%.
from datetime import datetime 
from collections import OrderedDict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from tabular_data import load_airbnb
from torch import nn
from torchmetrics import R2Score
from torch.utils.data import Dataset
import joblib
import json
import numpy as np
import pandas as pd
import json
import os
import time
import torch
import torch.nn.functional as F
import warnings
import yaml
np.random.seed(2)
warnings.filterwarnings('ignore')

def evaluate_all_models():
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(SGDRegressor(),X,y,X_test,y_test,{
        'learning_rate':['constant','optimal','invscaling','adaptive'],
        'penalty':['l2', 'l1', 'elasticnet','None'],
        'alpha':[0.0001,0.0002,0.0003],
        'l1_ratio':[0.15,0.1,0.25], 
        'max_iter':[1000,1250,1500,1750,2000], 
        'tol':[0.001,0.02,0.003],
        'epsilon':[0.1,0.2,0.3,0.5,0.9],
        'eta0':[0.01,0.02,0.03,0.05,0.09],
        'power_t':[0.25,0.35,0.45],
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/linear_regression/')
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(DecisionTreeRegressor(),X,y,X_test,y_test,{
        "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        "splitter":["best","random"],
        "max_depth" : [1,3,5,7,9,11,12],
        "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
        "min_samples_split": [0,1,2,3],
        "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
        "max_features":[1.0,"log2","sqrt",None],
        "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90],
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/decision_tree/')
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(RandomForestRegressor(),X,y,X_test,y_test,{
        "min_samples_split": [0,1,2,3],
        "n_estimators" : [5,20,50,100],
        "max_features":[1.0,"log2","sqrt",None],
        "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
        "max_depth" : [1,3,5,7,9,11,12],
        "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
        "bootstrap" : [True, False],
        
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/random_forest/')
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(GradientBoostingRegressor(),X,y,X_test,y_test,{
         "max_depth" : [1,3,5,7,9,11,12],
         "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
         "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
         "max_features":[1.0,"log2","sqrt",None],
         "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/gradient_boosting/')
    
    model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(LogisticRegression(),X,y,X_test,y_test,{
           'penalty':['l2','none'],
           'tol':[0.0001,0.0002,0.0003],
           'C':[0.1,0.5,1.0,1.5],
           'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
           'max_iter':[50,75,100,125,150,175,200],
    
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/classification/logistic_regression/')
    
    model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(DecisionTreeClassifier(),X,y,X_test,y_test,{
            "splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "max_features":[1.0,"log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
            "ccp_alpha":[0,1,2,3,4]
    
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/classification/decision_tree/')
    
    model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(RandomForestClassifier(),X,y,X_test,y_test,{
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
            "max_features":[1.0,"log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90], 
            "ccp_alpha":[0,1,2,3,4]
    
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/classification/random_forest/')
    
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
    performance_metrics = mean_squared_error(y_test, predictions, squared=False)
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
        predictions = clf.predict(X_test)
        performance_metrics = precision_score(y_test, predictions, average="macro")
        return (clf,clf.best_params_,performance_metrics)
    
                                                       
def save_model(model,best_parameters,performance_metrics,folder,module):
    if module == 'PyTorch':
        parent_directory = folder
        time_stamp = datetime.now()
        model_directory = os.path.join(parent_directory, time_stamp.strftime('%Y-%m-%d_%H-%M-%S')+'/')
        model_directory_folder = os.makedirs(model_directory, exist_ok = True)
        model_path = 'model.pt'
        model_path = os.path.join(model_directory,model_path)
        model = torch.save(model,model_path)
        hyparemeters_path = 'hyperparameters.json' 
        hyparemeters_path = os.path.join(model_directory,hyparemeters_path)
        performance_metrics_path = 'performance_metrics.json'
        performance_metrics_path = os.path.join(model_directory,performance_metrics_path)
        with open(hyparemeters_path, mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((best_parameters), default=str))   
        with open(performance_metrics_path, mode="w", encoding= "utf-8") as file:
            file.write(json.dumps((performance_metrics), default=str))  
    
    elif module == 'sklearn':
        parent_directory = folder
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

def find_best_model(task_folder):
    parent_directory =  task_folder
    linear_regression_path = 'linear_regression/performance_metrics.json'
    linear_regression_path = os.path.join(parent_directory,linear_regression_path)
    decision_tree_path = 'decision_tree/performance_metrics.json'
    decision_tree_path = os.path.join(parent_directory,decision_tree_path)
    random_forest_path = 'random_forest/performance_metrics.json'
    random_forest_path = os.path.join(parent_directory,random_forest_path)
    gradient_boosting_path = 'gradient_boosting/performance_metrics.json'
    gradient_boosting_path = os.path.join(parent_directory,gradient_boosting_path)
    regression_performance_metrics_list = [linear_regression_path,decision_tree_path,random_forest_path,gradient_boosting_path]
    best_model = 0
    for path in regression_performance_metrics_list:
        with open(path,mode='r') as f:
            hyperparameters = json.load(f)
            if hyperparameters < best_model or best_model == 0:
                best_model = hyperparameters

    print(f'{regression_performance_metrics_list[3][18:35]}' " " 'model has the lowest RMSE.')  
    best_model_path = 'gradient_boosting/models.joblib' 
    best_model_path = os.path.join(parent_directory,best_model_path)
    best_model = joblib.load(best_model_path, mmap_mode=None)
    best_model_hyperparameters = 'gradient_boosting/hyperparameters.json' 
    best_model_hyperparameters = os.path.join(parent_directory,best_model_hyperparameters)
    with open(gradient_boosting_path, mode='r') as f:
        best_performance_metrics= json.load(f)
    with open(best_model_hyperparameters, mode='r') as f:
        best_model_hyperparameters= json.load(f)    
    return(best_model,best_performance_metrics,best_model_hyperparameters)
    
    classification_performance_metrics_list = [logistic_regression_path,decision_tree_path,random_forest_path,gradient_boosting_path]
    best_model = 0
    for path in classification_performance_metrics_list:
        with open(path,mode='r') as f:
            hyperparameters = json.load(f)
            if hyperparameters > best_model:
                best_model = hyperparameters

    print(f'{classification_performance_metrics_list[2][22:35]}' " " 'model has the highest precision.')  
    best_model_path = 'random_forest/models.joblib' 
    best_model_path = os.path.join(parent_directory,best_model_path)
    best_model = joblib.load(best_model_path, mmap_mode=None)
    best_model_hyperparameters = 'random_forest/hyperparameters.json' 
    best_model_hyperparameters = os.path.join(parent_directory,best_model_hyperparameters)
    with open(random_forest_path, mode='r') as f:
        best_performance_metrics= json.load(f)
    with open(best_model_hyperparameters, mode='r') as f:
        best_model_hyperparameters= json.load(f)    
    return(best_model,best_performance_metrics,best_model_hyperparameters)
## nn
class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X,self.y = load_airbnb(df)
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy()
        
    
    def __getitem__(self,idx):
        return (torch.tensor(self.X[idx]),torch.tensor(self.y[idx]))
    
    def __len__(self):
        return len(self.X)
class NeuralNetwork(nn.Module):
    def __init__(self,config,):
        super().__init__()
        self.in_feature = 16
        self.out_feature = config[0]['hidden_layer_width']
        self.linear_input_layer = torch.nn.Linear(9,16)
        self.linear_output_layer = torch.nn.Linear(5,1)
        self.ordered_dict = OrderedDict({'linear_input_layer':self.linear_input_layer,'ReLU':torch.nn.ReLU()})
        self.depth = config[0]['depth']
        self.hidden_layer_depth = [x for x in range(self.depth)]
        for hidden_layer in self.hidden_layer_depth:
            self.ordered_dict['hidden_layer_'+str(hidden_layer)] = torch.nn.Linear(self.in_feature,self.out_feature)
            self.ordered_dict['ReLU_'+str(hidden_layer)] = torch.nn.ReLU()
        self.ordered_dict['linear_output_layer'] = self.linear_output_layer
        self.layers = torch.nn.Sequential(self.ordered_dict)
        print('ordered_dict',self.ordered_dict)
        print('depth',self.depth)
    
    def forward(self,X):
        return self.layers(X)
        
def train(model,config,epochs=10,):
    average_time =[]
    RMSE_Train = []
    R2_Train = []
    RMSE_Validation = []
    R2_Validation = []
    start_time = datetime.now()
    if config['Optimizer'] == 'SGD':
        print(config['Optimizer'])
        optimiser = torch.optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])
    elif config['Optimizer'] == 'ADADELTA':
        print(config['Optimizer'])
        optimiser = torch.optim.Adadelta(model.parameters(),lr=config['lr'], rho=config['rho'],
                                         eps=config['eps'],weight_decay=config['weight_decay'])
    elif config['Optimizer'] == 'ADAM':
        print(config['Optimizer'])
        optimiser = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=config['eps'],weight_decay=config['weight_decay'],
                                         amsgrad=config['amsgrad'],foreach=config['foreach'], maximize=config['maximize'], fused=config['fused']

                                         )
    elif config['Optimizer'] == 'ADAGRAD':
        print(config['Optimizer'])
        optimiser = torch.optim.Adagrad(model.parameters(),lr=config['lr'],lr_decay=config['lr_decay'], eps=config['eps'],weight_decay=config['weight_decay'],
                                        foreach=config['foreach'], maximize=config['maximize'],

                                         )
    batch_idx=0
    for epoch in range(epochs):
        for batch in dataloader["test"]:
            features,labels = batch
            features = features.to(torch.float32)
            features = features.reshape(BATCH_SIZE, -1)
            labels = labels.to(torch.float32)
            labels = labels.view(5,1)
            optimiser.zero_grad()
            current_time = time.process_time()
            prediction=model(features)
            print('prediction',prediction)
            end_time = time.process_time()
            time_elapsed = (end_time - current_time) * 10**3
            average_time.append(time_elapsed)
            mse = F.mse_loss(prediction,labels)
            RMSE_Loss_Train = torch.sqrt(mse)
            mse.backward()
            optimiser.step()
            RMSE_Train.append(RMSE_Loss_Train.item())
            r2score = R2Score()
            r2score = r2score(prediction,labels) 
            R2_Train.append(r2score.item())
            
        for batch in dataloader["validation"]:
            features,labels = batch
            features = features.to(torch.float32)
            features = features.reshape(BATCH_SIZE, -1)
            labels = labels.to(torch.float32)
            labels = labels.view(5,1)
            prediction=model(features)
            mse = F.mse_loss(prediction,labels)
            RMSE_Loss_Validation = torch.sqrt(mse)
            RMSE_Validation.append(RMSE_Loss_Validation.item())
            r2score = R2Score()
            r2score =r2score(prediction,labels) 
            R2_Validation.append(r2score.item())
            batch_idx+=1
        print('epoch',epoch)
            
    model = model.state_dict()
    best_parameters  = config
    RMSE_Loss_Train = RMSE_Train[-1]
    #print('RMSE_LOSS_Train',RMSE_Loss_Train)
    R2_Score_Train = R2_Train[-1]
    RMSE_Loss_Validation = RMSE_Validation[-1]
    R2_Score_Validation = R2_Validation[-1]
    training_duration = datetime.now() - start_time
    inference_latency = sum(average_time)/len(average_time)
    performance_metrics ={'RMSE_Loss_Train':RMSE_Loss_Train,'R2_Score_Train':R2_Score_Train,'RMSE_Loss_Validation':RMSE_Loss_Validation,
                          'R2_Score_Train':R2_Score_Validation, 'training_duration_(H,M,S)':training_duration,'inference_latency_(ms)':inference_latency,
                           }
    save_model(model,best_parameters,performance_metrics,folder='models/neural_networks/regression/scenario_1/', module = 'PyTorch')
    
def get_nn_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config_dict= yaml.safe_load(f)
        return config_dict
    
def find_best_nn():
    config = get_nn_config('nn_config.yaml')
    for configuration in config:
        train(model,config=configuration)

if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    # X,y = load_airbnb(df)
    # #regression model
    # n_samples, n_features = 830, 9
    # rng = np.random.RandomState(0)
    # X = rng.randn(n_samples, n_features)
    # y = rng.randn(n_samples)
    # X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    # #classification model
    # y = y.to_frame().reset_index(drop=True)
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # evaluate_all_models()
    # best_model = find_best_model (task_folder='models/classification/')
    #neural network model
    dataset= AirbnbNightlyPriceImageDataset()
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset,[500, 165,165]) 
    BATCH_SIZE = 5
    dataloader = {
    "train": torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers = 8,),
    "validation": torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        pin_memory=torch.cuda.is_available(),
        num_workers = 8),
    "test": torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        pin_memory=torch.cuda.is_available(), 
        num_workers = 8),
    }
    configuration =  get_nn_config('nn_config.yaml')
    model = NeuralNetwork(config=configuration)
    find_best_nn()

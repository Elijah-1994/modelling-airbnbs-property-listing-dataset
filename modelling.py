from collections import OrderedDict
from datetime import datetime 
import matplotlib.pyplot as plt
import scikitplot as skplt
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

# sklearn Regression and Classification model train loop.

def evaluate_all_models():
    '''
        This function calls the tune_regression_model_hyperparameters() function for regression models and
        the tune_classification_model_hyperparameters() for classification models. The function also calls the save_model function.

    '''   
    
    model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(SGDRegressor(),X,y,X_test,y_test,{
        'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
        'penalty':['l2', 'l1', 'elasticnet', 'None'],
        'alpha':[0.0001, 0.0002, 0.0003],
        'l1_ratio':[0.15, 0.1, 0.25], 
        'max_iter':[1000, 1250, 1500, 1750, 2000], 
        'tol':[0.001, 0.02, 0.003],
        'epsilon':[0.1, 0.2, 0.3, 0.5, 0.9],
        'eta0':[0.01, 0.02, 0.03, 0.05, 0.09],
        'power_t':[0.25, 0.35, 0.45],
                                    })
    save_model(model,best_parameters,performance_metrics,folder='models/regression/linear_regression/', module='SKlearn')
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(DecisionTreeRegressor(),X,y,X_test,y_test,{
    #     "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    #     "splitter":["best", "random"],
    #     "max_depth" : [1, 3, 5, 7, 9, 11, 12],
    #     "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "min_samples_split": [0, 1, 2, 3],
    #     "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #     "max_features":[1.0, "log2", "sqrt", None],
    #     "max_leaf_nodes":[None, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/decision_tree/', module='SKlearn')
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(RandomForestRegressor(),X,y,X_test,y_test,{
    #     "min_samples_split": [0, 1, 2, 3],
    #     "n_estimators" : [5, 20, 50, 100],
    #     "max_features":[1.0, "log2", "sqrt", None],
    #     "max_leaf_nodes":[None, 10, 20, 30, 40, 50, 60, 70, 80, 90], 
    #     "max_depth" : [1, 3, 5, 7, 9, 11, 12],
    #     "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #     "bootstrap" : [True, False],
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/random_forest/', module='SKlearn')
    
    # model,best_parameters,performance_metrics = tune_regression_model_hyperparameters(GradientBoostingRegressor(),X,y,X_test,y_test,{
    #      "max_depth" : [1, 3, 5, 7, 9, 11, 12],
    #      "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #      "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #      "max_features":[1.0, "log2", "sqrt", None],
    #      "max_leaf_nodes":[None, 10, 20, 30, 40, 50, 60, 70, 80, 90] 
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/regression/gradient_boosting/', module='SKlearn')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(LogisticRegression(),X,y,X_test,y_test,{
    #        'penalty':['l2', 'none'],
    #        'tol':[0.0001, 0.0002, 0.0003],
    #        'C':[0.1, 0.5, 1.0, 1.5],
    #        'solver':['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
    #        'max_iter':[50, 75, 100, 125, 150, 175, 200],
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/logistic_regression/')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(DecisionTreeClassifier(),X,y,X_test,y_test,{
    #         "splitter":["best", "random"],
    #         "max_depth" : [1, 3, 5, 7, 9, 11, 12],
    #         "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #         "max_features":[1.0, "log2", "sqrt", None],
    #         "max_leaf_nodes":[None, 10, 20, 30, 40, 50, 60, 70, 80, 90], 
    #         "ccp_alpha":[0, 1, 2, 3, 4]
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/decision_tree/', module='SKlearn')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(RandomForestClassifier(),X,y,X_test,y_test,{
    #         "max_depth" : [1, 3, 5, 7, 9, 11, 12],
    #         "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #         "max_features":[1.0, "log2", "sqrt", None],
    #         "max_leaf_nodes":[None, 10, 20, 30, 40, 50, 60, 70, 80, 90], 
    #         "ccp_alpha":[0, 1, 2, 3, 4]
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/random_forest/')
    
    # model,best_parameters,performance_metrics = tune_classification_model_hyperparameters(GradientBoostingClassifier(),X,y,X_test,y_test,{
    #         "learning_rate" : [0.1, 1.0, 3.0, 6.0, 9.0, 12.0],
    #         "n_estimators" : [5, 20, 50, 100],
    #         "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #         "min_weight_fraction_leaf":[0.1, 0.2, 0.3, 0.4],
    #         "max_depth" : [1, 3, 5, 7, 9, 11, 12],         
    #         "tol":[0.0001, 0.0002, 0.0003]
    #                                 })
    # save_model(model,best_parameters,performance_metrics,folder='models/classification/gradient_boosting/')
    
def tune_regression_model_hyperparameters(model,X,y,X_test,y_test,*parameters) -> tuple:
    '''
        This function is used to call the GridSearchCV class from sklearn to tune the hyperparameters for the regression models.
        The function fits the regression model then makes a prediction and calculates the performance metrics.

        Returns:
            tuple: returns the  best model its hyperparameters and performance metrics.
            
    '''
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
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, predictions, label="predicted")
    plt.title("Airbnb property listing - test and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    
    return (clf,clf.best_params_,performance_metrics)
    
def tune_classification_model_hyperparameters(model,X,y,X_test,y_test,*parameters) -> tuple:
    '''
        This function is used to call the GridSearchCV class from sklearn to tune the hyperparameters for the classification models.
        The function fits the classification model then makes a prediction and calculates the performance metrics.

        Returns:
            tuple: returns the  best model its hyperparameters and performance metrics.
            
    '''
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
    
# Neural network model 
class AirbnbNightlyPriceDataset(Dataset):
    '''
        This class represents how the data for the neural network is converted into PyTorch Tensors
        
        Attributes:
            self.X,self.y: loads the features and label data from the pandas data frame.
            self.X: Features data converted into numpy arrays.
            self.y: Label data converted into numpy arrays.
                
    '''
    def __init__(self):
        '''
        See help(AirbnbNightlyPriceDataset) for accurate signature

        '''
        super().__init__()
        self.X,self.y = load_airbnb(df)
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy()
        
    def __getitem__(self,idx):
        '''
        This method loads and returns a sample from the dataset at the given index (idx).

        Returns:
            tuple: returns features and label as Pytorch tensors.
            
        '''
        return (torch.tensor(self.X[idx]),torch.tensor(self.y[idx]))
    
    def __len__(self):
        '''
        This method returns the number of samples in the dataset.

        Returns:
            Int: The number of samples in the dataset.
            
        '''
        
        return len(self.X)
        
class NeuralNetwork(nn.Module):
    '''
        This class represents how the neural network layers and configured passes and passes the features data
        into the neural network.
        
        Attributes:
            self.in_feature: in feature of the hidden layers.
            self.out_feature: out feature of the hidden layers.
            self.linear_input_layer: input linear layer.
            self.linear_output_layer: output linear layer.
            self.depth: depth of the hidden layers.
            self.X: Features data converted into numpy arrays.
            self.y: Label data converted into numpy arrays.
                
    '''
    def __init__(self, config):
        '''
        See help(NeuralNetwork) for accurate signature.

        '''
        
        super().__init__()
        self.in_feature = 5
        self.out_feature = config[0]['hidden_layer_width']
        self.linear_input_layer = torch.nn.Linear(9,5)
        self.linear_output_layer = torch.nn.Linear(5,1)
        self.ordered_dict = OrderedDict({'linear_input_layer':self.linear_input_layer,'ReLU':torch.nn.ReLU()})
        self.depth = config[0]['depth']
        self.hidden_layer_depth = [x for x in range(self.depth)]
        for hidden_layer in self.hidden_layer_depth:
            self.ordered_dict['hidden_layer_'+ str(hidden_layer)] = torch.nn.Linear(self.in_feature,self.out_feature)
            self.ordered_dict['ReLU_'+ str(hidden_layer)] = torch.nn.ReLU()
        self.ordered_dict['linear_output_layer'] = self.linear_output_layer
        self.layers = torch.nn.Sequential(self.ordered_dict)
        print('ordered_dict',self.ordered_dict)
        print('depth',self.depth)
    
    def forward(self,X):
        '''
        This method passes the features data into neural network.

        Returns:
            Pytorch Tensor
            
        '''
        return self.layers(X)
    
def get_nn_config(yaml_file) -> dict:
    '''
        This function loads the configuration yaml file and returns the configuration in dictionary format.

        Returns:
            dictionary: returns dictionary of the neural network configurations.
            
    '''
    
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        return config_dict
    
def find_best_nn():
    ''' function loads the dictionary for each configuration and passes it as an argument in the the train function.
            
    '''
    config = get_nn_config('nn_config.yaml')
    for configuration in config:
        train(model,config=configuration)
        
def train(model,config,epochs=10,):
    '''
        This function provides the loop to train the neural network based on the optimizer and its parameters, get a prediction,
        and calculate the performance metrics of the trained model.

    '''   
    average_time = []
    RMSE_Train = []
    R2_Train = []
    RMSE_Validation = []
    R2_Validation = []
    start_time = datetime.now()
    batch_idx = 0
    if config['Optimizer'] == 'SGD':
        print(config['Optimizer'])
        optimiser = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif config['Optimizer'] == 'ADADELTA':
        print(config['Optimizer'])
        optimiser = torch.optim.Adadelta(model.parameters(),lr=config['lr'], rho=config['rho'], 
                                         eps=config['eps'], weight_decay=config['weight_decay'])
    elif config['Optimizer'] == 'ADAM':
        print(config['Optimizer'])
        optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=config['eps'], weight_decay=config['weight_decay'],
                                         amsgrad=config['amsgrad'], foreach=config['foreach'], maximize=config['maximize'], fused=config['fused']

                                         )
    elif config['Optimizer'] == 'ADAGRAD':
        print(config['Optimizer'])
        optimiser = torch.optim.Adagrad(model.parameters(), lr=config['lr'], lr_decay=config['lr_decay'], eps=config['eps'], weight_decay=config['weight_decay'],
                                        foreach=config['foreach'], maximize=config['maximize'],

                                         )
    for epoch in range(epochs):
        for batch in dataloader["test"]:
            features,labels = batch
            features = features.to(torch.float32)
            features = features.reshape(BATCH_SIZE, -1)
            labels = labels.to(torch.float32)
            labels = labels.view(5,1)
            optimiser.zero_grad()
            current_time = time.process_time()
            prediction = model(features)
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
            prediction = model(features)
            mse = F.mse_loss(prediction,labels)
            RMSE_Loss_Validation = torch.sqrt(mse)
            RMSE_Validation.append(RMSE_Loss_Validation.item())
            r2score = R2Score()
            r2score = r2score(prediction,labels) 
            R2_Validation.append(r2score.item())
            batch_idx+=1
        print('epoch',epoch)
            
    model = model.state_dict()
    best_parameters  = config
    RMSE_Loss_Train = RMSE_Train[-1]
    R2_Score_Train = R2_Train[-1]
    RMSE_Loss_Validation = RMSE_Validation[-1]
    R2_Score_Validation = R2_Validation[-1]
    training_duration = datetime.now() - start_time
    inference_latency = sum(average_time)/len(average_time)
    performance_metrics ={'RMSE_Loss_Train':RMSE_Loss_Train, 'R2_Score_Train':R2_Score_Train, 'RMSE_Loss_Validation':RMSE_Loss_Validation,
                          'R2_Score_Train':R2_Score_Validation, 'training_duration_(H,M,S)':training_duration, 'inference_latency_(ms)':inference_latency,
                           }
    save_model(model,best_parameters,performance_metrics,folder='models/neural_networks/regression/scenario_2/', module='PyTorch')
    

# Save and find best (neural,regression,classification) model code
                                                       
def save_model(model,best_parameters,performance_metrics,folder,module):
    '''
        This function checks if the model is a sklearn or Pytorch model and saves the model along with its hyperparameters/parameters
        and performance metrics.

    '''   
    if module == 'PyTorch':
        parent_directory = folder
        time_stamp = datetime.now()
        model_directory = os.path.join(parent_directory, time_stamp.strftime('%Y-%m-%d_%H-%M-%S')+'/')
        model_directory_folder = os.makedirs(model_directory, exist_ok=True)
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
    
    elif module == 'SKlearn':
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

def find_best_model(task_folder, module):
    '''
        This function loads the performace metrics of the neural network, classification and regression models and
        finds the best model for each scenario.
        
    '''   
    if module == 'Pytorch':
        parent_directory =  task_folder
        model_1_ADELTA_path = '2023-01-17_10-30-07/performance_metrics.json' 
        model_1_ADELTA_path = os.path.join(parent_directory,model_1_ADELTA_path)
        model_2_ADELTA_path = '2023-01-17_10-37-20/performance_metrics.json' 
        model_2_ADELTA_path = os.path.join(parent_directory,model_2_ADELTA_path)
        model_3_ADELTA_path = '2023-01-17_10-44-31/performance_metrics.json' 
        model_3_ADELTA_path = os.path.join(parent_directory,model_3_ADELTA_path)
        model_4_ADELTA_path = '2023-01-17_10-51-41/performance_metrics.json' 
        model_4_ADELTA_path = os.path.join(parent_directory,model_4_ADELTA_path)
        model_5_SGD_path = '2023-01-17_10-58-51/performance_metrics.json' 
        model_5_SGD_path = os.path.join(parent_directory,model_5_SGD_path)
        model_6_SGD_path = '2023-01-17_11-06-07/performance_metrics.json' 
        model_6_SGD_path = os.path.join(parent_directory,model_6_SGD_path)
        model_7_SGD_path = '2023-01-17_11-13-22/performance_metrics.json' 
        model_7_SGD_path = os.path.join(parent_directory,model_7_SGD_path)
        model_8_SGD_path = '2023-01-17_11-20-34/performance_metrics.json' 
        model_8_SGD_path = os.path.join(parent_directory,model_8_SGD_path)
        model_9_ADAM_path = '2023-01-17_11-27-46/performance_metrics.json' 
        model_9_ADAM_path = os.path.join(parent_directory,model_9_ADAM_path)
        model_10_ADAM_path = '2023-01-17_11-27-46/performance_metrics.json' 
        model_10_ADAM_path = os.path.join(parent_directory,model_10_ADAM_path)
        model_11_ADAM_path = '2023-01-17_11-27-46/performance_metrics.json' 
        model_11_ADAM_path = os.path.join(parent_directory,model_11_ADAM_path)
        model_12_ADAM_path = '2023-01-17_11-27-46/performance_metrics.json' 
        model_12_ADAM_path = os.path.join(parent_directory,model_12_ADAM_path)
        model_13_ADAGRAD_path = '2023-01-17_11-58-53/performance_metrics.json' 
        model_13_ADAGRAD_path = os.path.join(parent_directory,model_13_ADAGRAD_path)
        model_14_ADAGRAD_path = '2023-01-17_12-06-04/performance_metrics.json' 
        model_14_ADAGRAD_path = os.path.join(parent_directory,model_14_ADAGRAD_path)
        model_15_ADAGRAD_path = '2023-01-17_12-13-46/performance_metrics.json' 
        model_15_ADAGRAD_path = os.path.join(parent_directory,model_15_ADAGRAD_path)
        model_16_ADAGRAD_path = '2023-01-17_12-21-43/performance_metrics.json' 
        model_16_ADAGRAD_path = os.path.join(parent_directory,model_16_ADAGRAD_path)
        
        models_RMSE = []
        best_model = 0
        regression_performance_paths = [model_1_ADELTA_path, model_2_ADELTA_path, model_3_ADELTA_path, model_4_ADELTA_path,
                                               model_5_SGD_path, model_6_SGD_path, model_7_SGD_path, model_8_SGD_path, model_9_ADAM_path, 
                                               model_10_ADAM_path, model_11_ADAM_path, model_13_ADAGRAD_path, model_14_ADAGRAD_path,
                                               model_15_ADAGRAD_path, model_16_ADAGRAD_path
                                        ]
             
        for path in regression_performance_paths:
            with open(path,mode='r') as f:
                performance_metrics = json.load(f)
                models_RMSE.append(performance_metrics["RMSE_Loss_Validation"])
        
        for RMSE in models_RMSE:
            if RMSE < best_model or best_model == 0:
                best_model = RMSE
        print('model_8_SGD has the lowest RMSE.')
        
        best_model_path = '2023-01-17_11-20-34/model.pt' 
        best_model_path = os.path.join(parent_directory,best_model_path) 
        best_model = torch.load(best_model_path)
        best_model_hyperparameters = '2023-01-17_11-20-34/hyperparameters.json' 
        best_model_hyperparameters = os.path.join(parent_directory,best_model_hyperparameters)
        with open(gradient_boosting_path, mode='r') as f:
            best_performance_metrics= json.load(f)
        with open(best_model_hyperparameters, mode='r') as f:
            best_model_hyperparameters= json.load(f)    
        return(best_model,best_performance_metrics,best_model_hyperparameters)
    
    elif module == 'SKlearn':
        parent_directory =  task_folder
        linear_regression_path = 'linear_regression/performance_metrics.json'
        linear_regression_path = os.path.join(parent_directory,linear_regression_path)
        #logistic_regression_path = 'logistic_regression/performance_metrics.json'
        #logistic_regression_path = os.path.join(parent_directory,logistic_regression_path)
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
                performance_metric = json.load(f)
            if performance_metric < best_model or best_model == 0:
                best_model = performance_metric
                print('best_model',best_model)

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
                performance_metric = json.load(f)
                if performance_metric > best_model:
                    best_model = performance_metric

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

if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    X,y = load_airbnb(df)
    #regression model
    n_samples, n_features = 830, 9
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    X_train, X_test,y_train, y_test= train_test_split(X, y, test_size=0.3)
    evaluate_all_models()
    best_model = find_best_model(task_folder='models/regression/', module='SKlearn')
    # # classification model
    # y = y.to_frame().reset_index(drop=True)
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # evaluate_all_models()
    # best_model = find_best_model(task_folder='models/classification/', module='SKlearn')
    # #neural network regression model
    # dataset= AirbnbNightlyPriceDataset()
    # train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset,[500, 165,165]) 
    # BATCH_SIZE = 5
    # dataloader = {
    # "train": torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     pin_memory=torch.cuda.is_available(),
    #     num_workers = 8,),
    # "validation": torch.utils.data.DataLoader(
    #     validation_dataset, 
    #     batch_size=BATCH_SIZE, 
    #     pin_memory=torch.cuda.is_available(),
    #     num_workers = 8),
    # "test": torch.utils.data.DataLoader(
    #     test_dataset, 
    #     batch_size=BATCH_SIZE, 
    #     pin_memory=torch.cuda.is_available(), 
    #     num_workers = 8),
    # }
    # configuration =  get_nn_config('nn_config.yaml')
    # model = NeuralNetwork(config=configuration)
    # find_best_nn()
    # best_model = find_best_model (task_folder='models/neural_networks/regression/scenario_2/', module = 'Pytorch')

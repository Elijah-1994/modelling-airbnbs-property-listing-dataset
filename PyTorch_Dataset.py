#%%
from tabular_data import load_airbnb
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pandas as pd
import torch
import torch.nn.functional as F
df =  pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")

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
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,5),
            nn.ReLU(),
            torch.nn.Linear(5,1)
        )
        print('self.layers', self.layers)
    
    def forward(self,X):
        #use the layers to process the features
        return self.layers(X)

dataset= AirbnbNightlyPriceImageDataset()

train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset,[500, 165,165]) 

BATCH_SIZE = 5
dataloader = {
    "train": torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers = 8,
    ),
    "validation": torch.utils.data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available(),num_workers = 8
    ),
    "test": torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available(), num_workers = 8
    ),
}

def train(model, epochs=10):
    '''
        This function provides the loop to train the neural network based on the optimizer and its parameters, get a prediction,
        and calculate the performance metrics of the trained model.

    '''   
    optimiser = torch.optim.SGD(model.parameters(),lr=0.001)
    writer = SummaryWriter()
    batch_idx=0
    for epoch in range(epochs):
        for batch in dataloader["train"]:
            features,labels = batch
            features = features.to(torch.float32)
            features = features.reshape(BATCH_SIZE, -1)
            labels = labels.to(torch.float32)
            labels = labels.view(5,1)
            optimiser.zero_grad()
            prediction=model(features)
            print('prediction', prediction)
            loss = F.mse_loss(prediction,labels)
            loss.backward()
            optimiser.step()
            writer.add_scalar('loss',loss.item(),)
        for batch in dataloader["validation"]:
            features,labels = batch
            features = features.to(torch.float32)
            features = features.reshape(BATCH_SIZE, -1)
            labels = labels.to(torch.float32)
            labels = labels.view(5,1)
            prediction = model(features)
            mse_validation = F.mse_loss(prediction,labels)
            writer.add_scalar('validation_loss',mse_validation.item(),)
            batch_idx+=1

if __name__ == '__main__':
    model = NeuralNetwork()
    train(model)
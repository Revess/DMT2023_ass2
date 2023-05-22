import pandas as pd
import numpy as np
from preprocessing import calc_CTR
import torch
from torch import nn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import fire
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import coo_matrix
import turicreate as tc

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class simpleNNRegressor(nn.Module):
    def __init__(self, n_feat, hidden_layers=[]):
        super().__init__()
        modules = [nn.Linear(n_feat, hidden_layers[0]),nn.ReLU()]
        if len(hidden_layers) > 1:
            for i in range(len(hidden_layers)-1):
                modules.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_layers[-1],1))
        modules.append(nn.Sigmoid())
        self.main = nn.Sequential(*modules)
    
    def forward(self,x):
        return self.main(x)
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        super().__init__()
        self.features = torch.tensor(features).to(torch.float32)
        self.targets = torch.tensor(targets).to(torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target

def runNN(h_params={}, data=[]):
    batch_size = h_params["batch_size"]
    hidden_layers = h_params["hidden_layers"]
    epochs = h_params["epochs"] 
    lr = h_params["lr"]
    x_train, x_val, y_train, y_val = data

    train_set = Dataset(x_train, y_train)
    val_set = Dataset(x_val, y_val)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    model = simpleNNRegressor(x_train.shape[1], hidden_layers=hidden_layers)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        total_loss = np.array([])
        for i, (inputs, targets) in enumerate(trainloader):
            model.zero_grad()
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            pred = model(inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            total_loss = np.append(total_loss, loss.item())
            if i % 100 == 0 and i != 0:
                print(f'current_loss: {np.mean(total_loss)}', end="\r")
        print(f'Loss epoch {epoch}: {np.mean(total_loss)}')
        print("Validating")
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valloader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                loss = loss_fn(pred, targets)
                total_loss = np.append(total_loss, loss.item())
            print(f'val_loss: {np.mean(total_loss)}')
        model.train()
    torch.save(model, "./w1_0.ckpt")

def runFFM(training_params, data):
    actions = tc.SFrame.read_csv("./data/cleaned_training_set_VU_DM.csv")
    training_data, validation_data = tc.recommender.util.random_split_by_user(actions, 'srch_id', "prop_id")
    model = tc.recommender.create(training_data, 'srch_id', "prop_id", target='rating')
    model.summary()
    model.recommend()

def runKNN(data):
    x_train, x_val, y_train, y_val = data
    param_grid = {
        'n_neighbors': [3, 5, 7]
    }
    neigh = KNeighborsRegressor(50)
    neigh.fit(x_train, y_train)
    pred = neigh.predict(x_val)
    print(mean_squared_error(y_val,pred))
    # grid_search = GridSearchCV(estimator=neigh, param_grid=param_grid, cv=2, verbose=2, scoring='neg_mean_squared_error', n_jobs=-1)
    # grid_search.fit(x_train,y_train)
    # print(grid_search.best_estimator_, grid_search.best_score_)
    

def main(model = "NN", batch_size=128, hidden_layers=[20], epochs=4, lr=0.0001, reg_lambda=0.002):
    print("Loading Dataset")
    df = pd.read_csv("./data/cleaned_training_set_VU_DM.csv").drop(["prop_review_score",'srch_id', 'prop_id'],axis=1)
    df = (df - df.min()) / (df.max() - df.min()) #Normalize the data
    columns = df.drop('rating', axis=1).columns
    target = np.expand_dims(df['rating'].to_numpy(),-1)
    features = df.drop('rating', axis=1).to_numpy()

    # df = pd.read_csv("./data/training_set_VU_DM.csv")
    # print("Processing")
    # df = df.dropna(axis=1)
    # df['rating'] = pd.merge(
    #     df[['srch_id','click_bool', 'booking_bool']].groupby('srch_id').sum().reset_index().drop("booking_bool", axis=1).rename(columns={'click_bool':"summed_click_bool"}), 
    #     df[['srch_id','click_bool', 'booking_bool']], 
    #     how='left', 
    #     on= 'srch_id').apply(lambda row: calc_CTR(row['click_bool'], row['booking_bool'], row['summed_click_bool']) ,axis=1)
    # df = df.drop(['date_time', 'srch_id', 'site_id', 'prop_id'],axis=1)
    # df = (df - df.min()) / (df.max() - df.min()) #Normalize the data
    # target = df['rating'].to_numpy()
    # features = df.drop('rating', axis=1).to_numpy()
    # print(df.dtypes)

    data = train_test_split(features, target, test_size=0.1) #10% for validation

    print("Starting training")
    if model == "NN":
        runNN(
            {
                "batch_size": batch_size,
                "hidden_layers": hidden_layers,
                "epochs": epochs,
                "lr": lr
            },
            data
        )        
    elif model == 'FFM':
        runFFM({'epochs': epochs, 'reg_lambda': reg_lambda}, data)
    elif model == "KNN":
        runKNN(data)


if __name__ == "__main__":
    fire.Fire(main)

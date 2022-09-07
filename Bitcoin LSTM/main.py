from typing import Sequence

from numpy.random import shuffle
from pylab import rcParms
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib

import pandas as pd
import numpy as np


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict








#Pytorch Dataset
class BTCDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label = torch.tensor(label).float()
        )


#Price Data Module [Lightning]
class PriceDataModule(pl.LightningDataModule):
    def __init__(self, train_squences, test_squences, batch_size = 8):
        super().__init__()
        self.train_squences = train_squences
        self.test_squences = test_squences
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = BTCDataset(self.train_squences)
        self.test_dataset = BTCDataset(self.test_squences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.batch_size,
            shuffle=False,
            num_workers=8,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers = 8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers = 8
        )



#Hyper parameters for model
num_epochs = 8
batch_size = 64

data_module = PriceDataModule(train_squences, test_squences, batch_size=batch_size)
data_module.setup()

train_dataset = BTCDataset(train_sequences)

for item in train_dataset:
    print(item["sequence"].shape)
    print(item["label"].shape)
    print(item["label"])
    break





#price Prediction Model [Torch]
class PricePredictionModel(nn.Module):
    def __init__(self, n_features, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size = n_features,
            hdden_size = n_hidden,
            batch_first = True,
            num_layers = n_layers,
            dropout = 0.2,

        )

        self.regressor = nn.Linear(n_hidden, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]

        return self.regressor(out)

    


    
#Price Predictor [Lightning]
class PricePredictor(pl.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()

        self.model = PricePredictionModel(n_features)
        self.criterion = nn.MSELoss()

    def forward(self, x, lables = None):
        output = self.model(x)
        loss = 0
        if lables is not None:
            loss = self.criterion(output, lables.unsqueeze(dim =1))

        return loss, output

    ###PICK UP FROM HERE
    def training_step(self, batch, batch_inx):
        sequences = batch["sequence"]
        lables = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_inx):
        sequences = batch["sequence"]
        lables = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_inx):
        sequences = batch["sequence"]
        lables = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss        


    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.0001)




model = PricePredictionModel(n_features=train__df.shape[1])



data_loader = 


from tqdm import tqdm
 

def create_sequences(df, target_column, sequence_length):
    sequences = []

    data_size = len(df)
  
    for i in tqdm(range(data_size - sequence_length)):

        sequence = df[i:i+sequence_length]

        label_position = i + sequence_length

        label = df.iloc[label_position][target_column]

        sequences.append((sequence, label))

    return sequences
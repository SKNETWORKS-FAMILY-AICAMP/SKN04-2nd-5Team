from src.data import CPDataset, CPDataModule
from src.model.mlp import MLP
from src.training import CPModule
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.preprocessed_data import preprocessed_data

import pandas as pd
import numpy as np
import json

import nni
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(configs):
    # load dataset
    data = pd.read_csv('./data/train.csv')
    # preprocessing
    data = preprocessed_data(data)
    y = data['Churn']
    data = data.drop(columns=['Churn'])

    # train set, valid set split
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, y, test_size=0.4, shuffle=True
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True
    )

    standard_scaler_x = StandardScaler()
    X_train = standard_scaler_x.fit_transform(X_train)
    X_valid = standard_scaler_x.transform(X_valid)
    X_test = standard_scaler_x.transform(X_test)

    # convert to Dataset object
    train_dataset = CPDataset(X_train, y_train)
    valid_dataset = CPDataset(X_valid, y_valid)
    test_dataset = CPDataset(X_test, y_test)

    # create data module and prepare data
    cp_data_module = CPDataModule(batch_size=configs.get('batch_size'))
    cp_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    configs.update({'input_dim': len(data.columns)})
    # create model
    mlp = MLP(configs)

    # create LightningModule
    cp_module = CPModule(
        model=mlp,
        configs=configs
    )

    # create Trainer instance
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10) 
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'cp/test',
        ),        
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=cp_module,
        datamodule=cp_data_module
    )
    trainer.test(
        model=cp_module,
        datamodule=cp_data_module
    )

if __name__ == '__main__':

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

main(configs)
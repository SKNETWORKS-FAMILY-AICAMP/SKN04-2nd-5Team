from src.data import CPDataset, CPDataModule
from src.model.mlp import MLP
from src.training import CPModule
# from src.utils import convert_object_into_integer

import pandas as pd
import numpy as np
import json

import nni
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def convert_object_into_integer(df: pd.DataFrame):
    label_encoders = {}
    for column in df.columns:
        if df.dtypes[column] == object:
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
            label_encoders.update({column: label_encoder})
    
    return df, label_encoders


def main(configs):
    # load dataset
    data = pd.read_csv('/Users/macbook/Desktop/AI_edu/SKN04-2nd-5Team/mlp/data/train.csv')

    # NNI 하이퍼파라미터 업데이트
    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)

    # preprocessing
    data = data.dropna()
    data, _ = convert_object_into_integer(data)
    data = data.astype(np.float32)
    y = data['Churn']
    data = data.drop(columns=['Churn'])

    # train set, valid set split
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, y, test_size=0.6, shuffle=True
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

    # create model
    mlp = MLP(configs)

    # create LightningModule
    cp_module = CPModule(
        model=mlp,
        learning_rate = configs.get('learning_rate')
    )

    # create Trainer instance
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10) 
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'cp/{exp_name}',
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
    # NNI 최종 결과 보고
    if configs.get('nni'):
        nni.report_final_result(np.mean(cp_module.test_losses))

if __name__ == '__main__':

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    with open('/Users/macbook/Desktop/AI_edu/SKN04-2nd-5Team/mlp/configs.json', 'r') as file:
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
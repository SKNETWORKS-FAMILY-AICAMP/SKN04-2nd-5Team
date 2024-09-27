import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

import numpy as np

class CPDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).float()
        y = torch.Tensor([self.y.iloc[idx]]).float()
        
        return {
            'X': X,
            'y': y,
        }


class CPDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size  # 배치 크기를 저장

    def prepare(self, train_dataset, valid_dataset, test_dataset):
        # 데이터셋을 저장하는 메서드
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = self.train_dataset
            self.valid_data = self.valid_dataset

        if stage == "test":
            self.test_data = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

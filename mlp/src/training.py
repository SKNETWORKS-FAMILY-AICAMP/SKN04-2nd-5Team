import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import lightning as L

from sklearn.metrics import precision_score, recall_score, f1_score

class CPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.val_losses = []
        self.test_losses = []

        self.y_true = []
        self.y_pred = []

    def training_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        self.loss = F.binary_cross_entropy_with_logits(output, y)  # calculate loss

        y_pred = (output > 0.5).float()
        self.acc = (y_pred == y).float().mean()

        return self.loss

    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict({
            'loss/train_loss': self.loss,
            'acc/train_acc': self.acc,
            'learning_rate': self.learning_rate
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_losses.clear()
            self.y_true.clear()
            self.y_pred.clear()

        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        self.val_loss = F.binary_cross_entropy_with_logits(output, y)
        self.val_losses.append(self.val_loss)

        y_pred = (output > 0.5).float()
        self.val_acc = (y_pred == y).float().mean()

        self.y_true.extend(y)
        self.y_pred.extend(y_pred)

        return self.val_loss
    
    def on_validation_epoch_end(self):
        precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)

        self.log_dict({
            'loss/val_loss': np.mean(self.val_losses),
            'acc/val_acc': self.val_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'learning_rate': self.learning_rate
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_losses.clear()
            self.y_true.clear()
            self.y_pred.clear()

        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        self.test_loss = F.binary_cross_entropy_with_logits(output, y)
        self.test_losses.append(self.test_loss)

        y_pred = (output > 0.5).float()
        self.test_acc = (y_pred == y).float().mean()

        self.y_true.extend(y)
        self.y_pred.extend(y_pred)

        return self.test_loss
    
    def on_test_epoch_end(self):
        precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)

        self.log_dict({
            'loss/test_loss': np.mean(self.test_losses),
            'acc/test_acc': self.test_acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,            
            'learning_rate': self.learning_rate
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )    

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            # weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss/val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

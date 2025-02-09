import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import nni
import numpy as np

import lightning as L

from sklearn.metrics import precision_score, recall_score, f1_score

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report 출력
    report = classification_report(y_true, y_pred)
    print(report)

class CPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        configs,
    ):
        super().__init__()
        self.model = model
        self.configs = configs
        self.learning_rate = configs.get('learning_rate')

        self.val_losses = []
        self.test_losses = []

        self.y_true = []
        self.y_pred = []

    def training_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        logit = F.softmax(output, dim=-1)
        self.loss = F.cross_entropy(logit, y)  # calculate loss

        y_pred = logit.argmax(axis=-1)
        self.acc = (y_pred == y).float().mean()

        return self.loss

    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict({
            'loss/train_loss': self.loss,
            'acc/train_acc': self.acc
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
        logit = F.softmax(output, dim=-1)
        self.val_loss = F.cross_entropy(logit, y)  # calculate loss

        y_pred = logit.argmax(axis=-1)
        self.val_losses.append(self.val_loss)
        self.val_acc = (y_pred == y).float().mean()

        self.y_true.extend(y)
        self.y_pred.extend(y_pred)

        return self.val_loss
    
    def on_validation_epoch_end(self):
        precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)

        val_loss_mean = np.mean(self.val_losses)
        
        self.log_dict({
            'loss/val_loss':self.val_loss,
            'acc/val_acc': self.val_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'learning_rate': self.optimizers().param_groups[0].get('lr')
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # NNI 중간 결과 보고
        if hasattr(self, 'configs') and self.configs.get('nni'):
            nni.report_intermediate_result(val_loss_mean)  # 중간 결과로 validation loss를 보고


    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_losses.clear()
            self.y_true.clear()
            self.y_pred.clear()

        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        logit = F.softmax(output, dim=-1)
        self.test_loss = F.cross_entropy(logit, y) 
        self.test_losses.append(self.test_loss)
        y_pred = logit.argmax(axis=-1)
        
        self.test_acc = (y_pred == y).float().mean()

        self.y_true.extend(y)
        self.y_pred.extend(y_pred)

        return logit
    
    def on_test_epoch_end(self):
        precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)

        test_loss_mean = np.mean(self.test_losses)
        # Test 종료 시점에서만 혼동 행렬 시각화
        # plot_confusion_matrix(self.y_true, self.y_pred)
        print(classification_report(self.y_true, self.y_pred))
        # NNI 최종 결과 보고
        if hasattr(self, 'configs') and self.configs.get('nni'):
            nni.report_final_result(test_loss_mean)  # 최종 결과로 test loss를 보고

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
            'optimizer': optimizer,   # 옵티마이저 반환
            'scheduler': scheduler,   # 학습률 스케줄러 반환
        }

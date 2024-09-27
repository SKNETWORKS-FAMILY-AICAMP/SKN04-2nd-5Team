import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class CPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        output = torch.sigmoid(output)
        self.loss = F.binary_cross_entropy(output, y)  # calculate loss (MSE 손실 함수 사용)
        # with logits?

        y_pred = output.argmax(axis=-1)
        self.acc = (y_pred == y).float().mean()

        return self.loss  # 계산된 손실 반환
    
    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict(
            {'loss/train_loss': self.loss, 'acc/train_acc': self.acc},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def validation_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        output = torch.sigmoid(output)
        self.val_loss = F.binary_cross_entropy(output, y)
        # with logits?

        y_pred = output.argmax(axis=-1)
        self.val_acc = (y_pred == y).float().mean()

        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        self.log_dict(
            {'loss/val_loss': self.val_loss,
             'acc/val_acc': self.val_acc, 
             'learning_rate': self.learning_rate},
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')
        # y = y.squeeze()

        output = self.model(X)

        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

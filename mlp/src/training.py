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
        # output = torch.sigmoid(output)
        self.loss = F.binary_cross_entropy_with_logits(output, y)  # calculate loss

        y_pred = (output > 0.5).float()
        self.acc = (y_pred == y).float().mean()

        return {
            'loss': self.loss,
            'acc': self.acc
        }
    
    def on_train_epoch_end(self, *args, **kwargs):
        self.log_dict(
            {'loss/train_loss': self.loss,
             'acc/train_acc': self.acc},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def validation_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')

        output = self.model(X)
        # output = torch.sigmoid(output)
        self.val_loss = F.binary_cross_entropy_with_logits(output, y)
        # with logits?

        y_pred = (output > 0.5).float()
        self.val_acc = (y_pred == y).float().mean()

        return {
            'loss': self.val_loss,
            'acc': self.val_acc
        }
    
    def on_validation_epoch_end(self):
        self.log_dict(
            {'loss/val_loss': self.val_loss,
             'acc/val_acc': self.val_acc},
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        X = batch.get('X')
        y = batch.get('y')
        # y = y.squeeze()

        output = self.model(X)
        # output = torch.sigmoid(output)
        self.test_loss = F.binary_cross_entropy_with_logits(output, y)
        # with logits?

        y_pred = (output > 0.5).float()
        self.test_acc = (y_pred == y).float().mean()

        return {
            'loss': self.test_loss,
            'acc': self.test_acc
        }
    
    def on_test_epoch_end(self):
        self.log_dict(
            {'loss/test_loss': self.test_loss,
             'acc/test_acc': self.test_acc},
            prog_bar=True,
            logger=True,
        )    

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

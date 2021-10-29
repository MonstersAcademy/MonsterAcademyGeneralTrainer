import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from .metrics.wandbConfMat import WandbConfMat
import wandb


class ImagenetTransferLearning(LightningModule):
    def __init__(self, hparams={"None": 0}):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.get_metrics()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=7, factor=0.5, verbose=True)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_loss"}

    def get_metrics(self):
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.train_conf = WandbConfMat(
            classNames=self.hparams['classNames'])
        self.val_conf = WandbConfMat(
            classNames=self.hparams['classNames'])

    def training_step(self, batch, batch_idx):
        images, targets = batch

        logits = self.forward(images)
        loss = F.cross_entropy(logits, targets)
        with torch.no_grad():
            # acc = (torch.argmax(logits, dim=1) == labels).float().mean()
            self.train_accuracy(logits, targets)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        self.train_conf.update(preds[:, 0], targets)

        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        wandb_logger = self.get_wandb_logger()
        self.log('training accuracy', self.train_accuracy.compute())
        self.train_accuracy.reset()
        predictions, ground_truth, class_names = self.train_conf.compute()
        wandb_logger.log({
            "conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                    y_true=ground_truth,
                                                    preds=predictions,
                                                    class_names=class_names)})
        self.train_conf.reset()

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        logits = self.forward(images)
        loss = F.cross_entropy(logits, targets)
        with torch.no_grad():
            # acc = (torch.argmax(logits, dim=1) == labels).float().mean()
            self.validation_accuracy(logits, targets)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        self.val_conf.update(preds[:, 0], targets)

        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        wandb_logger = self.get_wandb_logger()
        self.log('validation accuracy', self.validation_accuracy.compute())
        self.validation_accuracy.reset()
        predictions, ground_truth, class_names = self.val_conf.compute()
        wandb_logger.log({
            "val_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=ground_truth,
                                                        preds=predictions,
                                                        class_names=class_names)})
        self.val_conf.reset()

        return super().training_epoch_end(outputs)

    def get_wandb_logger(self):
        """get_wandb_logger Helper function to retrive the wandb logger
        for the logger list.
        We differentiate it because only the wandb logger can log figures.
        Returns:
            wandb.logger: The wandb logger among all loggers
        """
        wandb = None
        for logger in self.logger.experiment:
            if 'wandb' in logger.__class__.__module__:
                wandb = logger
                break
        return wandb

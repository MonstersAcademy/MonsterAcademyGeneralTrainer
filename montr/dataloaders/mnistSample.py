from pytorch_lightning import LightningDataModule
import torchvision
import torchvision.transforms as transform
import torch


class MnistDataloader(LightningDataModule):
    def __init__(self,
                 hyperparams,):
        super().__init__()
        self.hyperparams = hyperparams

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=True,
                                       transform=transform.Compose([
                                           transform.ToTensor(),
                                           transform.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.hyperparams['batchSizeTrain'],
            shuffle=True,
            num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=False, download=True,
                                       transform=transform.Compose([
                                           transform.ToTensor(),
                                           transform.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.hyperparams['batchSizeTrain'],
            shuffle=True,
            num_workers=12)

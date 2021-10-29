from pytorch_lightning import LightningDataModule
from datasets.imageDataset import ImageDataset
from torch.utils.data import DataLoader
import os


class ImageDataloader(LightningDataModule):
    def __init__(self, imagesFolder,
                 hyperparams,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         test_transforms=test_transforms,
                         dims=dims)
        self.imagesFolder = imagesFolder
        self.hyperparams = hyperparams
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self):
        # TODO: add an exception if there is not set
        self.imagesTrain = ImageDataset(
            os.path.join(self.imagesFolder, 'Train'),
            transforms=self.train_transforms)
        self.imagesVal = ImageDataset(
            os.path.join(self.imagesFolder, 'Val'),
            transforms=self.val_transforms)
        self.imagesTest = ImageDataset(
            os.path.join(self.imagesFolder, 'Test'),
            transforms=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.imagesTrain,
                          batch_size=self.hyperparams['batchSize'])

    def val_dataloader(self):
        return DataLoader(self.imagesVal,
                          batch_size=self.hyperparams['batchSize'])

    def test_dataloader(self):
        return DataLoader(self.imagesTest,
                          batch_size=self.hyperparams['batchSize'])

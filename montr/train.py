from pytorch_lightning.loggers import WandbLogger, wandb
import yaml
import pytorch_lightning as pl
import torch
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from models.imageModels import ImagenetTransferLearning
from dataloaders.mnistSample import MnistDataloader
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def check_if_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def main(cfg):
    with open(cfg, 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    model = ImagenetTransferLearning(
        hparams=hparams)

    check_if_folder_exists(hparams['logsDir'])
    wandb_logger = WandbLogger(
        save_dir=hparams['logsDir'],
        project=hparams['projectName'],
        offline=False,
        log_model='all',
    )
    dataloader = MnistDataloader(hparams)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="validation accuracy",
        mode="max",
        save_top_k=3,
        verbose=True,
        dirpath=os.path.join(wandb_logger.experiment.dir, "checkpoints"),
    )
    wandb_logger.watch(model)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hparams["epochs"],
        # early_stop_callback=early_stop_callback,
        callbacks=[
            # early_stop_callback,
            checkpoint_callback,
            lr_monitor,
        ],
        logger=[wandb_logger],
        fast_dev_run=False,
        deterministic=True,
        stochastic_weight_avg=False,
        # overfit_batches=4,
    )
    trainer.fit(model, datamodule=dataloader)


if __name__ == "__main__":
    main("montr/configs/mnist.yaml")

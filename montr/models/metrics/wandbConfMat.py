import numpy as np
from torchmetrics.metric import Metric
import torch


class WandbConfMat(Metric):
    def __init__(self, classNames, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.totalPreds = []
        self.totalGroundTruth = []
        self.classNames = classNames

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states

        self.totalPreds.append(preds)
        self.totalGroundTruth.append(target)

    def compute(self):
        # compute final result
        self.totalPreds = torch.cat(self.totalPreds).to("cpu").numpy()
        self.totalGroundTruth = torch.cat(
            self.totalGroundTruth).to("cpu").numpy()
        return self.totalPreds, self.totalGroundTruth, self.classNames

    def reset(self) -> None:
        self.totalPreds = []
        self.totalGroundTruth = []
        return super().reset()

from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        n_epochs: int,
        n_batches: int,
    ):
        self.optimizer = optimizer
        self.model = model
        self.n_epochs = n_epochs
        self.n_batches = n_batches

    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        validation_data: Dict[str, torch.Tensor],
    ):

        raise NotImplementedError

        all_n_samples = np.array([train_data[key] for key in train_data.keys()])
        n_samples = all_n_samples[0]
        if not (n_samples == all_n_samples).all():
            raise ValueError(
                f"Elements in the input data contain inconsistent rows: {all_n_samples}"
            )

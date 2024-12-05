import numpy as np
import torch
import tqdm
from datasets import Dataset
from torch import nn
from torch.optim import Optimizer

from transformers_tutorial.utils import get_logger

logger = get_logger(__name__)

LABEL_KEY = "label"


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        loss,
        model: nn.Module,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.model = model

    def train(
        self,
        train_data: Dataset,
        validation_data: Dataset,
        n_epochs: int,
        batch_size: int,
        label_key: str = LABEL_KEY,
    ):

        all_n_samples = np.array(
            [len(train_data[key]) for key in train_data.features.keys()]
        )
        n_samples = all_n_samples[0]
        if not (n_samples == all_n_samples).all():
            raise ValueError(
                f"Elements in the input data contain inconsistent rows: {all_n_samples}"
            )

        device_ = self.model.device

        batches_per_epochs = n_samples // batch_size
        validation_data_ = validation_data[:]
        validation_labels_ = validation_data_[label_key].to(device_)

        for epoch in range(n_epochs):
            with tqdm.trange(batches_per_epochs, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for batch in bar:
                    indices_ = [
                        k for k in range(batch * batch_size, (batch + 1) * batch_size)
                    ]
                    batch_data = train_data.select(indices_)[:]
                    labels = batch_data[label_key].to(device_)

                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(batch_data), labels)
                    loss.backward()
                    self.optimizer.step()

            with torch.no_grad():
                loss_val = self.loss(
                    self.model(validation_data_),
                    validation_labels_,
                )
                logger.info(f"validation loss: {loss_val}")

        return self.model

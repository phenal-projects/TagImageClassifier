from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import Tensor, LongTensor
from torch.nn import functional as F
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        weight_decay: float = 1e-4,
        learning_rate: float = 0.001,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            "deit3_small_patch16_224_in21ft1k",
            drop_rate=0.2,
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        :param images: a batch of images. Shape: (batch_size x 3 x image_size x image_size)
        :return: probabilities for the image classes. Shape: (batch_size x classes_per_image)
        """
        return self.model(images)

    def single_step(self, batch, stage: str = "train") -> Tuple[Tensor, Tensor, Tensor]:
        images, y = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # accuracy
        acc = ((logits > 0) == y).float().mean()
        self.log(
            f"{stage}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss, acc, logits

    def epoch_end(self, outputs, stage: str = "train"):
        outputs = self.all_gather(outputs)
        logits = (
            torch.cat([output["logits"] for output in outputs])
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        targets = (
            torch.cat([output["targets"] for output in outputs])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        if len(np.unique(targets)) != 1:
            self.log(
                f"{stage}_roc_auc",
                torch.tensor(roc_auc_score(targets, logits), dtype=torch.float32),
            )
            self.log(
                f"{stage}_average_precision",
                torch.tensor(
                    average_precision_score(targets, logits), dtype=torch.float32
                ),
            )
            self.log(
                f"{stage}_target_balance",
                torch.tensor(np.mean(targets), dtype=torch.float32),
            )

    def training_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "train")
        return {"loss": loss, "logits": logits, "targets": batch[-1]}

    def validation_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "val")
        return {"logits": logits, "targets": batch[-1]}

    def test_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "test")
        return {"logits": logits, "targets": batch[-1]}

    def training_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs,
        )
        scheduler.step()  # skip 0 lr epoch
        return [optimizer], [scheduler]

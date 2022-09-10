from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import timm
from pytorch_lightning.loggers.base import DummyLogger
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import Tensor, LongTensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau



class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "deit3_small_patch16_224_in21ft1k",
        num_classes: int = 1000,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        learning_rate: float = 0.001,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name,
            drop_rate=dropout,
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
        images, y, pos_weight = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight[0])
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
            torch.cat([output["logits"] for output in outputs]).detach().cpu().numpy()
        )
        targets_matrix = (
            torch.cat([output["targets"] for output in outputs]).cpu().numpy()
        )
        scores_data = []
        if not isinstance(self.logger, DummyLogger):
            for i in range(targets_matrix.shape[1]):
                targets = targets_matrix[:, i]
                if len(np.unique(targets)) != 1:
                    scores_data.append(
                        (
                            i,
                            roc_auc_score(targets, logits[:, i]),
                            average_precision_score(targets, logits[:, i]),
                            np.mean(targets),
                        )
                    )

            self.logger.log_table(
                key=f"scores_{stage}",
                columns=["tag_idx", "roc_auc", "average_precision", "balance"],
                data=scores_data,
            )

    def training_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "val")
        return {"logits": logits, "targets": batch[-1]}

    def test_step(self, batch, batch_idx):
        loss, _, logits = self.single_step(batch, "test")
        return {"logits": logits, "targets": batch[-1]}

    def validation_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "test")

    def freeze_pretrained(self):
        for name, param in self.model.named_parameters():
            if "fc." in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,
            patience=2,
            cooldown=2,
            threshold=1e-4,
            min_lr=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }

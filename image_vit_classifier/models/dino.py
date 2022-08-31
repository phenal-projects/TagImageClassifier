import pytorch_lightning as pl
from torch import Tensor, LongTensor
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup
from vit_pytorch.vit_for_small_dataset import ViT
from vit_pytorch import Dino


class LightningDino(pl.LightningModule):
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
        self.image_encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.learner = Dino(
            self.image_encoder,
            image_size=image_size,
            hidden_layer="to_latent",
            projection_hidden_size=256,  # projector network hidden dimension
            projection_layers=4,  # number of layers in projection network
            num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
            student_temp=0.9,  # student temperature
            teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay=0.9,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay=0.9,  # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        :param images: a batch of images. Shape: (batch_size x 3 x image_size x image_size)
        :return: embeddings for the images. Shape: (batch_size x emb_size)
        """
        return self.image_encoder(images)

    def single_step(self, batch, stage: str = "train") -> Tensor:
        images, classes_idx, y = batch
        loss = self.learner(images)
        # loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def epoch_end(self, outputs, stage: str = "train"):
        if stage == "train":
            self.learner.update_moving_average()

    def training_step(self, batch, batch_idx):
        loss = self.single_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.single_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.single_step(batch, "test")

    def training_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.learner.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.learner.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
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

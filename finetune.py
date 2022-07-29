import hydra
import pandas as pd
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms

from image_vit_classifier.data import TaggedImageFolderDataModule
from image_vit_classifier.models import ImageClassifier


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg["random_seed"])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 180), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomResizedCrop(size=cfg["data"]["image_size"]),
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomInvert(p=0.5),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=7),
                ],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(size=cfg["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # initialize data module
    data = TaggedImageFolderDataModule(
        cfg["data"]["image_folder"],
        cfg["data"]["image_metadata_db"],
        train_image_transform=train_transform,
        val_image_transform=val_transform,
        min_tag_freq=cfg["data"]["minimal_tag_frequency"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        random_seed=cfg["random_seed"],
    )
    data.setup()

    # initialize the model
    wandb.init(project="ViTClfR34", job_type="Finetuning")
    artifact = wandb.run.use_artifact(cfg["model"]["finetuning_start"], type='model')
    artifact_dir = artifact.download()
    model = ImageClassifier.load_from_checkpoint(f"{artifact_dir}/model.ckpt").eval()
    model.hparams.weight_decay = cfg["model"]["weight_decay"]
    model.hparams.learning_rate = cfg["model"]["learning_rate"]

    # enable all grads
    for parameter in model.parameters():
        parameter.requires_grad = True

    trainer = pl.Trainer(
        default_root_dir=cfg["training"]["checkpoints_folder"],
        gpus=cfg["training"]["gpus"],
        max_epochs=cfg["training"]["max_epochs"],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="val_acc",
                save_top_k=4,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=WandbLogger(project="ViTClfR34", log_model="all"),
    )

    trainer.logger.log_text(
        key="tag2idx", dataframe=pd.DataFrame.from_dict({"idx": data.tag2idx}).reset_index()
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

import os
from glob import glob
from typing import Callable, Optional, Union

import pandas as pd
from .datasets import TaggedImages
from sqlalchemy import create_engine
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import logging


class TaggedImageFolderDataModule(LightningDataModule):
    def __init__(
        self,
        image_folder: Union[str, os.PathLike],
        image_metadata_db: str,
        train_image_transform: Optional[Callable] = None,
        val_image_transform: Optional[Callable] = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        min_tag_freq: int = 50,
        batch_size: int = 8,
        random_seed: int = 42,
        prefetch_factor: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tag2idx = None
        self.file2tags = None

    def setup(self, stage: Optional[str] = None):
        # read the db
        engine = create_engine(self.hparams.image_metadata_db)
        with engine.connect() as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT filename, tags  FROM images_metadata", conn
            )

        # read and split image paths
        filenames = set(df.filename)
        image_paths = sorted(
            [
                path
                for path in glob(os.path.join(self.hparams.image_folder, "*"))
                if path.split("/")[-1].split(".")[0] in filenames
            ]
        )

        train_set, vt_set = train_test_split(
            image_paths,
            test_size=self.hparams.test_size + self.hparams.val_size,
            random_state=self.hparams.random_seed,
        )
        val_set, test_set = train_test_split(
            vt_set,
            test_size=self.hparams.test_size
            / (self.hparams.test_size + self.hparams.val_size),
            random_state=self.hparams.random_seed,
        )
        train_set = set(train_set)
        val_set = set(val_set)
        test_set = set(test_set)

        self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(df.tags.explode().unique()))}

        df.index = df.filename
        self.file2tags = {
            path: [self.tag2idx[tag] for tag in tags]
            for path, tags in zip(
                image_paths,
                df.loc[
                    [path.split("/")[-1].split(".")[0] for path in image_paths], "tags"
                ],
            )
        }
        logging.info("Tags extracted")

        self.train_dataset = TaggedImages(
            self.hparams.image_folder,
            {k: v for k, v in self.file2tags.items() if k in train_set},
            len(self.tag2idx),
            self.hparams.train_image_transform,
        )
        self.val_dataset = TaggedImages(
            self.hparams.image_folder,
            {k: v for k, v in self.file2tags.items() if k in val_set},
            len(self.tag2idx),
            self.hparams.val_image_transform,
        )
        self.test_dataset = TaggedImages(
            self.hparams.image_folder,
            {k: v for k, v in self.file2tags.items() if k in test_set},
            len(self.tag2idx),
            self.hparams.val_image_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

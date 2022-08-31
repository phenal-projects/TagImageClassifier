import os
from typing import Union, Set, Dict, Any, Callable, Optional
from glob import glob


import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import drop_alpha


class TaggedImages(Dataset):
    def __init__(
        self,
        image_folder: Union[str, os.PathLike],
        file2tags: Dict[str, Set[int]],
        num_tags: int,
        transform: Optional[Callable[[Image], Any]] = None,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.file2tags = file2tags
        self.transform = transform

        self.images = sorted(
            [
                path
                for path in glob(os.path.join(self.image_folder, "*"))
                if path in self.file2tags
            ]
        )
        self.num_tags = num_tags

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]
        # load image
        image = Image.open(path)
        image = drop_alpha(image).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # sample tags
        tag_indices = self.file2tags[path]
        ys = torch.zeros((self.num_tags, ))
        for tag_idx in tag_indices:
            ys[tag_idx] = 1.0

        return (
            image,
            ys,
        )

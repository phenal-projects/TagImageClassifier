import os
from typing import Union, Set, Dict, Any, Callable, Optional
from glob import glob
from random import sample, randint

import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import drop_alpha


class TaggedImages(Dataset):
    def __init__(
        self,
        image_folder: Union[str, os.PathLike],
        file2tags: Dict[str, Set[int]],
        transform: Optional[Callable[[Image], Any]] = None,
        views_per_image: int = 4,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.file2tags = file2tags
        self.max_tag_id = max([max(x) for x in self.file2tags.values() if len(x) > 0])
        self.transform = transform
        self.views_per_image = views_per_image

        self.images = sorted(
            [
                path
                for path in glob(os.path.join(self.image_folder, "*"))
                if path in self.file2tags
            ]
        )

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
        tags = self.file2tags[path]
        if len(tags) > (self.views_per_image // 2):
            pos_tags = sample(tags, self.views_per_image // 2)
        else:
            pos_tags = tags

        neg_tags = [
            randint(0, self.max_tag_id)
            for _ in range(self.views_per_image - len(pos_tags))
        ]

        return (
            image,
            torch.LongTensor(pos_tags + neg_tags),
            torch.FloatTensor([1] * len(pos_tags) + [0] * len(neg_tags)),
        )

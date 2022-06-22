import torch, os, json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from typing import List, Optional
from pydantic import BaseModel

class TDA(BaseModel):
    t: int
    x: float
    y: float
    w: float
    h: float

    def scale(self, width: int, height: int):
        return (self.t, int(self.x * width), int(self.y * height), int(self.w * width), int(self.h * height))

    def as_list(self):
        return [
            self.t,
            self.x,
            self.y,
            self.w,
            self.h
        ]

    def as_tensor(self):
        return torch.tensor((
            self.t,
            self.x,
            self.y,
            self.w,
            self.h
        ), dtype=torch.float32)

class Annotation(BaseModel):
    from_image: Optional[str]
    to_image: Optional[str]
    data: List[TDA]

    def as_list(self):
        return [x.as_list() for x in self.data]

class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

class TDADataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        self.img_dir = os.path.join(filepath.replace(os.path.basename(filepath), ''), 'images')
        self.data = self._read_labels(filepath)
        self.length = sum([len(x.data) for x in self.data])

        self.t = ImageCompose([
            transforms.Resize((32, 24)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return self.length

    def _read_labels(self, path: str):
        with open(path, 'r') as f:
            db = json.loads(f.read())
            return [Annotation.parse_obj(x) for x in db]

    def _crop_object(self, im: Image, tda: TDA, is_to: bool = False):
        imw, imh = im.size
        _, x, y, w, h = tda.scale(imw, imw)
        
        # if is for a 'to' image, increase the size of the search box
        if is_to:
            w = int(h * 1.2)
            h = int(h * 1.2)

        # clamping values to fit to the image
        x1 = max(0, min(x-w, imh))
        x2 = max(0, min(x+w, imh))
        y1 = max(0, min(y-h, imw))
        y2 = max(0, min(y+h, imw))

        return im.crop((x1, y1, x2, y2))

    def __getitem__(self, index):
        # calculate true index
        i, j = 0, 0
        for idx, d in enumerate(self.data):
            if index < i + len(d.data):
                j = index - i
                index = idx
                break
            i += len(d.data)

        item = self.data[index]

        from_image = Image.open(os.path.join(self.img_dir, item.from_image))
        to_image = Image.open(os.path.join(self.img_dir, item.to_image))
        tda = item.data[j]

        frm = self._crop_object(from_image, tda)
        to = self._crop_object(to_image, tda, True)
        if self.t:
            frm = self.t(frm)
            to = self.t(to)

        # crops = []
        # for tda in item.data:
        #     frm = self._crop_object(from_image, tda)
        #     to = self._crop_object(to_image, tda, True)
        #     if self.t:
        #         frm = self.t(frm)
        #         to = self.t(to)

        #     crops.append((
        #         frm, to, tda.as_tensor()
        #     ))

        return frm, to, tda.as_tensor()

import os
from typing import List, Optional
from pydantic import BaseModel
from torch import tensor
from PIL import Image

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
        return tensor(self.as_list())

class WebAnnotation(BaseModel):
    from_image: Optional[str]
    to_image: Optional[str]
    data: List[TDA]

    def as_list(self):
        return [x.as_list() for x in self.data]


class Annotation(BaseModel):
    frm: str
    to: str
    tda: TDA

    def read(self, base_path: str):
        return Image.open(os.path.join(base_path, self.frm)), Image.open(os.path.join(base_path, self.to))
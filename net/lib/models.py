from typing import List, Optional
from pydantic import BaseModel
from torch import tensor

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

class Annotation(BaseModel):
    from_image: Optional[str]
    to_image: Optional[str]
    data: List[TDA]

    def as_list(self):
        return [x.as_list() for x in self.data]
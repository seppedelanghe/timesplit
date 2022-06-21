from typing import List, Optional
from pydantic import BaseModel



class TDA(BaseModel):
    t: int
    x: float
    y: float
    w: float
    h: float

class Annotation(BaseModel):
    from_image: Optional[str]
    to_image: Optional[str]
    data: List[TDA]
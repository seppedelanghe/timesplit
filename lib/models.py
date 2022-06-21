from pydantic import BaseModel



class TDA(BaseModel):
    t: int
    x: float
    y: float
    w: float
    h: float

class Annotation(BaseModel):
    im_uid: str
    tda: TDA
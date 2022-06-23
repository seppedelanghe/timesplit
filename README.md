# TimeSplit
TimeSplit is an attempt to build an object detector optimized for object tracking in chronological data.

## Before
- `pip install fastapi[all] tqdm`
- install pytorch => https://pytorch.org/

## Running annotator
`uvicorn annotator.main:app --reload`

## Train a model
- update env values in `train.py`
- `python net/train.py`


## Dataset

Custom dataset format for use with model.

```json
{
    "from": "from_image.jpg",
    "to": "to_image.jpg",
    "tda": {
        "t": 0.0,
        "x": 0.0,
        "y": 0.0,
        "w": 0.0,
        "h": 0.0,
    }
}
```
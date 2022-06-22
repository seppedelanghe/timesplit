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
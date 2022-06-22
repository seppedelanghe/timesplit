import os, json, re
from typing import List
from net.lib.models import Annotation
from fastapi import HTTPException, UploadFile

BASE_PATH = 'annotator/static/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
SAVE_PATH = os.path.join(DATA_PATH, 'images')
DB_PATH = os.path.join(DATA_PATH, 'db.json')
os.makedirs(SAVE_PATH, exist_ok=True)

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_db():
    if not os.path.isfile(DB_PATH):
        return []

    with open(DB_PATH, 'r') as f:
        db = json.loads(f.read())
        return [Annotation.parse_obj(x) for x in db]

def save_db(db: List[Annotation]):
    db = [x.dict() for x in db]

    with open(DB_PATH, 'w') as f:
        f.write(json.dumps(db))

    return True

def save_tda(annot: Annotation):
    db = load_db()  
    db.append(annot)
    save_db(db)

def upload_images(imgs: List[UploadFile]):
    for im in imgs:
        if os.path.isfile(os.path.join(SAVE_PATH, im.filename)):
            raise HTTPException(status_code=422, detail=f"file with name {im.filename} already exists!")

        with open(os.path.join(SAVE_PATH, im.filename), 'wb') as f:
            f.write(im.file.read())
    
    return True

def upload_labels(lbls: List[UploadFile]):
    for lbl in lbls:
        if os.path.isfile(os.path.join(SAVE_PATH, lbl.filename)):
            raise HTTPException(status_code=422, detail=f"file with name {lbl.filename} already exists!")

        with open(os.path.join(SAVE_PATH, lbl.filename), 'wb') as f:
            f.write(lbl.file.read())
    
    return True

def get_all():
    return {
        'images': sorted_nicely([file for file in os.listdir(SAVE_PATH) if file[-4:] == '.jpg']),
        'labels': [file for file in os.listdir(SAVE_PATH) if file[-4:] == '.txt'],
    }

def get_lbl_for_img(img: str):
    filename = img.replace('.jpg', '.txt')
    p = os.path.join(SAVE_PATH, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail='Label does not exists.')

    lbls = []
    with open(p, 'r') as f:
        for line in f.readlines():
            lbls.append([float(x) for x in line.strip().split(' ')])

    return lbls
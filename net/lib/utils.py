import os, json, re
from PIL import Image
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


def crop_object(im: Image, coords: tuple, is_to: bool = False):
    imw, imh = im.size
    x, y, w, h = int(coords[0] * imw), int(coords[1] * imh), int(coords[2] * imw), int(coords[3] * imh)

    # if is for a 'to' image, increase the size of the search box
    if is_to:
        w = int(h * 1.2)
        h = int(h * 1.2)

    # clamping values to fit to the image
    x1 = max(0, min(x-w, imw))
    x2 = max(0, min(x+w, imw))
    y1 = max(0, min(y-h, imh))
    y2 = max(0, min(y+h, imh))


    return im.crop((x1, y1, x2, y2))

def export_db():
    db = load_db()
    data = []
    CROP_DIR = os.path.join(SAVE_PATH, 'crops')
    os.makedirs(CROP_DIR, exist_ok=True)

    for x in db:
        original_lbls = get_lbl_for_img(x.from_image)
        for i, lbl in enumerate(original_lbls):
            if any([y == -1 for y in x.data[i].as_list()]):
                continue

            frm = crop_object(Image.open(os.path.join(SAVE_PATH, x.from_image)), lbl[1:])
            to = crop_object(Image.open(os.path.join(SAVE_PATH, x.to_image)), lbl[1:], True)
            frm.save(os.path.join(CROP_DIR, f"{i}_{x.from_image}"))
            to.save(os.path.join(CROP_DIR, f"{i}_{x.to_image}"))

            data.append({
                'from': f"{i}_{x.from_image}",
                'to': f"{i}_{x.to_image}",
                'tda': x.data[i].dict()
            })
        
    with open(os.path.join(DATA_PATH, 'export.json'), 'w') as f:
        f.write(json.dumps(data))

    return data
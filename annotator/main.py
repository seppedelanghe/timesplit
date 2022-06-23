from typing import List
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from net.lib.models import WebAnnotation
from net.lib.utils import export_db, save_tda, get_all, upload_images, upload_labels, get_lbl_for_img

app = FastAPI()
app.mount("/static", StaticFiles(directory="annotator/static"), name='static')

@app.get('/', response_class=HTMLResponse)
async def root():
    return RedirectResponse('/static/app.html', status_code=302)

@app.get('/all')
async def get():
    return get_all()

@app.get('/lbl')
async def annot(img: str):
    return get_lbl_for_img(img)

@app.post('/annot')
async def annot(annot: WebAnnotation):
    print(annot)
    save_tda(annot)
    return "OK"

@app.post('/img')
async def upload_img(files: List[UploadFile]):
    upload_images(files)
    return RedirectResponse('/', status_code=302)

@app.post('/lbl')
async def upload_label(files: List[UploadFile]):
    upload_labels(files)
    return RedirectResponse('/', status_code=302)

@app.get('/export')
async def export():
    try:
        export_db()
        return 'Export saved in data path'
    except:
        return 'Failed to export, please check logs.'
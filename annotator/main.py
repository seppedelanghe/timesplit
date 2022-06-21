from typing import List
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from lib.models import TDA
from lib.utils import save_tda, get_all, upload_images, upload_labels, get_lbl_for_img

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

@app.post('/annot/{uid}')
async def annot(tda: TDA, uid: str):
    save_tda(tda, uid)
    return "OK"

@app.post('/img')
async def upload_img(files: List[UploadFile]):
    upload_images(files)
    return RedirectResponse('/', status_code=302)

@app.post('/lbl')
async def upload_label(files: List[UploadFile]):
    upload_labels(files)
    return RedirectResponse('/', status_code=302)



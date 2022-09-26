import io
import os

import requests
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    data = {}
    return templates.TemplateResponse("page.html", {"request": request, "data": data})


@app.post("/upload")
async def upload(file: UploadFile):
    img_bytes: bytes = file.file.read()
    resp = requests.post(os.environ["FOREGROUND_URL"], files={"file": img_bytes})
    return StreamingResponse(io.BytesIO(resp.content), media_type="image/png")

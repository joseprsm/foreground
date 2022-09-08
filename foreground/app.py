from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from fastapi import FastAPI, UploadFile
from starlette.responses import FileResponse

from foreground.predict import inference

app = FastAPI()


@app.post("/")
def remove_bg(file: UploadFile):
    temp_input_dir = Path(mkdtemp())
    temp_output_dir = Path(mkdtemp())

    with open(temp_input_dir / "file.jpg", "wb") as f:
        f.write(file.file.read())

    no_bg = inference(temp_input_dir / "file.jpg")
    rmtree(temp_input_dir)

    no_bg.save(temp_output_dir / "no_bg.png")
    no_bg.close()

    return FileResponse(temp_output_dir / "no_bg.png")

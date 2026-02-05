from fastapi import FastAPI, UploadFile
import shutil
from pipelines.run_analysis import run_pipeline

app = FastAPI()

@app.post("/analyze")
def analyze(file: UploadFile):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = run_pipeline(path)
    return result

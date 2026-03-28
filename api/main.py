import shutil
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import RunCreateResponse, RunStatusResponse
from services.run_service import create_run, execute_run_by_id
from services.artifact_store import load_run_state, save_run_state

app = FastAPI(title="Autonomous Decision Intelligence API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/runs/upload", response_model=RunCreateResponse)
def upload_dataset(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    state = create_run(str(save_path))
    save_run_state(state.run_id, asdict(state))

    return RunCreateResponse(
        run_id=state.run_id,
        filename=state.filename,
        status=state.status,
        mode=state.mode,
    )


@app.post("/runs/{run_id}/start")
def start_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    file_path: str,
    target_override: str | None = None,
    mode: str = "deterministic",
):
    try:
        payload = load_run_state(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")

    payload["file_path"] = file_path
    payload["filename"] = Path(file_path).name
    payload["target_override"] = target_override
    payload["mode"] = mode
    payload["status"] = "running"
    payload["current_stage"] = "queued"
    save_run_state(run_id, payload)
    background_tasks.add_task(execute_run_by_id, run_id, file_path, target_override)

    return {
        "run_id": run_id,
        "status": "running",
        "current_stage": "queued",
        "errors": [],
    }


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str):
    try:
        payload = load_run_state(run_id)
        return RunStatusResponse(**payload)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")

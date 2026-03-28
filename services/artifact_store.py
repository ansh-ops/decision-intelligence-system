import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path("artifacts")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def save_run_state(run_id: str, payload: Dict[str, Any]) -> None:
    run_dir = BASE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "state.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_run_state(run_id: str) -> Dict[str, Any]:
    run_file = BASE_DIR / run_id / "state.json"
    if not run_file.exists():
        raise FileNotFoundError(f"No saved state found for run_id={run_id}")
    with open(run_file, "r") as f:
        return json.load(f)
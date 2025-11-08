# api/predict.py
import os
import tempfile
from pathlib import Path
import traceback

import requests
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="Symptoms Predict API")

# allow CORS (mobile/web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# CONFIG: public URLs you gave
# ---------------------------------------------------------------------
MODEL_URL = "https://github.com/Dilan-Rajitha/Symptoms_Predict_002/releases/download/v1.0.0/model_small.joblib"
DATA_URL = "https://github.com/Dilan-Rajitha/Symptoms_Predict_002/releases/download/v1.0.001/mega_symptom_dataset_500k.csv"

# local preferred paths (for local dev)
ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = ROOT / "models" / "model_small.joblib"
LOCAL_DATA_PATH = ROOT / "data" / "mega_symptom_dataset_500k.csv"

# tmp paths (for vercel / serverless)
TMP_DIR = Path(os.getenv("VERCEL_TMP_DIR") or tempfile.gettempdir())
TMP_MODEL_PATH = TMP_DIR / "model_small.joblib"
TMP_DATA_PATH = TMP_DIR / "mega_symptom_dataset_500k.csv"

# globals
PIPELINE = None
MLB = None
META = {}


def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    print(f"[DOWNLOAD] saved {dest} ({dest.stat().st_size} bytes)")


def ensure_model():
    """
    Make sure model file exists (local or tmp), then load into memory.
    """
    global PIPELINE, MLB, META

    if PIPELINE is not None and MLB is not None:
        return  # already loaded

    # 1) prefer local file (dev)
    if LOCAL_MODEL_PATH.exists():
        model_path = LOCAL_MODEL_PATH
    else:
        # 2) otherwise use tmp (serverless) and download if missing
        model_path = TMP_MODEL_PATH
        if not model_path.exists():
            _download_file(MODEL_URL, model_path)

    saved = joblib.load(model_path)
    PIPELINE = saved["pipeline"]
    MLB = saved["mlb"]
    META = saved.get("meta", {})
    print(f"[MODEL] loaded from {model_path}")


def ensure_dataset():
    """
    Download dataset if we need it.
    API doesn’t actually use it for prediction, but we expose a /dataset-info
    endpoint to verify it's there.
    """
    # local dev
    if LOCAL_DATA_PATH.exists():
        return LOCAL_DATA_PATH

    # serverless tmp
    if not TMP_DATA_PATH.exists():
        _download_file(DATA_URL, TMP_DATA_PATH)

    return TMP_DATA_PATH


# ---------------- request schema ----------------
class SymptomRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None


# ---------------- triage logic ----------------
def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}

    t0 = top[0]

    # high-risk list
    if t0["id"] in {"ami", "meningitis", "heatstroke", "dka", "stroke", "seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}

    # same-day urgent list
    if t0["id"] in {"appendicitis", "angina", "dengue_fever", "kidney_stones", "cholera", "typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}

    # low-confidence fallback
    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}

    # default moderate risk
    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}


# ---------------- endpoints ----------------
@app.get("/")
def health():
    # don't force-download dataset/model on health
    return {
        "ok": True,
        "message": "POST /ai/symptom-check",
        "model_source": MODEL_URL,
        "data_source": DATA_URL,
    }


@app.get("/dataset-info")
def dataset_info():
    path = ensure_dataset()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


@app.post("/ai/symptom-check")
def symptom_check(req: SymptomRequest):
    ensure_model()   # make sure we have model in memory

    proba = PIPELINE.predict_proba([req.text])[0]
    idxs = np.argsort(proba)[::-1][:3]

    top = []
    for i in idxs:
        label = str(MLB.classes_[i])
        p = float(proba[i])
        top.append({
            "id": label,
            "name": label.replace("_", " ").title(),
            "prob": p,
            "prob_pct": round(p * 100.0, 2),
        })

    tri = simple_triage(top)

    return {
        "top_conditions": top,
        "triage": tri,
        "input": {"text": req.text, "lang": req.lang},
        "disclaimer": "Educational aid; not a medical diagnosis."
    }

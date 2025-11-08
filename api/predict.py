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

# -------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------
# local dev path
LOCAL_MODEL = Path("models/model_small.joblib")

# vercel tmp dir (writable)
TMP_MODEL = Path(os.getenv("VERCEL_TMP_DIR") or tempfile.gettempdir()) / "model_small.joblib"

# if local file exists -> use it, else use tmp
MODEL_PATH = LOCAL_MODEL if LOCAL_MODEL.exists() else TMP_MODEL

# 🔴 your GitHub release URL
MODEL_URL = "https://github.com/Dilan-Rajitha/Symptoms_Predict_002/releases/download/v1.0.0/model_small.joblib"

PIPELINE = None
MLB = None
META = {}


def ensure_model():
    """
    Make sure model file is present (download if missing) and load into memory.
    This avoids shipping the big file inside the Vercel function (250 MB limit).
    """
    global PIPELINE, MLB, META

    # already loaded
    if PIPELINE is not None and MLB is not None:
        return

    # make sure dir exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # download if file is not present
    if not MODEL_PATH.exists():
        print(f"[MODEL] downloading from {MODEL_URL} → {MODEL_PATH}")
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        print("[MODEL] download complete")

    # load
    saved = joblib.load(MODEL_PATH)
    PIPELINE = saved["pipeline"]
    MLB = saved["mlb"]
    META = saved.get("meta", {})
    print("[MODEL] loaded OK")


# -------------------------------------------------
# Request schema
# -------------------------------------------------
class SymptomRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None


# -------------------------------------------------
# Triage logic (same as before)
# -------------------------------------------------
def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}

    t0 = top[0]

    # life-threatening shortlist
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


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def health():
    # don't force download here, so health is fast
    return {
        "ok": True,
        "message": "POST /ai/symptom-check",
        "model_path": str(MODEL_PATH),
        "download_url": MODEL_URL,
        "meta": META,
    }


@app.post("/ai/symptom-check")
def symptom_check(req: SymptomRequest):
    try:
        ensure_model()
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
    except Exception as e:
        # helpful for Vercel logs
        return {
            "error": str(e),
            "trace": traceback.format_exc(),
        }

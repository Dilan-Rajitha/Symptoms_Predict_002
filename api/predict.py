# api/predict.py
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Symptoms Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- load model ----------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_small.joblib"   # 👈 new name

print(f"[INIT] loading model from {MODEL_PATH}")
_saved = joblib.load(MODEL_PATH)
PIPELINE = _saved["pipeline"]
MLB = _saved["mlb"]
META = _saved.get("meta", {})
print("[INIT] model loaded OK")

# ---------- request schema ----------
class SymptomRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None

# ---------- triage ----------
def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}

    t0 = top[0]

    if t0["id"] in {"ami", "meningitis", "heatstroke", "dka", "stroke", "seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}

    if t0["id"] in {"appendicitis", "angina", "dengue_fever", "kidney_stones", "cholera", "typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}

    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}

    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}

# ---------- endpoints ----------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "POST /ai/symptom-check",
        "meta": META,
        # just to show you've got the dataset in repo:
        "has_dataset": (ROOT / "data" / "mega_symptom_dataset_500k.csv").exists(),
    }

@app.post("/ai/symptom-check")
def symptom_check(req: SymptomRequest):
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

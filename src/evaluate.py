# src/evaluate.py
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mega_symptom_dataset_500k.csv"
DEFAULT_MODEL = ROOT / "models" / "model_small.joblib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate on first N rows (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["labels"] = df["labels"].apply(lambda s: s.split("|"))

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    saved = joblib.load(args.model)
    pipe = saved["pipeline"]
    mlb = saved["mlb"]

    texts = df["text"].tolist()
    y_true = mlb.transform(df["labels"])

    proba = pipe.predict_proba(texts)
    top1_idx = proba.argmax(axis=1)

    correct = int(sum(y_true[i, top1_idx[i]] == 1 for i in range(len(texts))))
    acc = correct / len(texts)

    print(f"Top-1 accuracy: {acc*100:.2f}% on {len(texts)} samples")

    # show first 3 predictions
    for i in range(min(3, len(texts))):
        preds = mlb.classes_[top1_idx[i]]
        print(f"• {texts[i][:80]} → {preds}")


if __name__ == "__main__":
    main()

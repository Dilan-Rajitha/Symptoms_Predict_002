# src/train.py
from pathlib import Path
import argparse
import json
import time
import platform

import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mega_symptom_dataset_500k.csv"
DEFAULT_OUT = ROOT / "models" / "model_small.joblib"


def build_pipeline(ngram_low, ngram_high, min_df, max_features, alpha):
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_low, ngram_high),
        min_df=min_df,
        max_features=max_features,
    )
    clf = OneVsRestClassifier(MultinomialNB(alpha=alpha))
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                    help="path to mega_symptom_dataset_500k.csv")
    ap.add_argument("--model_out", type=str, default=str(DEFAULT_OUT),
                    help="where to save model_small.joblib")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # vectorizer knobs
    ap.add_argument("--ngram_low", type=int, default=3)
    ap.add_argument("--ngram_high", type=int, default=5)
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=1.5)

    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    df["labels"] = df["labels"].apply(lambda s: s.split("|"))

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values,
        df["labels"].values,
        test_size=args.val_split,
        random_state=args.seed,
        shuffle=True,
    )

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_train)
    Y_val = mlb.transform(y_val)

    pipe = build_pipeline(
        args.ngram_low,
        args.ngram_high,
        args.min_df,
        args.max_features,
        args.alpha,
    )

    pipe.fit(X_train, Y_train)

    # quick top-1 accuracy on val
    proba = pipe.predict_proba(X_val)
    top1_idx = proba.argmax(axis=1)
    correct = int(sum(Y_val[i, top1_idx[i]] == 1 for i in range(len(X_val))))
    acc = correct / len(X_val)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path.resolve()),
        "n_samples": int(len(df)),
        "val_split": args.val_split,
        "top1_val_acc": round(acc, 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "tfidf": {
            "ngram_range": [args.ngram_low, args.ngram_high],
            "min_df": args.min_df,
            "max_features": args.max_features,
        },
        "nb": {
            "alpha": args.alpha
        },
    }

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # strong compression to keep file smaller
    joblib.dump(
        {"pipeline": pipe, "mlb": mlb, "meta": meta},
        out_path,
        compress=("xz", 9),
    )

    print(f"[OK] saved model -> {out_path}")
    print(f"[OK] top-1 val acc: {acc*100:.2f}% on {len(X_val)} samples")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

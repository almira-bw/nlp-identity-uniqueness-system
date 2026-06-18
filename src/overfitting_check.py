"""
Overfitting gate for address classifier.
Plots learning curve and checks train-val F1 gap <= 0.10.
Must pass before unlocking the test set.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

from src.clustering import address_pair_features
from src.preprocessing import normalize_address


def run_overfitting_gate(labeled_path: str) -> bool:
    """
    Plot learning curve for address classifier.
    Returns True (pass) if train-val F1 gap <= 0.10 at full training size.
    """
    clf = joblib.load("models/address_classifier.pkl")
    vec = joblib.load("models/address_tfidf.pkl")

    pairs = pd.read_csv(labeled_path)
    pairs = pairs[pairs["label"].isin(["SAME", "DIFFERENT"])].copy()
    pairs["a1"] = pairs["address_1_raw"].apply(normalize_address)
    pairs["a2"] = pairs["address_2_raw"].apply(normalize_address)
    feats = [address_pair_features(r.a1, r.a2, vec) for _, r in pairs.iterrows()]
    X = pd.DataFrame(feats)
    y = (pairs["label"] == "SAME").astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, tr_scores, val_scores = learning_curve(
        clf, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv, scoring="f1", n_jobs=-1,
        shuffle=True, random_state=42,
    )

    tr_m  = tr_scores.mean(axis=1)
    val_m = val_scores.mean(axis=1)
    gap   = float(tr_m[-1] - val_m[-1])

    plt.figure(figsize=(8, 4))
    plt.plot(sizes, tr_m,  "o-", label="Train F1")
    plt.plot(sizes, val_m, "o-", label="Val F1")
    status = "✓ OK" if gap <= 0.10 else "⚠ OVERFIT — lower logreg_C or add data"
    plt.title(f"Learning Curve  |  gap={gap:.3f}  {status}")
    plt.xlabel("Training pairs")
    plt.ylabel("F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/learning_curve.png", dpi=120)
    plt.close()

    print(f"\nOverfitting Gate:")
    print(f"  Train F1 @ full : {tr_m[-1]:.3f}")
    print(f"  Val   F1 @ full : {val_m[-1]:.3f}")
    print(f"  Gap             : {gap:.3f}  (threshold: 0.10)")
    result = gap <= 0.10
    print(f"  Result: {'✓ PASS — proceed to test set' if result else '✗ FAIL — fix before test'}")
    return result


if __name__ == "__main__":
    passed = run_overfitting_gate("data/labeled/address_pairs_to_label.csv")
    if not passed:
        raise SystemExit("Overfitting gate FAILED. Do not unlock the test set.")

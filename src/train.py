"""
Train address pair classifier.
Fits TF-IDF on training corpus, sweeps 9 candidate models via cross-validation,
selects best by precision >= 0.85 then F1, saves artifacts to models/.
All experiments logged to MLflow.
"""
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing import normalize_address
from src.clustering import build_tfidf, address_pair_features

with open("config.json") as f:
    CONFIG = json.load(f)


def _build_feature_matrix(pairs: pd.DataFrame, vec) -> tuple:
    """Build feature matrix from labeled address pairs."""
    pairs = pairs[pairs["label"].isin(["SAME", "DIFFERENT"])].copy()
    pairs["a1"] = pairs["address_1_raw"].apply(normalize_address)
    pairs["a2"] = pairs["address_2_raw"].apply(normalize_address)
    features = [address_pair_features(r.a1, r.a2, vec) for _, r in pairs.iterrows()]
    X = pd.DataFrame(features)
    y = (pairs["label"] == "SAME").astype(int).values
    return X, y


def train_address_classifier(labeled_path: str, train_csv: str) -> Pipeline:
    """
    Fit TF-IDF on training corpus, sweep candidates, save best model.
    Returns the fitted best pipeline.
    """
    train_df = pd.read_csv(train_csv, dtype=str)
    train_addresses = (
        train_df[train_df["element_category"] == "Alamat"]["value"]
        .apply(normalize_address)
        .dropna()
        .tolist()
    )
    vec = build_tfidf(train_addresses)

    pairs = pd.read_csv(labeled_path)
    X, y = _build_feature_matrix(pairs, vec)

    print(f"Training on {len(y)} labeled pairs | SAME: {y.sum()} | DIFFERENT: {(1-y).sum()}")

    candidates = {
        "LogReg_C0.01":  Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=0.01, class_weight="balanced",
                                                     max_iter=500, random_state=42))]),
        "LogReg_C0.1":   Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                                     max_iter=500, random_state=42))]),
        "LogReg_C1.0":   Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                     max_iter=500, random_state=42))]),
        "LogReg_C10":    Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=10.0, class_weight="balanced",
                                                     max_iter=500, random_state=42))]),
        "RF_50":         Pipeline([("clf", RandomForestClassifier(n_estimators=50,
                                    class_weight="balanced", random_state=42, n_jobs=-1))]),
        "RF_200":        Pipeline([("clf", RandomForestClassifier(n_estimators=200,
                                    class_weight="balanced", random_state=42, n_jobs=-1))]),
        "GBM":           Pipeline([("sc", StandardScaler()),
                          ("clf", GradientBoostingClassifier(n_estimators=100,
                                                             max_depth=3, random_state=42))]),
        "SVM_RBF":       Pipeline([("sc", StandardScaler()),
                          ("clf", SVC(kernel="rbf", class_weight="balanced",
                                      probability=True, random_state=42))]),
        "SVM_Linear":    Pipeline([("sc", StandardScaler()),
                          ("clf", SVC(kernel="linear", class_weight="balanced",
                                      probability=True, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    with mlflow.start_run(run_name="address-classifier-sweep"):
        for name, pipeline in candidates.items():
            cv_res = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=["precision", "recall", "f1"],
                return_train_score=True,
            )
            val_p    = cv_res["test_precision"].mean()
            val_r    = cv_res["test_recall"].mean()
            val_f1   = cv_res["test_f1"].mean()
            train_f1 = cv_res["train_f1"].mean()
            gap      = train_f1 - val_f1
            overfit  = gap > 0.10

            results.append({"model": name, "val_precision": val_p, "val_recall": val_r,
                            "val_f1": val_f1, "train_f1": train_f1, "gap": gap,
                            "overfit": overfit})

            mlflow.log_metrics({f"{name}_val_p": val_p, f"{name}_val_f1": val_f1,
                                f"{name}_gap": gap})
            flag = "⚠ OVERFIT" if overfit else "✓"
            print(f"  {flag:<10} {name:<14}  P={val_p:.3f}  R={val_r:.3f}"
                  f"  F1={val_f1:.3f}  gap={gap:.3f}")

    results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    print(f"\n{'='*60}")
    print(results_df[["model","val_precision","val_recall","val_f1","gap","overfit"]]
          .to_string(index=False, float_format="{:.3f}".format))

    # Select: precision >= 0.85, non-overfit, highest F1
    qualified = results_df[(results_df["val_precision"] >= 0.85) & (~results_df["overfit"])]
    if qualified.empty:
        qualified = results_df[~results_df["overfit"]]
        print("⚠ No model hits precision>=0.85. Picking best non-overfit by F1.")
    if qualified.empty:
        qualified = results_df

    best_name = qualified.iloc[0]["model"]
    print(f"\n✓ Best model: {best_name}")

    best_pipeline = candidates[best_name]
    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, "models/address_classifier.pkl")
    joblib.dump(vec,           "models/address_tfidf.pkl")

    with mlflow.start_run(run_name="address-classifier-final", nested=True):
        mlflow.log_param("selected_model", best_name)

    print("✓ Saved models/address_classifier.pkl and models/address_tfidf.pkl")
    return best_pipeline

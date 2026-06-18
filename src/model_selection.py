"""
Comprehensive method comparison for all four modules.
Every candidate method is evaluated on labeled pairs.
Best method per module (precision >= 0.85 then F1) is written to config.json.
All experiments logged to MLflow.
"""
import json
import joblib
import mlflow
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_address, normalize_phone, normalize_id_number
from src.clustering import address_pair_features, build_tfidf

with open("config.json") as f:
    CONFIG = json.load(f)


# ─── MODULE 1 — ADDRESS ──────────────────────────────────────────────────────

def _get_address_methods(vec, classifier=None) -> dict:
    """Return all candidate address similarity methods as {name: (predict_fn, threshold)}."""
    methods = {}

    methods["ExactMatch"] = (lambda a1, a2: float(a1 == a2), 1.0)

    for t in [0.80, 0.85, 0.88, 0.90, 0.93]:
        methods[f"TokenSet@{t}"]  = (lambda a1, a2, _t=t: fuzz.token_set_ratio(a1, a2) / 100, t)
        methods[f"TokenSort@{t}"] = (lambda a1, a2, _t=t: fuzz.token_sort_ratio(a1, a2) / 100, t)
        methods[f"MaxFuzzy@{t}"]  = (lambda a1, a2, _t=t:
            max(fuzz.token_set_ratio(a1, a2), fuzz.token_sort_ratio(a1, a2)) / 100, t)

    for t in [0.75, 0.80, 0.85, 0.88]:
        methods[f"TFIDF_Cosine@{t}"] = (lambda a1, a2, _t=t, _v=vec:
            float(cosine_similarity(_v.transform([a1]), _v.transform([a2]))[0][0])
            if a1 and a2 else 0.0, t)

    if classifier is not None:
        def clf_predict(a1, a2, _c=classifier, _v=vec):
            feats = address_pair_features(a1, a2, _v)
            return float(_c.predict_proba(pd.DataFrame([feats]))[0][1])
        methods["Classifier"] = (clf_predict, 0.5)

    return methods


def evaluate_address_methods(val_pairs_path: str, vec, classifier=None) -> pd.DataFrame:
    """Evaluate all address methods on val pairs. Returns ranked DataFrame."""
    pairs = pd.read_csv(val_pairs_path)
    pairs = pairs[pairs["label"].isin(["SAME", "DIFFERENT"])].copy()
    y_true = (pairs["label"] == "SAME").astype(int).values
    pairs["a1"] = pairs["address_1_raw"].apply(normalize_address)
    pairs["a2"] = pairs["address_2_raw"].apply(normalize_address)

    methods = _get_address_methods(vec, classifier)
    results = []

    with mlflow.start_run(run_name="address-method-comparison"):
        for name, (predict_fn, threshold) in methods.items():
            scores = pairs.apply(lambda r: predict_fn(r.a1, r.a2), axis=1).values
            y_pred = (scores >= threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            results.append({"method": name, "threshold": threshold,
                            "val_precision": p, "val_recall": r, "val_f1": f})
            safe = name.replace("@", "_").replace(".", "_").replace("≤", "_lte_")
            mlflow.log_metrics({f"addr_{safe}_p": p, f"addr_{safe}_f1": f})

    df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    print("\n=== ADDRESS METHODS (sorted by F1) ===")
    print(df.to_string(index=False, float_format="{:.3f}".format))
    return df


# ─── MODULE 2 — PHONE ────────────────────────────────────────────────────────

def evaluate_phone_methods(val_pairs_path: str) -> pd.DataFrame:
    """Evaluate all phone prefix/distance methods on val pairs."""
    pairs = pd.read_csv(val_pairs_path)
    pairs = pairs[pairs["label"].isin(["SAME", "DIFFERENT"])].copy()
    pairs["p1"] = pairs["phone_1"].apply(normalize_phone)
    pairs["p2"] = pairs["phone_2"].apply(normalize_phone)
    pairs = pairs[pairs["p1"].notna() & pairs["p2"].notna()]
    y_true = (pairs["label"] == "SAME").astype(int).values

    results = []
    with mlflow.start_run(run_name="phone-method-comparison"):
        for pl in [7, 8, 9, 10]:
            y_pred = (pairs.p1.str[:pl] == pairs.p2.str[:pl]).astype(int).values
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            results.append({"method": f"Prefix@{pl}", "val_precision": p,
                            "val_recall": r, "val_f1": f})

        for max_d in [1, 2]:
            y_pred = pairs.apply(
                lambda r: Levenshtein.distance(r.p1, r.p2) <= max_d, axis=1
            ).astype(int).values
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            results.append({"method": f"Levenshtein≤{max_d}", "val_precision": p,
                            "val_recall": r, "val_f1": f})

        for pl in [7, 8, 9, 10]:
            for max_d in [1, 2]:
                y_pred = pairs.apply(
                    lambda r, _pl=pl, _d=max_d:
                        r.p1[:_pl] == r.p2[:_pl] or
                        Levenshtein.distance(r.p1, r.p2) <= _d,
                    axis=1).astype(int).values
                p, r, f, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0)
                results.append({"method": f"Prefix@{pl}+Lev≤{max_d}",
                                "val_precision": p, "val_recall": r, "val_f1": f})

        y_pred = (pairs.p1.str[:4] == pairs.p2.str[:4]).astype(int).values
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        results.append({"method": "OperatorOnly@4", "val_precision": p,
                        "val_recall": r, "val_f1": f})

    df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    print("\n=== PHONE METHODS (sorted by F1) ===")
    print(df.to_string(index=False, float_format="{:.3f}".format))
    return df


# ─── MASTER RUNNER ────────────────────────────────────────────────────────────

def run_all_module_comparisons(address_val_pairs: str,
                                phone_val_pairs: str) -> dict:
    """
    Run all method comparisons for all modules.
    Selects best method per module and writes to config.json and logs/.
    """
    import os
    vec        = joblib.load("models/address_tfidf.pkl") \
                 if os.path.exists("models/address_tfidf.pkl") else None
    classifier = joblib.load("models/address_classifier.pkl") \
                 if os.path.exists("models/address_classifier.pkl") else None

    # ── Address ──────────────────────────────────────────────────────────────
    addr_df = evaluate_address_methods(address_val_pairs, vec, classifier)
    qualified_addr = addr_df[addr_df["val_precision"] >= 0.85]
    if qualified_addr.empty:
        qualified_addr = addr_df
    best_addr = qualified_addr.iloc[0]

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone_df = evaluate_phone_methods(phone_val_pairs)
    qualified_phone = phone_df[phone_df["val_precision"] >= 0.85]
    if qualified_phone.empty:
        qualified_phone = phone_df
    best_phone = qualified_phone.iloc[0]

    best_phone_prefix = (
        int(best_phone["method"].split("@")[1].split("+")[0])
        if "@" in best_phone["method"]
        else CONFIG["phone"]["prefix_length"]
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "address": {
            "best_method":        best_addr["method"],
            "best_val_precision": float(best_addr["val_precision"]),
            "best_val_recall":    float(best_addr["val_recall"]),
            "best_val_f1":        float(best_addr["val_f1"]),
            "all_results":        addr_df.to_dict("records"),
        },
        "phone": {
            "best_method":        best_phone["method"],
            "best_val_precision": float(best_phone["val_precision"]),
            "best_val_recall":    float(best_phone["val_recall"]),
            "best_val_f1":        float(best_phone["val_f1"]),
            "all_results":        phone_df.to_dict("records"),
        },
        "ktp_npwp": {"method": "ExactDistinctCount + NIKStructuralValidation"},
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/model_selection_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    CONFIG["selected_address_method"] = best_addr["method"]
    CONFIG["selected_phone_prefix"]   = best_phone_prefix
    with open("config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  BEST ADDRESS METHOD : {best_addr['method']}")
    print(f"    Precision={best_addr['val_precision']:.3f}  F1={best_addr['val_f1']:.3f}")
    print(f"  BEST PHONE METHOD   : {best_phone['method']}")
    print(f"    Precision={best_phone['val_precision']:.3f}  F1={best_phone['val_f1']:.3f}")
    print(f"  Report → logs/model_selection_report.json")
    print(f"  Config → config.json updated")
    return report

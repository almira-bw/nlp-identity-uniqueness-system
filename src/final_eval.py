"""
Phase 7 — Final evaluation on the locked test set.
One-way door: run once, log results, do not retune after this point.
"""
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd

from src.clustering import (
    cluster_addresses, phone_unique_count, ktp_uniqueness, npwp_uniqueness,
)


def _build_records(group: pd.DataFrame) -> dict:
    """Convert EAV group rows into keyed lists for each module."""
    def vals(cat):
        return group.loc[group["element_category"] == cat, "value"].dropna().tolist()

    return {
        "alamat":     [{"alamat":    v} for v in vals("Alamat")],
        "nomor_hp":   [{"nomor_hp":  v} for v in vals("Nomor Telepon Seluler")],
        "nomor_ktp":  [{"nomor_ktp": v} for v in vals("Nomor Identitas")],
        "npwp":       [{"npwp":      v} for v in vals("NPWP")],
    }


def run_final_eval(test_path: str = "data/processed/test.csv") -> dict:
    """
    Evaluate all 4 modules on the locked test set.
    Logs results to MLflow under run name 'FINAL-TEST-EVAL'.
    Returns summary dict.
    """
    vec        = joblib.load("models/address_tfidf.pkl")
    classifier = joblib.load("models/address_classifier.pkl")

    df = pd.read_csv(test_path)
    identities = df["nomor_identitas"].unique()
    print(f"\nLoaded test set: {len(df)} rows, {len(identities)} identities")

    rows = []
    for nid in identities:
        group   = df[df["nomor_identitas"] == nid]
        records = _build_records(group)

        addr_count  = cluster_addresses(records["alamat"], vec, classifier=classifier)
        phone_count = phone_unique_count(records["nomor_hp"])
        ktp_res     = ktp_uniqueness(records["nomor_ktp"])
        npwp_res    = npwp_uniqueness(records["npwp"])

        rows.append({
            "nomor_identitas":  nid,
            "addr_unique":      addr_count,
            "phone_unique":     phone_count,
            "ktp_unique":       ktp_res["ktp_unique_count"],
            "ktp_category":     ktp_res["ktp_category"],
            "ktp_near_dups":    len(ktp_res["near_duplicates"]),
            "npwp_unique":      npwp_res["npwp_unique_count"],
            "npwp_category":    npwp_res["npwp_category"],
            "npwp_near_dups":   len(npwp_res["near_duplicates"]),
        })

    results_df = pd.DataFrame(rows)

    # ── Summary statistics ────────────────────────────────────────────────────
    has_addr  = results_df["addr_unique"]  > 0
    has_phone = results_df["phone_unique"] > 0
    has_ktp   = results_df["ktp_unique"]   > 0
    has_npwp  = results_df["npwp_unique"]  > 0

    summary = {
        "test_identities":    int(len(identities)),
        "test_rows":          int(len(df)),

        # ── M1 Address ──────────────────────────────────────────────────────
        "addr_coverage_pct":  round(float(has_addr.mean()) * 100, 1),
        "addr_mean_unique":   round(float(results_df.loc[has_addr, "addr_unique"].mean()), 3),
        "addr_max_unique":    int(results_df["addr_unique"].max()),
        "addr_multi_pct":     round(float((results_df["addr_unique"] > 1).mean()) * 100, 1),

        # ── M2 Phone ────────────────────────────────────────────────────────
        "phone_coverage_pct": round(float(has_phone.mean()) * 100, 1),
        "phone_mean_unique":  round(float(results_df.loc[has_phone, "phone_unique"].mean()), 3),
        "phone_max_unique":   int(results_df["phone_unique"].max()),
        "phone_multi_pct":    round(float((results_df["phone_unique"] > 1).mean()) * 100, 1),

        # ── M3a KTP ─────────────────────────────────────────────────────────
        "ktp_coverage_pct":   round(float(has_ktp.mean()) * 100, 1),
        "ktp_anomaly_rate":   round(float((results_df["ktp_category"] == "Anomalous").mean()) * 100, 1),
        "ktp_near_dup_cases": int((results_df["ktp_near_dups"] > 0).sum()),

        # ── M3b NPWP ────────────────────────────────────────────────────────
        "npwp_coverage_pct":  round(float(has_npwp.mean()) * 100, 1),
        "npwp_anomaly_rate":  round(float((results_df["npwp_category"] == "Anomalous").mean()) * 100, 1),
        "npwp_near_dup_cases": int((results_df["npwp_near_dups"] > 0).sum()),
    }

    # ── Print report ─────────────────────────────────────────────────────────
    _print_report(summary, results_df)

    # ── MLflow ───────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="FINAL-TEST-EVAL"):
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_param("test_path", test_path)

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    results_df.to_csv("logs/final_eval_per_identity.csv", index=False)

    full_report = {"summary": summary, "per_identity": results_df.to_dict("records")}
    with open("logs/final_eval_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print("\nArtefacts saved:")
    print("  logs/final_eval_report.json")
    print("  logs/final_eval_per_identity.csv")

    return summary


def _print_report(summary: dict, df: pd.DataFrame) -> None:
    w = 55
    print(f"\n{'='*w}")
    print(f"  FINAL TEST SET EVALUATION")
    print(f"  Identities: {summary['test_identities']}   Rows: {summary['test_rows']}")
    print(f"{'='*w}")

    print(f"\n  M1 — Address (Classifier, best method)")
    print(f"    Coverage          : {summary['addr_coverage_pct']}%")
    print(f"    Mean unique/id    : {summary['addr_mean_unique']:.3f}")
    print(f"    Max unique        : {summary['addr_max_unique']}")
    print(f"    Identities > 1    : {summary['addr_multi_pct']}%")
    _dist("addr_unique", df)

    print(f"\n  M2 — Phone (Prefix@{_phone_prefix()})")
    print(f"    Coverage          : {summary['phone_coverage_pct']}%")
    print(f"    Mean unique/id    : {summary['phone_mean_unique']:.3f}")
    print(f"    Max unique        : {summary['phone_max_unique']}")
    print(f"    Identities > 1    : {summary['phone_multi_pct']}%")
    _dist("phone_unique", df)

    print(f"\n  M3a — KTP (Nomor Identitas)")
    print(f"    Coverage          : {summary['ktp_coverage_pct']}%")
    print(f"    Anomaly rate      : {summary['ktp_anomaly_rate']}%")
    print(f"    Near-dup cases    : {summary['ktp_near_dup_cases']}")

    print(f"\n  M3b — NPWP")
    print(f"    Coverage          : {summary['npwp_coverage_pct']}%")
    print(f"    Anomaly rate      : {summary['npwp_anomaly_rate']}%")
    print(f"    Near-dup cases    : {summary['npwp_near_dup_cases']}")

    print(f"\n{'='*w}\n")


def _dist(col: str, df: pd.DataFrame) -> None:
    vc = df[col].value_counts().sort_index()
    parts = "  ".join(f"{k}→{v}" for k, v in vc.items())
    print(f"    Distribution      : {parts}")


def _phone_prefix() -> int:
    with open("config.json") as f:
        return json.load(f)["selected_phone_prefix"]


if __name__ == "__main__":
    run_final_eval()

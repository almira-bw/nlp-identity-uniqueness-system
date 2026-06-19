"""
Phase 11 — Feature Extraction Table for Underwriting Consumption.

Produces a long-format table: one row per (identity, feature_type, unique_value).
Each row carries first_seen / last_seen dates from tahun_bulan_data.

Core feature types (Address, Phone, KTP, NPWP) use the same normalization and
clustering logic as the NLP modules. All other element_category values are
deduplicated by lowercased exact match.
"""

import itertools
from typing import Optional

import networkx as nx
import pandas as pd

from src.preprocessing import normalize_address, normalize_phone, normalize_id_number
from src.clustering import address_pair_features


# element_category → feature group label
FEATURE_GROUP_MAP = {
    "Alamat": "address",
    "Nomor Telepon Seluler": "phone",
    "Nomor Telepon": "phone",
    "Nomor Identitas": "ktp",
    "NPWP": "npwp",
    "Tempat Bekerja": "workplace",
    "Email": "email",
    "Nama Debitur": "name",
    "Tanggal Lahir": "dob",
    "Nama Gadis Ibu Kandung": "mother_name",
}


def _mode_or_first(series: pd.Series) -> str:
    m = series.dropna().mode()
    return str(m.iloc[0]) if len(m) > 0 else (str(series.dropna().iloc[0]) if not series.dropna().empty else "")


def _date_range(dates: pd.Series):
    d = dates.dropna()
    if d.empty:
        return None, None
    return d.min(), d.max()


def _cluster_address_labels(raw_values: list, vec, classifier) -> list:
    """
    Assign a cluster label (int) to each raw address value.
    Returns list of ints, same length as raw_values.
    """
    cleaned = [normalize_address(v) for v in raw_values]
    n = len(cleaned)

    if n == 0:
        return []
    if n == 1:
        return [0]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, j in itertools.combinations(range(n), 2):
        a1, a2 = cleaned[i], cleaned[j]
        if not a1 or not a2:
            continue
        feats = address_pair_features(a1, a2, vec)
        pred = classifier.predict(pd.DataFrame([feats]))[0]
        if pred == 1:
            G.add_edge(i, j)

    # Map root component → compact label 0, 1, 2, ...
    components = list(nx.connected_components(G))
    node_to_label = {}
    for label, component in enumerate(components):
        for node in component:
            node_to_label[node] = label

    return [node_to_label[i] for i in range(n)]


def _emit_address(records, nid, sub, vec, classifier):
    """One row per address cluster per identity."""
    raw_vals = sub["value"].tolist()
    labels = _cluster_address_labels(raw_vals, vec, classifier)

    sub = sub.copy().reset_index(drop=True)
    sub["_label"] = labels

    for lbl, grp in sub.groupby("_label"):
        rep_raw = _mode_or_first(grp["value"])
        norm = normalize_address(rep_raw) or rep_raw
        first, last = _date_range(grp["tahun_bulan_data"])
        records.append({
            "nomor_identitas": nid,
            "feature_type": "Alamat",
            "feature_group": "address",
            "normalized_value": norm,
            "raw_value_sample": rep_raw,
            "first_seen": first,
            "last_seen": last,
            "occurrence_count": len(grp),
            "cluster_id": int(lbl),
        })


def _emit_phone(records, nid, cat, sub):
    """One row per distinct phone prefix (first 9 digits) per identity — consistent with M2 prefix@9 matching."""
    sub = sub.copy()
    sub["_norm"] = sub["value"].apply(normalize_phone)
    sub = sub[sub["_norm"].notna()]
    sub["_prefix"] = sub["_norm"].str[:9]

    for prefix, grp in sub.groupby("_prefix"):
        rep_norm = grp["_norm"].mode().iloc[0]
        first, last = _date_range(grp["tahun_bulan_data"])
        records.append({
            "nomor_identitas": nid,
            "feature_type": cat,
            "feature_group": "phone",
            "normalized_value": rep_norm,
            "raw_value_sample": _mode_or_first(grp["value"]),
            "first_seen": first,
            "last_seen": last,
            "occurrence_count": len(grp),
            "cluster_id": None,
        })


def _emit_id(records, nid, cat, group_name, sub, length: int):
    """One row per distinct normalized ID number per identity."""
    sub = sub.copy()
    sub["_norm"] = sub["value"].apply(lambda v: normalize_id_number(v, length))
    sub = sub[sub["_norm"].notna()]

    for norm, grp in sub.groupby("_norm"):
        first, last = _date_range(grp["tahun_bulan_data"])
        records.append({
            "nomor_identitas": nid,
            "feature_type": cat,
            "feature_group": group_name,
            "normalized_value": norm,
            "raw_value_sample": _mode_or_first(grp["value"]),
            "first_seen": first,
            "last_seen": last,
            "occurrence_count": len(grp),
            "cluster_id": None,
        })


def _emit_generic(records, nid, cat, sub):
    """Exact dedup (lowercased) for non-core feature types."""
    sub = sub.copy()
    sub["_norm"] = sub["value"].apply(
        lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else None
    )
    sub = sub[sub["_norm"].notna()]

    group_name = FEATURE_GROUP_MAP.get(cat, "other")
    for norm, grp in sub.groupby("_norm"):
        first, last = _date_range(grp["tahun_bulan_data"])
        records.append({
            "nomor_identitas": nid,
            "feature_type": cat,
            "feature_group": group_name,
            "normalized_value": norm,
            "raw_value_sample": _mode_or_first(grp["value"]),
            "first_seen": first,
            "last_seen": last,
            "occurrence_count": len(grp),
            "cluster_id": None,
        })


def extract_identity_features(
    raw_df: pd.DataFrame,
    vec,
    classifier,
    output_path: Optional[str] = "data/output/identity_features.csv",
) -> pd.DataFrame:
    """
    Extract unique feature values per identity with min/max seen dates.

    Parameters
    ----------
    raw_df      : Full EAV DataFrame (all element_category values included).
    vec         : Fitted TfidfVectorizer (from models/address_tfidf.pkl).
    classifier  : Trained address classifier pipeline (from models/address_classifier.pkl).
    output_path : If given, saves CSV to this path.

    Returns
    -------
    pd.DataFrame with columns:
        nomor_identitas, feature_type, feature_group, normalized_value,
        raw_value_sample, first_seen, last_seen, occurrence_count, cluster_id
    """
    records = []
    all_cats = raw_df["element_category"].dropna().unique().tolist()
    identities = raw_df["nomor_identitas"].dropna().unique()
    n_ids = len(identities)

    print(f"Extracting features for {n_ids:,} identities across {len(all_cats)} feature types...")

    for idx, nid in enumerate(identities, 1):
        if idx % 1000 == 0:
            print(f"  {idx:,} / {n_ids:,} ({idx/n_ids*100:.0f}%)")

        group = raw_df[raw_df["nomor_identitas"] == nid]

        # Combine both phone categories first so the same normalized number
        # appearing in both "Nomor Telepon Seluler" and "Nomor Telepon" is
        # counted only once per identity.
        phone_parts = [
            group[group["element_category"] == c].dropna(subset=["value"])
            for c in ("Nomor Telepon Seluler", "Nomor Telepon")
        ]
        phone_parts = [p for p in phone_parts if not p.empty]
        if phone_parts:
            _emit_phone(records, str(nid), "phone", pd.concat(phone_parts))

        for cat in all_cats:
            sub = group[group["element_category"] == cat].dropna(subset=["value"])
            if sub.empty:
                continue

            if cat == "Alamat":
                _emit_address(records, str(nid), sub, vec, classifier)
            elif cat in ("Nomor Telepon Seluler", "Nomor Telepon"):
                pass  # handled above as combined group
            elif cat == "Nomor Identitas":
                _emit_id(records, str(nid), cat, "ktp", sub, length=16)
            elif cat == "NPWP":
                _emit_id(records, str(nid), cat, "npwp", sub, length=15)
            else:
                _emit_generic(records, str(nid), cat, sub)

    df = pd.DataFrame(records)
    if df.empty:
        print("Warning: no records extracted.")
        return df

    df = df.sort_values(
        ["nomor_identitas", "feature_group", "feature_type", "first_seen"]
    ).reset_index(drop=True)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(df):,} rows → {output_path}")

    return df


def generate_identity_summary(
    features_df: pd.DataFrame,
    output_path: str = "data/output/identity_summary.csv",
) -> pd.DataFrame:
    """
    Pivot the long-format features table into one row per identity.

    Input : output of extract_identity_features() or identity_features.csv.
    Output columns:
        nomor_identitas,
        addr_unique,  addr_min_date,  addr_max_date,
        phone_unique, phone_min_date, phone_max_date,
        ktp_unique,   ktp_category,   ktp_min_date,  ktp_max_date,
        npwp_unique,  npwp_category,  npwp_min_date, npwp_max_date
    """

    def _risk_category(n):
        """Map unique count to risk label using M1/M2 thresholds (from config.json)."""
        if n == 0: return "Not Available"
        if n == 1: return "Normal"
        if n == 2: return "Low"
        if n == 3: return "Medium"
        return "High"

    def _summarise(g):
        def _count(group_name):
            return int((g["feature_group"] == group_name).sum())

        def _dates(group_name):
            sub = g[g["feature_group"] == group_name]
            return sub["first_seen"].dropna().min(), sub["last_seen"].dropna().max()

        addr_n  = _count("address")
        phone_n = _count("phone")
        ktp_n   = _count("ktp")
        npwp_n  = _count("npwp")

        addr_min,  addr_max  = _dates("address")
        phone_min, phone_max = _dates("phone")
        ktp_min,   ktp_max   = _dates("ktp")
        npwp_min,  npwp_max  = _dates("npwp")

        ktp_cat  = ("Not Available" if ktp_n  == 0 else
                    "Normal"        if ktp_n  == 1 else "Anomalous")
        npwp_cat = ("Not Available" if npwp_n == 0 else
                    "Normal"        if npwp_n == 1 else "Anomalous")

        return pd.Series({
            "addr_unique":    addr_n,
            "addr_category":  _risk_category(addr_n),
            "addr_min_date":  addr_min,
            "addr_max_date":  addr_max,
            "phone_unique":   phone_n,
            "phone_category": _risk_category(phone_n),
            "phone_min_date": phone_min,
            "phone_max_date": phone_max,
            "ktp_unique":     ktp_n,
            "ktp_category":   ktp_cat,
            "ktp_min_date":   ktp_min,
            "ktp_max_date":   ktp_max,
            "npwp_unique":    npwp_n,
            "npwp_category":  npwp_cat,
            "npwp_min_date":  npwp_min,
            "npwp_max_date":  npwp_max,
        })

    print(f"Building summary for {features_df['nomor_identitas'].nunique():,} identities...")
    result = (
        features_df
        .groupby("nomor_identitas")
        .apply(_summarise, include_groups=False)
        .reset_index()
    )

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"✓ Saved {len(result):,} rows → {output_path}")

    return result

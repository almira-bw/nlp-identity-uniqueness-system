import json
import itertools
from typing import Optional

import networkx as nx
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_address, normalize_phone, normalize_id_number

with open("config.json") as f:
    CONFIG = json.load(f)


def build_tfidf(corpus: list[str]) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on address corpus."""
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=1, max_df=0.95, sublinear_tf=True,
    )
    vec.fit([a for a in corpus if a])
    return vec


def address_pair_features(a1: Optional[str], a2: Optional[str],
                           vec: TfidfVectorizer) -> dict:
    """Compute all similarity features for a normalized address pair."""
    if not a1 or not a2:
        return {k: 0.0 for k in ["token_sort", "token_set", "partial_ratio",
                                  "len_ratio", "tfidf_cosine"]}
    tfidf_score = float(
        cosine_similarity(vec.transform([a1]), vec.transform([a2]))[0][0]
    )
    return {
        "token_sort":    fuzz.token_sort_ratio(a1, a2) / 100,
        "token_set":     fuzz.token_set_ratio(a1, a2) / 100,
        "partial_ratio": fuzz.partial_ratio(a1, a2) / 100,
        "len_ratio":     min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
        "tfidf_cosine":  tfidf_score,
    }


def cluster_addresses(records: list[dict], vec: TfidfVectorizer,
                       classifier=None, threshold: Optional[float] = None) -> int:
    """
    Return number of distinct address clusters for one nomor_identitas.
    Uses classifier if provided (Phase 5 best model), else threshold on max fuzzy score.
    """
    addrs = [normalize_address(r.get("alamat")) for r in records]
    addrs = [a for a in addrs if a]
    if not addrs:
        return 0

    t = threshold or CONFIG["address"]["token_set_threshold"] / 100
    G = nx.Graph()
    G.add_nodes_from(range(len(addrs)))

    for i, j in itertools.combinations(range(len(addrs)), 2):
        feats = address_pair_features(addrs[i], addrs[j], vec)
        if classifier is not None:
            import pandas as pd
            same = classifier.predict(pd.DataFrame([feats]))[0] == 1
        else:
            same = (
                max(feats["token_set"], feats["token_sort"]) >= t or
                feats["partial_ratio"] >= CONFIG["address"]["embedding_threshold"]
            )
        if same:
            G.add_edge(i, j)

    return nx.number_connected_components(G)


def phone_unique_count(records: list[dict],
                        prefix_length: Optional[int] = None) -> int:
    """Return number of distinct phone clusters for one nomor_identitas."""
    pl       = prefix_length or CONFIG["phone"]["prefix_length"]
    max_edit = CONFIG["phone"]["levenshtein_max_distance"]

    phones = [normalize_phone(r.get("nomor_hp")) for r in records]
    phones = [p for p in phones if p]
    if not phones:
        return 0

    G = nx.Graph()
    G.add_nodes_from(range(len(phones)))
    for i, j in itertools.combinations(range(len(phones)), 2):
        if (phones[i][:pl] == phones[j][:pl] or
                Levenshtein.distance(phones[i], phones[j]) <= max_edit):
            G.add_edge(i, j)

    return nx.number_connected_components(G)


def ktp_uniqueness(records: list[dict]) -> dict:
    """Return unique count, category, and near-duplicate warnings for KTP."""
    ktps = [normalize_id_number(r.get("nomor_ktp"), 16) for r in records]
    ktps = [k for k in ktps if k]
    if not ktps:
        return {"ktp_unique_count": 0, "ktp_category": "Not Available",
                "near_duplicates": []}

    unique = list(set(ktps))
    near_dups = [
        {"ktp_1": k1, "ktp_2": k2,
         "edit_distance": Levenshtein.distance(k1, k2)}
        for k1, k2 in itertools.combinations(unique, 2)
        if Levenshtein.distance(k1, k2) <= CONFIG["ktp"]["levenshtein_warning_distance"]
    ]
    return {
        "ktp_unique_count": len(unique),
        "ktp_category":     "Normal" if len(unique) == 1 else "Anomalous",
        "near_duplicates":  near_dups,
    }


def npwp_uniqueness(records: list[dict]) -> dict:
    """Return unique count, category, and near-duplicate warnings for NPWP."""
    npwps = [normalize_id_number(r.get("npwp"), 15) for r in records]
    npwps = [n for n in npwps if n]
    if not npwps:
        return {"npwp_unique_count": 0, "npwp_category": "Not Available",
                "near_duplicates": []}

    unique = list(set(npwps))
    near_dups = [
        {"npwp_1": n1, "npwp_2": n2,
         "edit_distance": Levenshtein.distance(n1, n2)}
        for n1, n2 in itertools.combinations(unique, 2)
        if Levenshtein.distance(n1, n2) <= CONFIG["npwp"]["levenshtein_warning_distance"]
    ]
    return {
        "npwp_unique_count": len(unique),
        "npwp_category":     "Normal" if len(unique) == 1 else "Anomalous",
        "near_duplicates":   near_dups,
    }

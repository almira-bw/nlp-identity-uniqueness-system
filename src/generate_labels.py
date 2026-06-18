"""Generate label files for address pairs (normalized + auto-labeled) and phone pairs (auto-labeled)."""
import json
import pandas as pd
import itertools
from rapidfuzz import fuzz
from src.preprocessing import normalize_address, normalize_phone

with open("config.json") as f:
    CONFIG = json.load(f)

ADDR_SAME_THRESHOLD      = CONFIG["address"]["token_set_threshold"]   # token_set >= this → SAME
ADDR_DIFFERENT_THRESHOLD = CONFIG["address"]["borderline_lower"]       # token_set < this  → DIFFERENT
ADDR_SORT_THRESHOLD      = 65    # for UNCLEAR: token_sort >= this → SAME
ADDR_PARTIAL_THRESHOLD   = 85    # for UNCLEAR: partial >= this   → SAME (abbreviation case)

df = pd.read_csv("data/raw/applications.csv", dtype=str)

# ── Address pairs → normalized + auto-labeled with fuzzy matching ─────────────
addr_df = df[df["element_category"] == "Alamat"].dropna(subset=["nomor_identitas", "value"])

pairs = []
for identity, group in addr_df.groupby("nomor_identitas"):
    raw_addrs = group["value"].tolist()
    idx       = group.index.tolist()
    for (i1, a1_raw), (i2, a2_raw) in itertools.combinations(zip(idx, raw_addrs), 2):
        a1 = normalize_address(a1_raw)
        a2 = normalize_address(a2_raw)
        if not a1 or not a2:
            continue
        token_set  = round(fuzz.token_set_ratio(a1, a2))
        token_sort = round(fuzz.token_sort_ratio(a1, a2))
        partial    = round(fuzz.partial_ratio(a1, a2))

        if token_set >= ADDR_SAME_THRESHOLD:
            label = "SAME"
        elif token_set < ADDR_DIFFERENT_THRESHOLD:
            label = "DIFFERENT"
        elif token_sort >= ADDR_SORT_THRESHOLD or partial >= ADDR_PARTIAL_THRESHOLD:
            label = "SAME"     # borderline resolved: reorder or abbreviation
        else:
            label = "DIFFERENT"  # borderline resolved: shared area tokens, different place
        score = token_set
        pairs.append({
            "nomor_identitas":   identity,
            "record_id_1":       i1,
            "address_1_raw":     a1_raw,
            "address_1_clean":   a1,
            "record_id_2":       i2,
            "address_2_raw":     a2_raw,
            "address_2_clean":   a2,
            "token_set_score":   score,
            "token_sort_score":  token_sort,
            "partial_score":     partial,
            "label":             label,
            "annotator":         "",
            "annotator_2":       "",
            "label_2":           "",   # dual annotation for kappa (fill 100 rows)
        })

addr_pairs_df = pd.DataFrame(pairs).sample(min(1000, len(pairs)), random_state=42)
addr_pairs_df.to_csv("data/labeled/address_pairs_to_label.csv", index=False)

label_counts = addr_pairs_df["label"].value_counts().to_dict()
print(f"→ Address pairs generated: {len(addr_pairs_df)} rows (fully auto-labeled)")
print(f"  SAME      : {label_counts.get('SAME', 0)}")
print(f"  DIFFERENT : {label_counts.get('DIFFERENT', 0)}")
print(f"  Dual-annotate 100 rows (fill label_2) for kappa check.")

# ── Phone pairs → cleansed (8X format) + programmatic labels ─────────────────
phone_df = df[df["element_category"] == "Nomor Telepon Seluler"].dropna(subset=["nomor_identitas", "value"])

phone_pairs = []
for identity, group in phone_df.groupby("nomor_identitas"):
    phones = [normalize_phone(p) for p in group["value"]]
    phones = list(dict.fromkeys(p for p in phones if p))  # deduplicate, preserve order
    for p1, p2 in itertools.combinations(phones, 2):
        phone_pairs.append({
            "nomor_identitas": identity,
            "phone_1":         p1,
            "phone_2":         p2,
            "label":           "SAME" if p1[:9] == p2[:9] else "DIFFERENT",
        })

phone_df_out = pd.DataFrame(phone_pairs)
phone_df_out.to_csv("data/labeled/phone_pairs_labeled.csv", index=False)
p_counts = phone_df_out["label"].value_counts().to_dict() if len(phone_df_out) else {}
print(f"\n→ Phone pairs auto-generated: {len(phone_pairs)} rows (8X format, prefix stripped)")
print(f"  SAME      : {p_counts.get('SAME', 0)}")
print(f"  DIFFERENT : {p_counts.get('DIFFERENT', 0)}")

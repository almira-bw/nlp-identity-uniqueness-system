"""Audit raw application data (EAV format) for the 4 NLP modules."""
import pandas as pd

RELEVANT_CATEGORIES = {
    "Alamat": "M1 — Address",
    "Nomor Telepon Seluler": "M2 — Phone",
    "Nomor Identitas": "M3a — KTP",
    "NPWP": "M3b — NPWP",
}

df = pd.read_csv("data/raw/applications.csv", dtype=str)

print(f"Total rows      : {len(df):,}")
print(f"Unique identities: {df['nomor_identitas'].nunique():,}")
print(f"Unique reporters : {df['id_pelapor'].nunique():,}")
print(f"Date range      : {df['tahun_bulan_data'].min()} → {df['tahun_bulan_data'].max()}")
print()

print("─── Coverage per module ────────────────────────────────")
for cat, label in RELEVANT_CATEGORIES.items():
    sub = df[df["element_category"] == cat]
    identities_with_data = sub["nomor_identitas"].nunique()
    total_identities = df["nomor_identitas"].nunique()
    coverage = identities_with_data / total_identities * 100 if total_identities else 0
    null_or_zero = sub["value"].isin(["0", "00000000000000", ""]).mean() * 100
    multi = (sub.groupby("nomor_identitas").size() > 1).mean() * 100
    print(f"  {label:<22} ({cat})")
    print(f"    Identities with data : {identities_with_data:,} / {total_identities:,}  ({coverage:.1f}% coverage)")
    print(f"    Placeholder/zero rows: {null_or_zero:.1f}%")
    print(f"    Multi-record identities: {multi:.1f}%  (need >5% for clustering)")
    print()

print("─── elemen_category distribution ──────────────────────")
print(df["element_category"].value_counts().to_string())

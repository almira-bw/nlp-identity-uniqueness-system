"""Identity-level train/val/test split — no identity appears in more than one split."""
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(df: pd.DataFrame, seed: int = 42) -> tuple:
    """
    Split EAV data at identity level (70/15/15).
    Stratified by number of address records per identity so each split
    has a representative mix of single-record and multi-record identities.
    """
    addr_counts = (
        df[df["element_category"] == "Alamat"]
        .groupby("nomor_identitas")
        .size()
        .reset_index(name="n_addresses")
    )

    all_ids = df["nomor_identitas"].unique()
    counts = (
        pd.Series(all_ids, name="nomor_identitas")
        .to_frame()
        .merge(addr_counts, on="nomor_identitas", how="left")
        .fillna({"n_addresses": 0})
    )
    counts["n_addresses"] = counts["n_addresses"].astype(int)

    counts["stratum"] = pd.cut(
        counts["n_addresses"],
        bins=[-1, 1, 3, 10, 9999],
        labels=["single", "few", "medium", "many"],
    )

    ids     = counts["nomor_identitas"].values
    strata  = counts["stratum"].values

    train_ids, tmp_ids, _, tmp_strata = train_test_split(
        ids, strata, test_size=0.30, stratify=strata, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=0.50, stratify=tmp_strata, random_state=seed
    )

    assert not set(train_ids) & set(test_ids), "LEAKAGE: train ∩ test"
    assert not set(val_ids)   & set(test_ids), "LEAKAGE: val ∩ test"

    train = df[df["nomor_identitas"].isin(train_ids)]
    val   = df[df["nomor_identitas"].isin(val_ids)]
    test  = df[df["nomor_identitas"].isin(test_ids)]

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv",     index=False)
    test.to_csv("data/processed/test.csv",   index=False)

    print(f"Train : {len(train_ids):3d} identities  ({len(train):4d} rows)")
    print(f"Val   : {len(val_ids):3d} identities  ({len(val):4d} rows)")
    print(f"Test  : {len(test_ids):3d} identities  ({len(test):4d} rows)")
    print("✓ No leakage confirmed. TEST SET LOCKED — do not load until Phase 6.")
    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv("data/raw/applications.csv", dtype=str)
    create_splits(df)

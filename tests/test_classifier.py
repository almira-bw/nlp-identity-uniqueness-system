import os
import pytest


def test_classifier_file_exists_after_training():
    """Model artifacts must be saved to disk after training."""
    from src.train import train_address_classifier
    train_address_classifier(
        labeled_path="data/labeled/address_pairs_to_label.csv",
        train_csv="data/processed/train.csv",
    )
    assert os.path.exists("models/address_classifier.pkl")
    assert os.path.exists("models/address_tfidf.pkl")


def test_classifier_predicts_same_for_identical_addresses():
    import joblib
    import pandas as pd
    clf = joblib.load("models/address_classifier.pkl")
    vec = joblib.load("models/address_tfidf.pkl")
    from src.clustering import address_pair_features
    from src.preprocessing import normalize_address
    a = normalize_address("Jl. Sudirman No. 5 Jakarta")
    feats = address_pair_features(a, a, vec)
    pred = clf.predict(pd.DataFrame([feats]))[0]
    assert pred == 1  # SAME


def test_classifier_predicts_different_for_distant_addresses():
    import joblib
    import pandas as pd
    clf = joblib.load("models/address_classifier.pkl")
    vec = joblib.load("models/address_tfidf.pkl")
    from src.clustering import address_pair_features
    from src.preprocessing import normalize_address
    a1 = normalize_address("Jl. Sudirman No. 5 Jakarta")
    a2 = normalize_address("Jl. Pahlawan No. 99 Surabaya")
    feats = address_pair_features(a1, a2, vec)
    pred = clf.predict(pd.DataFrame([feats]))[0]
    assert pred == 0  # DIFFERENT


def test_classifier_val_f1_above_floor():
    """Val F1 must be at least 0.70."""
    import joblib
    import pandas as pd
    from sklearn.metrics import f1_score
    clf = joblib.load("models/address_classifier.pkl")
    vec = joblib.load("models/address_tfidf.pkl")
    from src.clustering import address_pair_features
    from src.preprocessing import normalize_address

    pairs = pd.read_csv("data/labeled/address_pairs_to_label.csv")
    pairs = pairs[pairs["label"].isin(["SAME", "DIFFERENT"])].copy()
    pairs["a1"] = pairs["address_1_raw"].apply(normalize_address)
    pairs["a2"] = pairs["address_2_raw"].apply(normalize_address)
    feats = [address_pair_features(r.a1, r.a2, vec) for _, r in pairs.iterrows()]
    X = pd.DataFrame(feats)
    y = (pairs["label"] == "SAME").astype(int).values
    f1 = f1_score(y, clf.predict(X))
    assert f1 >= 0.70, f"Val F1 {f1:.3f} below floor 0.70"

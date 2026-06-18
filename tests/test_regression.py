"""
Regression tests — locked baselines from Phase 7 final evaluation.
These must never change unless a deliberate model update is made and
Phase 7 is re-run with explicit sign-off.
"""
import pytest
import joblib
import pandas as pd

from src.clustering import (
    cluster_addresses, phone_unique_count, ktp_uniqueness, npwp_uniqueness,
)
from src.categories import (
    address_category, phone_category, ktp_category_code, npwp_category_code,
    identity_risk_summary,
)

# ── Phase 7 locked baselines (do not edit without re-running final eval) ──────
# Updated 2026-06-18: test split re-run via notebook → 1,349 identities.
# Representative sample: single-attr, multi-addr, multi-phone, KTP anomaly, NPWP anomaly.
BASELINES = {
    # (nomor_identitas, addr_unique, phone_unique, ktp_unique, npwp_unique)
    "1103186505990001": {"addr": 4,  "phone": 1, "ktp": 1, "npwp": 0},
    "1106075004820004": {"addr": 1,  "phone": 1, "ktp": 1, "npwp": 0},
    "1113114203860001": {"addr": 1,  "phone": 1, "ktp": 1, "npwp": 1},
    "1115086307920001": {"addr": 5,  "phone": 3, "ktp": 2, "npwp": 0},
    "1201074910880001": {"addr": 1,  "phone": 3, "ktp": 1, "npwp": 1},
    "1207020411950003": {"addr": 2,  "phone": 1, "ktp": 2, "npwp": 0},
    "1207261209940006": {"addr": 1,  "phone": 3, "ktp": 1, "npwp": 2},
    "1209151206820001": {"addr": 7,  "phone": 4, "ktp": 1, "npwp": 1},
}

# ── Summary baselines ─────────────────────────────────────────────────────────
SUMMARY_BASELINES = {
    "test_identities":    1349,
    "addr_coverage":      1348,   # identities with at least 1 address record
    "addr_coverage_pct":  99.9,
    "addr_max_unique":    10,
    "phone_max_unique":   10,
    "ktp_anomaly_count":  71,
    "npwp_anomaly_count": 59,
}


@pytest.fixture(scope="module")
def test_results():
    """Run all 4 modules on the locked test set once per session."""
    vec        = joblib.load("models/address_tfidf.pkl")
    classifier = joblib.load("models/address_classifier.pkl")

    df = pd.read_csv("data/processed/test.csv")
    out = {}
    for nid in df["nomor_identitas"].unique():
        g = df[df["nomor_identitas"] == nid]

        def vals(cat):
            return g.loc[g["element_category"] == cat, "value"].dropna().tolist()

        addr_recs  = [{"alamat":    v} for v in vals("Alamat")]
        phone_recs = [{"nomor_hp":  v} for v in vals("Nomor Telepon Seluler")]
        ktp_recs   = [{"nomor_ktp": v} for v in vals("Nomor Identitas")]
        npwp_recs  = [{"npwp":      v} for v in vals("NPWP")]

        out[str(nid)] = {
            "addr":  cluster_addresses(addr_recs, vec, classifier=classifier),
            "phone": phone_unique_count(phone_recs),
            "ktp":   ktp_uniqueness(ktp_recs),
            "npwp":  npwp_uniqueness(npwp_recs),
        }
    return out


# ── Per-identity regression tests ─────────────────────────────────────────────

class TestPerIdentityBaselines:
    @pytest.mark.parametrize("nid,expected", list(BASELINES.items()))
    def test_addr_unique(self, nid, expected, test_results):
        assert test_results[nid]["addr"] == expected["addr"], \
            f"{nid}: addr_unique changed (expected {expected['addr']})"

    @pytest.mark.parametrize("nid,expected", list(BASELINES.items()))
    def test_phone_unique(self, nid, expected, test_results):
        assert test_results[nid]["phone"] == expected["phone"], \
            f"{nid}: phone_unique changed (expected {expected['phone']})"

    @pytest.mark.parametrize("nid,expected", list(BASELINES.items()))
    def test_ktp_unique(self, nid, expected, test_results):
        ktp_count = test_results[nid]["ktp"]["ktp_unique_count"]
        assert ktp_count == expected["ktp"], \
            f"{nid}: ktp_unique changed (expected {expected['ktp']})"

    @pytest.mark.parametrize("nid,expected", list(BASELINES.items()))
    def test_npwp_unique(self, nid, expected, test_results):
        npwp_count = test_results[nid]["npwp"]["npwp_unique_count"]
        assert npwp_count == expected["npwp"], \
            f"{nid}: npwp_unique changed (expected {expected['npwp']})"


# ── Summary regression tests ──────────────────────────────────────────────────

class TestSummaryBaselines:
    def test_test_set_size(self, test_results):
        assert len(test_results) == SUMMARY_BASELINES["test_identities"]

    def test_addr_coverage(self, test_results):
        with_addr = sum(1 for v in test_results.values() if v["addr"] > 0)
        assert with_addr == SUMMARY_BASELINES["addr_coverage"]

    def test_addr_max_unique(self, test_results):
        assert max(v["addr"] for v in test_results.values()) == SUMMARY_BASELINES["addr_max_unique"]

    def test_phone_max_unique(self, test_results):
        assert max(v["phone"] for v in test_results.values()) == SUMMARY_BASELINES["phone_max_unique"]

    def test_ktp_anomaly_count(self, test_results):
        anomalies = sum(
            1 for v in test_results.values()
            if v["ktp"]["ktp_category"] == "Anomalous"
        )
        assert anomalies == SUMMARY_BASELINES["ktp_anomaly_count"]

    def test_npwp_anomaly_count(self, test_results):
        anomalies = sum(
            1 for v in test_results.values()
            if v["npwp"]["npwp_category"] == "Anomalous"
        )
        assert anomalies == SUMMARY_BASELINES["npwp_anomaly_count"]


# ── Category mapping unit tests ───────────────────────────────────────────────

class TestAddressCategory:
    def test_no_data(self):         assert address_category(0) == 0
    def test_normal(self):          assert address_category(1) == 1
    def test_low(self):             assert address_category(2) == 2
    def test_medium(self):          assert address_category(3) == 3
    def test_high_exact(self):      assert address_category(4) == 4
    def test_high_over(self):       assert address_category(7) == 4


class TestPhoneCategory:
    def test_no_data(self):         assert phone_category(0) == 0
    def test_normal(self):          assert phone_category(1) == 1
    def test_low(self):             assert phone_category(2) == 2
    def test_medium(self):          assert phone_category(3) == 3
    def test_high(self):            assert phone_category(4) == 4
    def test_high_over(self):       assert phone_category(6) == 4


class TestKtpCategoryCode:
    def test_not_available(self):
        assert ktp_category_code({"ktp_category": "Not Available"}) == 0

    def test_normal(self):
        assert ktp_category_code({"ktp_category": "Normal"}) == 1

    def test_anomalous(self):
        assert ktp_category_code({"ktp_category": "Anomalous"}) == 4

    def test_missing_key(self):
        assert ktp_category_code({}) == 0


class TestNpwpCategoryCode:
    def test_not_available(self):
        assert npwp_category_code({"npwp_category": "Not Available"}) == 0

    def test_normal(self):
        assert npwp_category_code({"npwp_category": "Normal"}) == 1

    def test_anomalous(self):
        assert npwp_category_code({"npwp_category": "Anomalous"}) == 4


class TestIdentityRiskSummary:
    def test_all_normal(self):
        ktp  = {"ktp_category": "Normal"}
        npwp = {"npwp_category": "Normal"}
        s = identity_risk_summary(1, 1, ktp, npwp)
        assert s["addr_category"]  == 1
        assert s["phone_category"] == 1
        assert s["ktp_category"]   == 1
        assert s["npwp_category"]  == 1
        assert s["max_category"]   == 1

    def test_high_address_drives_max(self):
        ktp  = {"ktp_category": "Normal"}
        npwp = {"npwp_category": "Normal"}
        s = identity_risk_summary(4, 1, ktp, npwp)
        assert s["addr_category"] == 4
        assert s["max_category"]  == 4

    def test_no_data_excluded_from_max(self):
        ktp  = {"ktp_category": "Not Available"}
        npwp = {"npwp_category": "Not Available"}
        s = identity_risk_summary(0, 0, ktp, npwp)
        assert s["max_category"] == 0

    def test_anomalous_ktp_is_high(self):
        ktp  = {"ktp_category": "Anomalous"}
        npwp = {"npwp_category": "Not Available"}
        s = identity_risk_summary(1, 1, ktp, npwp)
        assert s["ktp_category"]  == 4
        assert s["max_category"]  == 4

import pytest
from src.clustering import cluster_addresses, phone_unique_count, ktp_uniqueness, npwp_uniqueness


class TestClusterAddresses:
    def test_single_record_returns_1(self, tfidf_vec):
        assert cluster_addresses([{"alamat": "Jl. Sudirman No. 5"}], tfidf_vec) == 1

    def test_same_location_different_format_returns_1(self, tfidf_vec):
        records = [{"alamat": "Jl. Sudirman No. 5 Jakarta"},
                   {"alamat": "Jalan Sudirman Nomor 5 Jakarta"}]
        assert cluster_addresses(records, tfidf_vec) == 1

    def test_different_cities_returns_2(self, tfidf_vec):
        records = [{"alamat": "Jl. Sudirman No. 5 Jakarta"},
                   {"alamat": "Jl. Pahlawan No. 1 Surabaya"}]
        assert cluster_addresses(records, tfidf_vec) == 2

    def test_three_distinct_returns_3(self, tfidf_vec):
        records = [{"alamat": "Jl. Sudirman No. 5 Jakarta"},
                   {"alamat": "Jl. Pahlawan No. 1 Surabaya"},
                   {"alamat": "Jl. Asia Afrika No. 20 Bandung"}]
        assert cluster_addresses(records, tfidf_vec) == 3

    def test_is_deterministic(self, tfidf_vec):
        records = [{"alamat": "Jl. A Jakarta"}, {"alamat": "Jl. B Surabaya"}]
        assert cluster_addresses(records, tfidf_vec) == \
               cluster_addresses(records, tfidf_vec)

    def test_null_address_excluded(self, tfidf_vec):
        records = [{"alamat": None}, {"alamat": "Jl. Sudirman No. 5"}]
        assert cluster_addresses(records, tfidf_vec) == 1

    def test_alamat_prefix_stripped_and_clustered(self, tfidf_vec):
        # Pefindo "ALAMAT. " prefix must not prevent SAME clustering
        records = [{"alamat": "ALAMAT. DUSUN BONTOMANAI RT. 002 RW. 002"},
                   {"alamat": "DUSUN BONTOMANAI RT 2 RW 2"}]
        assert cluster_addresses(records, tfidf_vec) == 1

    def test_zero_valid_addresses_returns_0(self, tfidf_vec):
        records = [{"alamat": None}, {"alamat": ""}]
        assert cluster_addresses(records, tfidf_vec) == 0


class TestPhoneUniqueCount:
    def test_same_number_returns_1(self):
        # Already normalized to 8X format
        records = [{"nomor_hp": "81257661115"}, {"nomor_hp": "81257661115"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_same_prefix_different_last_digit_returns_1(self):
        records = [{"nomor_hp": "81234567890"}, {"nomor_hp": "81234567891"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_different_operator_returns_2(self):
        records = [{"nomor_hp": "81234567890"}, {"nomor_hp": "85612345678"}]
        assert phone_unique_count(records, prefix_length=9) == 2

    def test_raw_with_prefix_normalized(self):
        # 081257661115 and 6281257661115 are the same number
        records = [{"nomor_hp": "081257661115"}, {"nomor_hp": "6281257661115"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_malformed_excluded(self):
        records = [{"nomor_hp": "1234"}, {"nomor_hp": "81234567890"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_null_excluded(self):
        records = [{"nomor_hp": None}, {"nomor_hp": "81257661115"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_zero_placeholder_excluded(self):
        records = [{"nomor_hp": "0"}, {"nomor_hp": "81257661115"}]
        assert phone_unique_count(records, prefix_length=9) == 1

    def test_single_valid_returns_1(self):
        assert phone_unique_count([{"nomor_hp": "81257661115"}], prefix_length=9) == 1

    def test_all_invalid_returns_0(self):
        records = [{"nomor_hp": "0"}, {"nomor_hp": None}]
        assert phone_unique_count(records, prefix_length=9) == 0


class TestKtpUniqueness:
    def test_one_ktp_normal(self):
        r = ktp_uniqueness([{"nomor_ktp": "3171012345678901"}] * 3)
        assert r["ktp_unique_count"] == 1
        assert r["ktp_category"] == "Normal"

    def test_two_ktps_anomalous(self):
        r = ktp_uniqueness([{"nomor_ktp": "3171012345678901"},
                            {"nomor_ktp": "3172019876543210"}])
        assert r["ktp_unique_count"] == 2
        assert r["ktp_category"] == "Anomalous"

    def test_near_duplicate_flagged(self):
        r = ktp_uniqueness([{"nomor_ktp": "3171012345678901"},
                            {"nomor_ktp": "3171012345678902"}])
        assert len(r["near_duplicates"]) > 0

    def test_null_excluded(self):
        r = ktp_uniqueness([{"nomor_ktp": None}, {"nomor_ktp": "3171012345678901"}])
        assert r["ktp_unique_count"] == 1

    def test_all_null_returns_zero(self):
        r = ktp_uniqueness([{"nomor_ktp": None}])
        assert r["ktp_unique_count"] == 0
        assert r["ktp_category"] == "Not Available"


class TestNpwpUniqueness:
    def test_one_npwp_normal(self):
        r = npwp_uniqueness([{"npwp": "730612481295000"}] * 2)
        assert r["npwp_unique_count"] == 1
        assert r["npwp_category"] == "Normal"

    def test_two_npwps_anomalous(self):
        r = npwp_uniqueness([{"npwp": "730612481295000"},
                             {"npwp": "123456789012345"}])
        assert r["npwp_unique_count"] == 2
        assert r["npwp_category"] == "Anomalous"

    def test_zero_placeholder_excluded(self):
        r = npwp_uniqueness([{"npwp": "0"}, {"npwp": "730612481295000"}])
        assert r["npwp_unique_count"] == 1

    def test_all_zeros_excluded(self):
        r = npwp_uniqueness([{"npwp": "00000000000000"}])
        assert r["npwp_unique_count"] == 0
        assert r["npwp_category"] == "Not Available"

    def test_null_excluded(self):
        r = npwp_uniqueness([{"npwp": None}, {"npwp": "730612481295000"}])
        assert r["npwp_unique_count"] == 1

    def test_near_duplicate_flagged(self):
        r = npwp_uniqueness([{"npwp": "730612481295000"},
                             {"npwp": "730612481295001"}])
        assert len(r["near_duplicates"]) > 0

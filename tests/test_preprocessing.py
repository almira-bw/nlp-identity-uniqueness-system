import pytest
from src.preprocessing import normalize_address, normalize_phone, normalize_id_number


class TestNormalizeAddress:
    def test_expands_jl(self):
        assert "jalan" in normalize_address("Jl. Sudirman No. 5")

    def test_expands_no(self):
        assert "nomor" in normalize_address("Jl. Sudirman No. 5")

    def test_expands_kec(self):
        assert "kecamatan" in normalize_address("Kec. Menteng")

    def test_expands_kel(self):
        assert "kelurahan" in normalize_address("Kel. Gondangdia")

    def test_lowercased(self):
        assert normalize_address("JL. SUDIRMAN") == normalize_address("jl. sudirman")

    def test_rt_rw_leading_zeros_stripped(self):
        r1 = normalize_address("RT 001 RW 002")
        r2 = normalize_address("RT.001/RW.002")
        r3 = normalize_address("RT1/RW2")
        assert "rt 1" in r1 and "rw 2" in r1
        assert r1 == r2 == r3

    def test_stopwords_removed(self):
        result = normalize_address("Jalan di Sudirman dari Jakarta")
        assert "di" not in result.split()
        assert "dari" not in result.split()

    def test_null_returns_none(self):
        assert normalize_address(None) is None
        assert normalize_address("") is None

    def test_punctuation_stripped(self):
        result = normalize_address("Jln. Gatot Subroto, Kav. 50-51")
        assert "," not in result

    def test_same_location_high_similarity(self):
        from rapidfuzz import fuzz
        a1 = normalize_address("Jl. Merdeka No. 10, Kec. Gambir, Jakarta Pusat")
        a2 = normalize_address("Jalan Merdeka Nomor 10 Kecamatan Gambir Jakarta Pusat")
        assert fuzz.token_set_ratio(a1, a2) >= 95

    def test_alamat_prefix_stripped(self):
        # Pefindo data often has "ALAMAT. " prefix
        r1 = normalize_address("ALAMAT. DUSUN BONTOMANAI RT. 002 RW. 002")
        r2 = normalize_address("DUSUN BONTOMANAI RT. 002 RW. 002")
        assert r1 == r2

    def test_zero_rt_rw_normalized(self):
        # RT 000 / RW 000 appears in data — should normalize to rt 0 rw 0
        result = normalize_address("RT. 000 RW. 000")
        assert "rt 0" in result and "rw 0" in result

    def test_disun_typo_preserved(self):
        # Typo "DISUN" (should be DUSUN) — we do not auto-correct typos, just normalize case
        result = normalize_address("DISUN BONTOMANAI")
        assert "disun" in result


class TestNormalizePhone:
    def test_strips_plus62(self):
        assert normalize_phone("+62 812-3456-7890") == "81234567890"

    def test_strips_62_prefix(self):
        assert normalize_phone("628123456789") == "8123456789"

    def test_strips_formatting(self):
        assert normalize_phone("0812 3456 7890") == "81234567890"
        assert normalize_phone("(0812) 3456-7890") == "81234567890"

    def test_too_short_is_none(self):
        assert normalize_phone("1234") is None

    def test_too_long_is_none(self):
        assert normalize_phone("0812345678901234") is None

    def test_null_is_none(self):
        assert normalize_phone(None) is None

    def test_zero_string_is_none(self):
        # "0" appears as placeholder in Pefindo data
        assert normalize_phone("0") is None

    def test_strips_plus6281_format(self):
        # 6281257661115 appears in data (missing leading +)
        assert normalize_phone("6281257661115") == "81257661115"

    def test_strips_leading_zeros_double_zero(self):
        # 0085240444 — after stripping single 0 → 085240444, starts with 0 not 8
        assert normalize_phone("0085240444") is None


class TestNormalizeIdNumber:
    def test_ktp_strips_spaces(self):
        assert normalize_id_number("3171 0123 4567 8901", 16) == "3171012345678901"

    def test_npwp_strips_dots_dashes(self):
        assert normalize_id_number("12.345.678.9-012.345", 15) == "123456789012345"

    def test_wrong_length_is_none(self):
        assert normalize_id_number("123456", 16) is None

    def test_null_is_none(self):
        assert normalize_id_number(None, 16) is None

    def test_all_zeros_ktp_is_none(self):
        # "0" placeholder — invalid after normalization
        assert normalize_id_number("0", 16) is None

    def test_all_zeros_npwp_is_none(self):
        # "00000000000000" (14 zeros) appears in data — wrong length for NPWP (15)
        assert normalize_id_number("00000000000000", 15) is None

    def test_valid_npwp(self):
        assert normalize_id_number("730612481295000", 15) == "730612481295000"

    def test_valid_ktp(self):
        assert normalize_id_number("7306124812950003", 16) == "7306124812950003"

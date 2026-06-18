import re
from typing import Optional
import pandas as pd

ADDRESS_ABBREV = {
    r"\bjl\b\.?": "jalan",
    r"\bjln\b\.?": "jalan",
    r"\bno\b\.?": "nomor",
    r"\bnr\b\.?": "nomor",
    r"\bgg\b\.?": "gang",
    r"\bkec\b\.?": "kecamatan",
    r"\bkel\b\.?": "kelurahan",
    r"\bklh\b\.?": "kelurahan",
    r"\bkab\b\.?": "kabupaten",
    r"\bprov\b\.?": "provinsi",
    r"\bkomp\b\.?": "kompleks",
    r"\bperum\b\.?": "perumahan",
    r"\bds\b\.?": "desa",
    r"\bvil\b\.?": "villa",
    r"\brt\s*\.?\s*0*(\d+)": r"rt \1",
    r"\brw\s*\.?\s*0*(\d+)": r"rw \1",
}

ADDRESS_STOPWORDS = {"di", "ke", "dari", "yang", "dan"}

# Pefindo data often prefixes address values with "ALAMAT. " or "ALAMAT "
_ALAMAT_PREFIX = re.compile(r"^alamat\.?\s*", re.IGNORECASE)


def normalize_address(raw: str) -> Optional[str]:
    """Normalize Indonesian address: strip Pefindo prefix, expand abbreviations, remove stopwords/punctuation."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
        return None
    text = str(raw).lower().strip()
    text = _ALAMAT_PREFIX.sub("", text)
    if not text:
        return None
    for pattern, replacement in ADDRESS_ABBREV.items():
        text = re.sub(pattern, replacement, text)
    tokens = [t for t in text.split() if t not in ADDRESS_STOPWORDS]
    text = re.sub(r"[^\w\s]", " ", " ".join(tokens))
    return re.sub(r"\s+", " ", text).strip() or None


def normalize_phone(raw: str) -> Optional[str]:
    """Clean Indonesian mobile phone to 8XXXXXXXXX (strip 0/62 prefix). Returns None if malformed or placeholder."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    digits = re.sub(r"\D", "", str(raw))
    if not digits or set(digits) == {"0"}:
        return None
    # Strip country code 62 or domestic prefix 0 — result must start with 8
    if digits.startswith("62"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    # Indonesian mobile numbers start with 8X
    if not digits.startswith("8"):
        return None
    return digits if 9 <= len(digits) <= 12 else None


def normalize_id_number(raw: str, expected_length: int) -> Optional[str]:
    """Strip non-digits from KTP (16 digits) or NPWP (15 digits). Returns None if length mismatch or all-zero."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    digits = re.sub(r"\D", "", str(raw))
    if not digits or set(digits) == {"0"}:
        return None
    return digits if len(digits) == expected_length else None

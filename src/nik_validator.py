from datetime import datetime
from typing import Optional

VALID_PROVINCE_CODES = set(range(11, 100))


def validate_nik(nik_clean: str, gender: Optional[str] = None) -> dict:
    """Validate Indonesian NIK structural integrity (province, date, gender consistency)."""
    result = {"valid": True, "issues": []}
    if not nik_clean or len(nik_clean) != 16:
        result["valid"] = False
        result["issues"].append(f"length={len(nik_clean) if nik_clean else 0}")
        return result

    prov = int(nik_clean[:2])
    if prov not in VALID_PROVINCE_CODES:
        result["valid"] = False
        result["issues"].append(f"invalid_province={prov}")

    day_raw = int(nik_clean[6:8])
    month   = int(nik_clean[8:10])
    yr2     = int(nik_clean[10:12])
    female  = day_raw > 40
    day     = day_raw - 40 if female else day_raw
    inferred_gender = "F" if female else "M"

    if gender and gender.upper() != inferred_gender:
        result["issues"].append(
            f"gender_mismatch:nik={inferred_gender},profile={gender.upper()}"
        )

    year = 2000 + yr2 if yr2 <= 24 else 1900 + yr2
    try:
        datetime(year, month, day)
    except ValueError:
        result["valid"] = False
        result["issues"].append(f"invalid_date:{day}/{month}/{year}")

    return result

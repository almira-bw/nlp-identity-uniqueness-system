"""
Maps module raw outputs (unique counts / category strings) to
numeric risk categories used in the final identity score.

Category scale (from config.json):
  1 = Normal   — single unique value, no anomaly
  2 = Low      — 2 distinct values
  3 = Medium   — 3 distinct values
  4 = High     — 4+ distinct values  /  ID-number anomaly
  0 = No Data  — module has no records for this identity
"""
import json
from typing import Union

with open("config.json") as f:
    _CFG = json.load(f)


def address_category(unique_count: int) -> int:
    """
    Convert address unique cluster count → risk category.
      0       → 0 (no data)
      1       → 1 (Normal)
      2       → 2 (Low)
      3       → 3 (Medium)
      4+      → 4 (High)
    """
    if unique_count == 0:
        return 0
    cats = _CFG["address"]["categories"]
    if unique_count == 1:
        return cats["normal"]
    if unique_count == 2:
        return cats["low"]
    if unique_count == 3:
        return cats["medium"]
    return cats["high"]


def phone_category(unique_count: int) -> int:
    """
    Convert phone unique cluster count → risk category.
    Same scale as address.
    """
    if unique_count == 0:
        return 0
    cats = _CFG["phone"]["categories"]
    if unique_count == 1:
        return cats["normal"]
    if unique_count == 2:
        return cats["low"]
    if unique_count == 3:
        return cats["medium"]
    return cats["high"]


def ktp_category_code(ktp_result: dict) -> int:
    """
    Convert ktp_uniqueness() result → risk category code.
      Not Available   → 0
      Normal (1 KTP)  → 1
      Anomalous       → 4  (multiple distinct KTPs is high risk)
    """
    cat = ktp_result.get("ktp_category", "Not Available")
    if cat == "Not Available":
        return 0
    if cat == "Normal":
        return _CFG["address"]["categories"]["normal"]
    return _CFG["address"]["categories"]["high"]


def npwp_category_code(npwp_result: dict) -> int:
    """
    Convert npwp_uniqueness() result → risk category code.
      Not Available   → 0
      Normal (1 NPWP) → 1
      Anomalous       → 4
    """
    cat = npwp_result.get("npwp_category", "Not Available")
    if cat == "Not Available":
        return 0
    if cat == "Normal":
        return _CFG["address"]["categories"]["normal"]
    return _CFG["address"]["categories"]["high"]


def identity_risk_summary(addr_unique: int,
                           phone_unique: int,
                           ktp_result: dict,
                           npwp_result: dict) -> dict:
    """
    Return a summary dict with category codes for all 4 modules.
    max_category is the highest risk signal observed.
    """
    codes = {
        "addr_category":  address_category(addr_unique),
        "phone_category": phone_category(phone_unique),
        "ktp_category":   ktp_category_code(ktp_result),
        "npwp_category":  npwp_category_code(npwp_result),
    }
    data_codes = [v for v in codes.values() if v > 0]
    codes["max_category"] = max(data_codes) if data_codes else 0
    return codes

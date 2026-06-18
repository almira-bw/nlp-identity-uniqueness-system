import os
import pytest


def test_model_selection_report_created():
    from src.model_selection import run_all_module_comparisons
    run_all_module_comparisons(
        address_val_pairs="data/labeled/address_pairs_to_label.csv",
        phone_val_pairs="data/labeled/phone_pairs_labeled.csv",
    )
    assert os.path.exists("logs/model_selection_report.json")


def test_selected_methods_saved_to_config():
    import json
    with open("config.json") as f:
        cfg = json.load(f)
    assert "selected_address_method" in cfg
    assert "selected_phone_prefix" in cfg


def test_address_best_method_precision_above_floor():
    import json
    with open("logs/model_selection_report.json") as f:
        report = json.load(f)
    assert report["address"]["best_val_precision"] >= 0.82, \
        f"Best address precision {report['address']['best_val_precision']:.3f} below 0.82"


def test_phone_best_method_precision_above_floor():
    import json
    with open("logs/model_selection_report.json") as f:
        report = json.load(f)
    assert report["phone"]["best_val_precision"] >= 0.85, \
        f"Best phone precision {report['phone']['best_val_precision']:.3f} below 0.85"

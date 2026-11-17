import pandas as pd

from src.classifiers import LABELS
from src.data_quality import run_data_quality_checks


def test_clean_dataset_passes():
    df = pd.DataFrame(
        [
            {"id": "1", "query": "reset password", "gold_label": "account"},
            {"id": "2", "query": "where is my parcel", "gold_label": "delivery"},
        ]
    )
    report = run_data_quality_checks(df, LABELS)
    assert report.passed is True


def test_duplicate_id_fails():
    df = pd.DataFrame(
        [
            {"id": "1", "query": "reset password", "gold_label": "account"},
            {"id": "1", "query": "request refund", "gold_label": "refund"},
        ]
    )
    report = run_data_quality_checks(df, LABELS)
    assert report.passed is False
    assert report.duplicate_id_count == 1

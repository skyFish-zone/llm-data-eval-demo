from pathlib import Path

import pandas as pd

from src.sql_reporting import build_sql_report


def test_sql_report_aggregates_error_rate(tmp_path: Path):
    predictions = pd.DataFrame(
        [
            {"gold_label": "account", "prediction": "account", "is_correct": True},
            {"gold_label": "account", "prediction": "refund", "is_correct": False},
            {"gold_label": "refund", "prediction": "refund", "is_correct": True},
        ]
    )
    sql_path = tmp_path / "report.sql"
    sql_path.write_text(
        """
        SELECT gold_label,
               COUNT(*) AS sample_count,
               SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS badcase_count,
               ROUND(1.0 * SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS error_rate
        FROM predictions
        GROUP BY gold_label
        ORDER BY error_rate DESC, gold_label
        """,
        encoding="utf-8",
    )
    report = build_sql_report(predictions, tmp_path, sql_path)
    assert report.iloc[0]["gold_label"] == "account"
    assert report.iloc[0]["error_rate"] == 0.5

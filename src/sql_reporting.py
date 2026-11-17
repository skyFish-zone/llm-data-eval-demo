from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def build_sql_report(
    predictions: pd.DataFrame,
    output_dir: Path,
    sql_path: Path = Path("sql/label_error_rate.sql"),
) -> pd.DataFrame:
    """Persist row-level predictions and run a small SQL quality report.

    SQLite keeps the demo self-contained while still making the aggregation
    logic explicit and inspectable as SQL.
    """
    db_path = output_dir / "evaluation.db"
    with sqlite3.connect(db_path) as conn:
        db_df = predictions.copy()
        db_df["is_correct"] = db_df["is_correct"].astype(int)
        db_df.to_sql("predictions", conn, if_exists="replace", index=False)
        sql = sql_path.read_text(encoding="utf-8")
        report = pd.read_sql_query(sql, conn)

    report.to_csv(output_dir / "label_error_rate.csv", index=False)
    return report

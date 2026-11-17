from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"id", "query", "gold_label"}


@dataclass
class DataQualityReport:
    row_count: int
    missing_columns: list[str]
    duplicate_id_count: int
    duplicate_query_count: int
    missing_query_count: int
    missing_label_count: int
    invalid_label_count: int
    empty_query_count: int
    min_query_length: int
    max_query_length: int
    label_distribution: dict[str, int]

    @property
    def passed(self) -> bool:
        return not any(
            [
                self.missing_columns,
                self.duplicate_id_count,
                self.missing_query_count,
                self.missing_label_count,
                self.invalid_label_count,
                self.empty_query_count,
            ]
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def run_data_quality_checks(
    df: pd.DataFrame,
    allowed_labels: Iterable[str],
) -> DataQualityReport:
    allowed = set(allowed_labels)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing_columns:
        return DataQualityReport(
            row_count=len(df),
            missing_columns=missing_columns,
            duplicate_id_count=0,
            duplicate_query_count=0,
            missing_query_count=0,
            missing_label_count=0,
            invalid_label_count=0,
            empty_query_count=0,
            min_query_length=0,
            max_query_length=0,
            label_distribution={},
        )

    query_text = df["query"].fillna("").astype(str).str.strip()
    labels = df["gold_label"].fillna("").astype(str).str.strip()
    lengths = query_text.str.len()

    return DataQualityReport(
        row_count=len(df),
        missing_columns=[],
        duplicate_id_count=int(df["id"].duplicated().sum()),
        duplicate_query_count=int(query_text.duplicated().sum()),
        missing_query_count=int(df["query"].isna().sum()),
        missing_label_count=int(df["gold_label"].isna().sum()),
        invalid_label_count=int((~labels.isin(allowed)).sum()),
        empty_query_count=int((query_text == "").sum()),
        min_query_length=int(lengths.min()) if len(lengths) else 0,
        max_query_length=int(lengths.max()) if len(lengths) else 0,
        label_distribution={
            str(k): int(v) for k, v in labels.value_counts().sort_index().items()
        },
    )

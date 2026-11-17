from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationArtifacts:
    summary: dict
    predictions: pd.DataFrame
    badcases: pd.DataFrame
    confusion: pd.DataFrame


def evaluate_predictions(
    df: pd.DataFrame,
    predictions: list[str],
    labels: list[str],
) -> EvaluationArtifacts:
    result = df.copy()
    result["prediction"] = predictions
    result["is_correct"] = result["gold_label"] == result["prediction"]

    report = classification_report(
        result["gold_label"],
        result["prediction"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "row_count": int(len(result)),
        "accuracy": float(accuracy_score(result["gold_label"], result["prediction"])),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "badcase_count": int((~result["is_correct"]).sum()),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }

    matrix = confusion_matrix(
        result["gold_label"], result["prediction"], labels=labels
    )
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "gold_label"
    confusion.columns.name = "predicted_label"

    badcases = result.loc[~result["is_correct"]].copy()
    return EvaluationArtifacts(summary, result, badcases, confusion)

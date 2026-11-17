import pandas as pd

from src.evaluation import evaluate_predictions


def test_evaluation_finds_badcase():
    labels = ["account", "refund"]
    df = pd.DataFrame(
        [
            {"id": "1", "query": "reset password", "gold_label": "account"},
            {"id": "2", "query": "money back", "gold_label": "refund"},
        ]
    )
    artifacts = evaluate_predictions(df, ["account", "account"], labels)
    assert artifacts.summary["accuracy"] == 0.5
    assert artifacts.summary["badcase_count"] == 1
    assert len(artifacts.badcases) == 1

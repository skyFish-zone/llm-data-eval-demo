from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.classifiers import LABELS, OpenAICompatibleClassifier, RuleBasedClassifier
from src.data_quality import run_data_quality_checks
from src.evaluation import evaluate_predictions
from src.sql_reporting import build_sql_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an intent classifier on synthetic data.")
    parser.add_argument("--input", default="data/sample_queries.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--classifier", choices=["rule", "llm"], default="rule")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    dq = run_data_quality_checks(df, LABELS)
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(dq.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not dq.passed:
        raise SystemExit("Data quality checks failed. See outputs/data_quality_report.json")

    classifier = RuleBasedClassifier() if args.classifier == "rule" else OpenAICompatibleClassifier()
    predictions = [classifier.predict(query) for query in df["query"].astype(str)]
    artifacts = evaluate_predictions(df, predictions, LABELS)

    (output_dir / "summary.json").write_text(
        json.dumps(artifacts.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    artifacts.predictions.to_csv(output_dir / "predictions.csv", index=False)
    artifacts.badcases.to_csv(output_dir / "badcases.csv", index=False)
    artifacts.confusion.to_csv(output_dir / "confusion_matrix.csv")
    build_sql_report(artifacts.predictions, output_dir)

    print(json.dumps(artifacts.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

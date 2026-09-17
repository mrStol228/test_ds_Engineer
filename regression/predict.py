from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from common import add_engineered_features


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--input-path", default="data/hidden_test.csv")
    parser.add_argument("--output-path", default="predictions/hidden_test_predictions.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    bundle = joblib.load(args.model_path)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    df = pd.read_csv(args.input_path)
    df = add_engineered_features(df)
    predictions = model.predict(df[feature_columns])

    output = pd.DataFrame({"id": range(len(predictions)), "target": predictions})
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()

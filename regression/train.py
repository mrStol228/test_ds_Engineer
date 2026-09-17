from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from common import TARGET_COLUMN, add_engineered_features, get_feature_columns

MODEL_BUILDERS = {
    "linear": lambda random_state: LinearRegression(),
    "rf": lambda random_state: RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=random_state
    ),
}

try:
    from lightgbm import LGBMRegressor

    MODEL_BUILDERS["lgbm"] = lambda random_state: LGBMRegressor(
        n_estimators=500, learning_rate=0.05, random_state=random_state, verbosity=-1
    )
except ImportError:
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--model-type", choices=sorted(MODEL_BUILDERS), default="linear")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def cross_validate(model_type: str, X: pd.DataFrame, y: pd.Series, folds: int, random_state: int) -> np.ndarray:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kfold.split(X):
        model = MODEL_BUILDERS[model_type](random_state)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(root_mean_squared_error(y.iloc[val_idx], preds))
    return np.array(scores)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    df = pd.read_csv(args.train_path)
    df = add_engineered_features(df)
    feature_columns = get_feature_columns(df)
    X, y = df[feature_columns], df[TARGET_COLUMN]

    cv_scores = cross_validate(args.model_type, X, y, args.cv_folds, args.random_state)
    print(
        f"[{args.model_type}] {args.cv_folds}-fold CV RMSE: "
        f"{cv_scores.mean():.6f} +/- {cv_scores.std():.6f}"
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    holdout_model = MODEL_BUILDERS[args.model_type](args.random_state)
    holdout_model.fit(X_tr, y_tr)
    holdout_rmse = root_mean_squared_error(y_val, holdout_model.predict(X_val))
    print(f"[{args.model_type}] Holdout RMSE ({args.test_size:.0%} held out): {holdout_rmse:.6f}")

    final_model = MODEL_BUILDERS[args.model_type](args.random_state)
    final_model.fit(X, y)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_columns": feature_columns, "model_type": args.model_type},
        model_path,
    )
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()

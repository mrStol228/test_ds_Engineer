# Regression on Tabular Data

`train.csv` has 53 anonymized columns and a target; predict the target for
`hidden_test.csv`, scored by RMSE.

Almost none of the 53 columns actually matter. Linear correlation with the
target is near zero across the board, but a quick Random Forest shows
feature importance piling up on just two columns. Squaring one of them and
adding the other reproduces the target almost exactly (residual around
1e-13 on all 90k rows). `notebooks/eda.ipynb` has the full walkthrough of
how that turned up.

## Setup

    cd regression
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Drop `train.csv` / `hidden_test.csv` into `data/`.

## Running it

    python train.py                  # trains, prints RMSE, saves models/model.joblib
    python train.py --model-type rf  # or lgbm, for comparison
    python predict.py                # writes predictions/hidden_test_predictions.csv

Default model is plain linear regression on the engineered feature — RMSE
comes out to ~0. Random Forest and LightGBM are kept around as a fallback /
sanity check in case the actual hidden test isn't generated the same way.

`--help` on either script shows the rest of the options (paths, cv folds,
random seed, etc).

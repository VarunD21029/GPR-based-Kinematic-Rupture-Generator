"""
Residual correlation matrix of the GPR scaling parameters
=========================================================

Computes the correlation matrix of the GPR residuals, i.e. the differences
between the observed (log10) values of the scaling parameters in the training
database and the GPR predictions for the same events, and writes it to
models/gpr_residual_correlation.csv.

The matrix describes how the scaling parameters co-vary from event to event
(e.g. an event wider than predicted tends to have a smaller mean slip than
predicted). run_pipeline(..., n_samples=N) uses it to draw the parameters
jointly when generating an ensemble of slip realizations.

Usage
-----
    python src/compute_residual_correlation.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
OUT_FILE = os.path.join(MODEL_DIR, "gpr_residual_correlation.csv")


def compute_residual_correlation():
    model = joblib.load(os.path.join(MODEL_DIR, "gpr_model.pkl"))
    mw_scaler = joblib.load(os.path.join(MODEL_DIR, "gpr_mw_scaler.pkl"))
    ohe = joblib.load(os.path.join(MODEL_DIR, "gpr_ohe.pkl"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "gpr_metadata.pkl"))
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "gpr_y_scaler.pkl"))

    target_cols = list(metadata["target_cols"])
    Y_obs = np.asarray(metadata["y_all"], dtype=float)                       # events x targets, log10
    X = np.hstack([mw_scaler.transform(np.asarray(metadata["Mw_original"], dtype=float).reshape(-1, 1)),
                   ohe.transform(metadata["X_cat"])])
    Y_pred = np.column_stack([est.predict(X) for est in model.estimators_]) * y_scaler.scale_ + y_scaler.mean_
    residuals = pd.DataFrame(Y_obs - Y_pred, columns=target_cols)
    return residuals.corr()


if __name__ == "__main__":
    corr = compute_residual_correlation()
    corr.to_csv(OUT_FILE, float_format="%.6f")
    print(f"Wrote {OUT_FILE} ({corr.shape[0]} x {corr.shape[1]})")

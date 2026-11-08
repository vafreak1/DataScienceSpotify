"""
Small helpers shared by all model notebooks (04 - 08), so each notebook
only contains the part that is specific to its model.
"""

import os
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import build_features

PRED_DIR = "predictions"  # every model saves its validation and test predictions here


def load_features():
    """Load the data, apply the fixed split from notebook 02 and build the features from features.py."""
    train = pd.read_csv("train.csv").merge(pd.read_csv("splits.csv"), on="id")
    test = pd.read_csv("test.csv")
    dev, valid = train[train["is_valid"] == 0], train[train["is_valid"] == 1]

    # learn the target encodings on dev, apply them to valid and test
    X_dev, (X_valid, X_test) = build_features(dev, [valid, test], all_rows=pd.concat([train, test]))
    return dev, valid, test, X_dev, X_valid, X_test


def baseline_auc():
    """The score to beat: best single feature on the validation set (artist_followers)."""
    return pd.read_csv("baseline_scores.csv")["valid_auc"].max()


def compare_to_baseline(model_name, y_valid, valid_pred):
    """Print the validation AUC of a model next to the baseline."""
    auc = roc_auc_score(y_valid, valid_pred)
    base = baseline_auc()
    print(f"{model_name}: validation AUC = {auc:.4f}")
    print(f"baseline (artist_followers alone)  = {base:.4f}")
    print(f"improvement                        = {auc - base:+.4f}")
    return auc


def save_predictions(model_name, valid, valid_pred, test, test_pred):
    """Save predictions so the ensemble notebook (08) can combine all models."""
    os.makedirs(PRED_DIR, exist_ok=True)
    pd.DataFrame({"id": valid["id"].values, "pred": valid_pred}).to_csv(f"{PRED_DIR}/{model_name}_valid.csv", index=False)
    pd.DataFrame({"id": test["id"].values, "pred": test_pred}).to_csv(f"{PRED_DIR}/{model_name}_test.csv", index=False)

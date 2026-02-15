"""
ML inference for holdout ANES variables (inferred_by_ml stance level).
Trains classifiers on ANES variables NOT used in clustering, using the 10 clustering features as predictors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple


# ============================================================================
# HOLDOUT ANES VARIABLES (pre-core, NOT in clustering features)
# ============================================================================
# These are topics measured in ANES but not used for clustering.
# We'll train ML models to predict cluster stances on these topics.

HOLDOUT_VARIABLES = {
    "V241177": {
        "desc": "Self ideology (1=liberal, 7=conservative)",
        "flip": False,
        "topic": "ideology",
        "type": "ordinal"
    },
    "V241725": {
        "desc": "Religious attendance (1=never, 6=weekly+)",
        "flip": False,
        "topic": "religious_attendance",
        "type": "ordinal"
    },
    "V241252": {
        "desc": "Guaranteed jobs/income (1=gov ensures, 7=people on their own)",
        "flip": False,  # Higher = individualist (conservative)
        "topic": "guaranteed_jobs",
        "type": "ordinal"
    },
    "V241381x": {
        "desc": "Gay marriage (summary scale, higher = oppose)",
        "flip": False,
        "topic": "gay_marriage",
        "type": "ordinal"
    },
    "V241741": {
        "desc": "Military assistance to Israel (1=favor, 3=neither, 2=oppose)",
        "map": {1: 1, 3: 3, 2: 5},  # Recode to scale
        "flip": True,  # Flip so higher = more support
        "topic": "israel_aid",
        "type": "ordinal"
    },
    "V241740": {
        "desc": "Weapons to Ukraine (1=favor, 3=neither, 2=oppose)",
        "map": {1: 1, 3: 3, 2: 5},
        "flip": True,  # Flip so higher = more support
        "topic": "ukraine_aid",
        "type": "ordinal"
    },
}


def clean_holdout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform holdout variables.
    """
    from helpers.preprocessing_v2 import clean_anes_column, flip_series

    Y = pd.DataFrame(index=df.index)

    for var, spec in HOLDOUT_VARIABLES.items():
        if var not in df.columns:
            continue

        s = clean_anes_column(df[var])

        # Apply map if present
        if "map" in spec:
            s = s.map(spec["map"])

        # Apply flip if needed
        if spec.get("flip", False):
            s = flip_series(s)

        Y[var] = s

    return Y


def train_holdout_predictors(X_train: np.ndarray, Y_train: pd.DataFrame, feature_names: List[str]) -> Dict:
    """
    Train classifiers for each holdout variable.

    Returns dict mapping variable -> {model, cv_score, predictions_by_cluster}
    """
    results = {}

    for var, spec in HOLDOUT_VARIABLES.items():
        if var not in Y_train.columns:
            continue

        y = Y_train[var].values
        valid = ~np.isnan(y)

        if valid.sum() < 50:  # Need minimum samples
            continue

        X_valid = X_train[valid]
        y_valid = y[valid]

        # Train random forest (handles ordinal reasonably well)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_valid, y_valid)

        # Cross-validation score
        cv_scores = cross_val_score(rf, X_valid, y_valid, cv=3, scoring='accuracy')
        cv_mean = float(cv_scores.mean())

        results[var] = {
            "model": rf,
            "cv_score": cv_mean,
            "topic": spec["topic"],
            "desc": spec["desc"],
            "feature_names": feature_names
        }

    return results


def predict_cluster_stances(models: Dict, cluster_centroids: np.ndarray) -> Dict:
    """
    Predict cluster-level stances for each holdout topic.

    Returns dict: {topic: {cluster_id: predicted_value}}
    """
    predictions = {}

    for var, model_info in models.items():
        model = model_info["model"]
        topic = model_info["topic"]

        # Predict for each cluster centroid
        preds = model.predict(cluster_centroids)

        predictions[topic] = {
            "variable": var,
            "desc": model_info["desc"],
            "cv_score": model_info["cv_score"],
            "cluster_predictions": {int(i): float(preds[i]) for i in range(len(preds))}
        }

    return predictions

"""
Feature Selection for Quiz Variables
======================================
Uses Random Forest classifier to identify the top 10 variables that best predict
cluster membership. These variables are used in the quiz, while the 3D cube shows
all clustering variables.

The goal is to minimize quiz length while maximizing predictive power.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def select_quiz_features(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    n_features: int = 10,
    random_state: int = 42
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select top N features for quiz using Random Forest feature importance.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features)
    labels : np.ndarray
        Cluster labels (n_samples,)
    feature_names : List[str]
        Feature names corresponding to X_scaled columns
    n_features : int
        Number of features to select for quiz (default: 10)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    selected_features : List[str]
        Top N feature names ranked by importance
    feature_importance : Dict[str, float]
        All features with their importance scores
    """

    # Train Random Forest classifier to predict cluster membership
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        random_state=random_state,
        n_jobs=-1
    )

    rf.fit(X_scaled, labels)

    # Get feature importance scores
    importances = rf.feature_importances_

    # Create dict of feature -> importance
    feature_importance = {
        feature_names[i]: float(importances[i])
        for i in range(len(feature_names))
    }

    # Rank by importance (descending)
    ranked_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top N
    selected_features = [feat for feat, _ in ranked_features[:n_features]]

    # Compute cross-validated accuracy as quality metric
    cv_scores = cross_val_score(rf, X_scaled, labels, cv=5, n_jobs=-1)
    cv_accuracy = float(np.mean(cv_scores))

    print(f"\n  Feature Selection Results:")
    print(f"  - RF cross-validated accuracy: {cv_accuracy:.3f}")
    print(f"  - Top {n_features} features selected for quiz:")
    for i, (feat, importance) in enumerate(ranked_features[:n_features], 1):
        print(f"    {i:2d}. {feat}: {importance:.4f}")

    return selected_features, feature_importance


def validate_quiz_features(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    selected_features: List[str],
    random_state: int = 42
) -> Dict[str, float]:
    """
    Validate that the selected quiz features maintain good predictive accuracy.

    Returns dictionary with:
    - full_accuracy: CV accuracy using all features
    - quiz_accuracy: CV accuracy using only quiz features
    - accuracy_ratio: quiz_accuracy / full_accuracy
    """

    # Get indices of selected features
    selected_indices = [i for i, feat in enumerate(feature_names) if feat in selected_features]
    X_quiz = X_scaled[:, selected_indices]

    # Train on all features
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    cv_full = cross_val_score(rf_full, X_scaled, labels, cv=5, n_jobs=-1)
    full_accuracy = float(np.mean(cv_full))

    # Train on quiz features only
    rf_quiz = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    cv_quiz = cross_val_score(rf_quiz, X_quiz, labels, cv=5, n_jobs=-1)
    quiz_accuracy = float(np.mean(cv_quiz))

    accuracy_ratio = quiz_accuracy / full_accuracy if full_accuracy > 0 else 0.0

    print(f"\n  Quiz Feature Validation:")
    print(f"  - Full model accuracy (all {len(feature_names)} features): {full_accuracy:.3f}")
    print(f"  - Quiz model accuracy (top {len(selected_features)} features): {quiz_accuracy:.3f}")
    print(f"  - Retention ratio: {accuracy_ratio:.1%}")

    if accuracy_ratio < 0.85:
        print(f"  WARNING: Quiz features retain <85% of full model accuracy")
    else:
        print(f"  OK: Quiz features retain >=85% of predictive power")

    return {
        "full_accuracy": full_accuracy,
        "quiz_accuracy": quiz_accuracy,
        "accuracy_ratio": accuracy_ratio,
        "n_features_full": len(feature_names),
        "n_features_quiz": len(selected_features)
    }

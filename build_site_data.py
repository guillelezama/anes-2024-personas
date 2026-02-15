"""
ANES 2024 Clustering + Static Site Data Generator
==================================================
Produces all JSON/CSV/HTML artifacts for GitHub Pages deployment.

Usage:
    python build_site_data.py --universe likely_voters
    python build_site_data.py --universe actual_voters
    python build_site_data.py --universe all_respondents --use-llm

Flags:
    --universe: "likely_voters" (default), "actual_voters", or "all_respondents"
    --use-llm: Enable LangChain for persona/story generation (requires API key in env)
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Import helpers
sys.path.insert(0, str(Path(__file__).parent))
from helpers.preprocessing_v2 import (
    CLUSTERING_FEATURES_V2 as CLUSTERING_FEATURES,
    clean_anes_column, apply_feature_transformations,
    define_analysis_universe, compute_vote_shares, compute_region_composition,
    compute_religion_composition, PREFERRED_WEIGHTS, VAR_PARTYID_PRE, VAR_PARTYID_POST,
    VAR_GENDER, VAR_SEX, VAR_AGE, VAR_EDU5, VAR_RACEETH,
    compute_variance_weights, apply_variance_weighting
)
from helpers.ml_inference import clean_holdout_features, train_holdout_predictors, predict_cluster_stances
from helpers.persona_generator import generate_avatars_json, generate_story_json
from helpers.feature_selection import select_quiz_features, validate_quiz_features
# LLM validation now runs separately via run_individual_llm_validation.py

warnings.filterwarnings('ignore')


# ============================================================================
# PATHS
# ============================================================================

CSV_PATH = Path("anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv")
OUT_DIR_LEGACY = Path("anes_cluster_output")
OUT_DIR_SITE = Path("docs")
OUT_DIR_DATA = OUT_DIR_SITE / "data"
OUT_DIR_DOCS = Path("docs")

# Create directories
for d in [OUT_DIR_LEGACY, OUT_DIR_SITE / "assets", OUT_DIR_DATA, OUT_DIR_DOCS]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

K_MIN = 8
K_MAX = 20
STABILITY_SEEDS = [42, 123, 456]  # Multi-seed stability check


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def weighted_mean(series: pd.Series, weight: pd.Series | None) -> float:
    """Compute weighted mean."""
    if series.empty or series.notna().sum() == 0:
        return np.nan
    if weight is None:
        return float(series.mean())
    valid = series.notna() & weight.notna()
    if not valid.any():
        return np.nan
    return float(np.average(series[valid], weights=weight[valid]))


def weighted_quantile(series: pd.Series, weight: pd.Series | None, q: float) -> float:
    """Compute weighted quantile."""
    if series.empty or series.notna().sum() == 0:
        return np.nan
    if weight is None:
        return float(series.quantile(q))
    valid = series.notna() & weight.notna()
    if not valid.any():
        return np.nan
    x = series[valid].values
    w = weight[valid].values
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return float(np.interp(q, cdf, x))


def compute_party_id(df: pd.DataFrame, mask: pd.Series, weight: pd.Series | None) -> tuple:
    """Compute party ID mean and p10/p90 (post preferred, else pre)."""
    pid = None
    for var in [VAR_PARTYID_POST, VAR_PARTYID_PRE]:
        if var in df.columns and df[var].notna().any():
            pid = df.loc[mask, var]
            break
    if pid is None or pid.empty:
        return np.nan, np.nan, np.nan

    w_mask = weight[mask] if weight is not None else None
    return (
        weighted_mean(pid, w_mask),
        weighted_quantile(pid, w_mask, 0.10),
        weighted_quantile(pid, w_mask, 0.90)
    )


def compute_demographics(df: pd.DataFrame, mask: pd.Series, weight: pd.Series | None) -> dict:
    """Compute demographic shares."""
    demo = {}

    # Gender
    if VAR_GENDER in df.columns:
        for code, label in [(1, "Man"), (2, "Woman"), (3, "Nonbinary"), (4, "Other")]:
            condition = (df.loc[mask, VAR_GENDER] == code).astype(float)
            demo[f"gender_{label}"] = weighted_mean(condition, weight[mask] if weight is not None else None)
    elif VAR_SEX in df.columns:
        for code, label in [(1, "Man"), (2, "Woman")]:
            condition = (df.loc[mask, VAR_SEX] == code).astype(float)
            demo[f"gender_{label}"] = weighted_mean(condition, weight[mask] if weight is not None else None)

    # Race/ethnicity
    if VAR_RACEETH in df.columns:
        race_labels = {
            1: "White NH", 2: "Black NH", 3: "Hispanic",
            4: "Asian/NHPI NH", 5: "Native/Other NH", 6: "Multiracial NH"
        }
        for code, label in race_labels.items():
            condition = (df.loc[mask, VAR_RACEETH] == code).astype(float)
            demo[f"race_{label}"] = weighted_mean(condition, weight[mask] if weight is not None else None)

    # Education
    if VAR_EDU5 in df.columns:
        college = (df.loc[mask, VAR_EDU5].isin([3, 4, 5])).astype(float)
        demo["education_college"] = weighted_mean(college, weight[mask] if weight is not None else None)

    # Age
    if VAR_AGE in df.columns:
        demo["age_mean"] = weighted_mean(df.loc[mask, VAR_AGE], weight[mask] if weight is not None else None)

    return demo


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(universe: str, use_llm: bool):
    print(f"\n{'='*70}")
    print(f"ANES 2024 Clustering Pipeline")
    print(f"{'='*70}")
    print(f"ANALYSIS_UNIVERSE: {universe}")
    print(f"Use LLM: {use_llm}")
    print(f"{'='*70}\n")

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("[1/12] Loading ANES data...")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"ANES data not found: {CSV_PATH}")

    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  Loaded {len(df_raw):,} respondents")

    # Clean all columns
    for col in df_raw.columns:
        df_raw[col] = clean_anes_column(df_raw[col])

    # ========================================================================
    # 2. SELECT WEIGHT
    # ========================================================================
    print("[2/12] Selecting weight variable...")
    weight_var = None
    for w in PREFERRED_WEIGHTS:
        if w in df_raw.columns and df_raw[w].notna().any():
            weight_var = w
            break
    if weight_var is None:
        print("  WARNING: No weight found, using unweighted analysis")
        weight_series = None
    else:
        print(f"  Using weight: {weight_var}")
        weight_series = df_raw[weight_var]

    # ========================================================================
    # 3. DEFINE ANALYSIS_UNIVERSE
    # ========================================================================
    print(f"[3/12] Defining ANALYSIS_UNIVERSE ({universe})...")
    universe_mask = define_analysis_universe(df_raw, universe)
    df = df_raw[universe_mask].copy()
    if weight_series is not None:
        weight_series = weight_series[universe_mask]
    print(f"  {len(df):,} respondents in universe ({100*len(df)/len(df_raw):.1f}% of full sample)")

    # ========================================================================
    # 4. PREPARE CLUSTERING FEATURES
    # ========================================================================
    print("[4/12] Preparing clustering features (~50 pre-core policy variables)...")
    X = apply_feature_transformations(df, CLUSTERING_FEATURES)

    # Require >=60% of features answered
    min_features = int(np.ceil(0.60 * len(CLUSTERING_FEATURES)))
    complete_mask = (X.notna().sum(axis=1) >= min_features)
    X = X[complete_mask]
    df = df[complete_mask]
    if weight_series is not None:
        weight_series = weight_series[complete_mask]

    print(f"  {len(X):,} respondents with >={min_features} features answered")
    print(f"  Features: {list(X.columns)}")

    # ========================================================================
    # 5. IMPUTE & SCALE
    # ========================================================================
    print("[5/12] Imputing missing values and scaling...")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # ========================================================================
    # 5a. COMPUTE VARIANCE WEIGHTS
    # ========================================================================
    print("[5a/12] Computing variance weights (exclude variables with >25% missing)...")
    variance_weights = compute_variance_weights(df, CLUSTERING_FEATURES, min_valid_pct=0.75)
    print(f"  Using {len(variance_weights)} variables after excluding high-missingness vars")

    # ========================================================================
    # 5b. APPLY VARIANCE WEIGHTING
    # ========================================================================
    print("[5b/12] Applying variance weighting to scaled features...")
    X_scaled = apply_variance_weighting(X_scaled, list(X.columns), variance_weights)
    print("  Features with higher variance now have higher weight in distance calculations")

    # Save preprocessing params
    preprocess_params = {
        "features": list(X.columns),
        "feature_specs": {var: {k: v for k, v in spec.items() if k != "desc"}
                          for var, spec in CLUSTERING_FEATURES.items() if var in X.columns},
        "imputation_medians": {var: float(imputer.statistics_[i])
                               for i, var in enumerate(X.columns)},
        "scaling_means": {var: float(scaler.mean_[i])
                          for i, var in enumerate(X.columns)},
        "scaling_stds": {var: float(scaler.scale_[i])
                         for i, var in enumerate(X.columns)},
        "variance_weights": {var: float(variance_weights[var])
                            for var in X.columns if var in variance_weights},
        "min_features_required": min_features
    }

    # ========================================================================
    # 6. SELECT K (silhouette + distance penalty from target K=15)
    # ========================================================================
    TARGET_K = 15
    MIN_CLUSTER_SIZE = 50  # Adjusted for smaller likely_voters universe (was 100)
    DISTANCE_PENALTY_WEIGHT = 0.05  # Penalize each step away from target K

    print(f"[6/12] Selecting K in [{K_MIN}, {K_MAX}] by penalized silhouette score...")
    print(f"         Target K={TARGET_K}, minimum cluster size = {MIN_CLUSTER_SIZE}")
    print(f"         Scoring: silhouette - {DISTANCE_PENALTY_WEIGHT} * |K - {TARGET_K}|")

    best_K, best_score, best_sil = None, -999, -1

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = km.fit_predict(X_scaled)

        # Check minimum cluster size
        cluster_sizes = pd.Series(labels).value_counts()
        min_size = cluster_sizes.min()

        if min_size < MIN_CLUSTER_SIZE:
            print(f"  K={k:2d}: skipped (min cluster size {min_size} < {MIN_CLUSTER_SIZE})")
            continue

        sil = silhouette_score(X_scaled, labels)
        distance_penalty = DISTANCE_PENALTY_WEIGHT * abs(k - TARGET_K)
        penalized_score = sil - distance_penalty

        print(f"  K={k:2d}: silhouette={sil:.4f}, penalty={distance_penalty:.4f}, score={penalized_score:.4f} (min size: {min_size})")

        if penalized_score > best_score:
            best_K, best_score, best_sil = k, penalized_score, sil

    K = best_K
    print(f"\n  Selected K={K} (silhouette={best_sil:.4f}, penalized_score={best_score:.4f})")

    # ========================================================================
    # 7. FIT FINAL CLUSTERING + STABILITY CHECK
    # ========================================================================
    print(f"[7/12] Fitting K-means with K={K} and stability check...")
    final_km = KMeans(n_clusters=K, n_init=50, random_state=42)
    labels = final_km.fit_predict(X_scaled)

    # Verify minimum cluster size
    final_cluster_sizes = pd.Series(labels).value_counts()
    print(f"\n  Cluster sizes (unweighted): min={final_cluster_sizes.min()}, max={final_cluster_sizes.max()}, mean={final_cluster_sizes.mean():.0f}")
    if final_cluster_sizes.min() < MIN_CLUSTER_SIZE:
        print(f"  WARNING: Some clusters have fewer than {MIN_CLUSTER_SIZE} people!")

    # Stability: run with different seeds and measure overlap
    stability_scores = []
    for seed in STABILITY_SEEDS[1:]:  # Skip first (already used)
        km_alt = KMeans(n_clusters=K, n_init=30, random_state=seed)
        labels_alt = km_alt.fit_predict(X_scaled)
        # Compute adjusted rand index (measure of agreement)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(labels, labels_alt)
        stability_scores.append(ari)

    stability_mean = float(np.mean(stability_scores))
    print(f"  Stability (mean ARI across {len(STABILITY_SEEDS)-1} seeds): {stability_mean:.3f}")

    centroids = final_km.cluster_centers_

    # ========================================================================
    # 7b. SELECT QUIZ FEATURES (top 10 by Random Forest importance)
    # ========================================================================
    print("[7b/12] Selecting quiz features by Random Forest importance...")
    quiz_features, all_feature_importance = select_quiz_features(
        X_scaled, labels, list(X.columns), n_features=10, random_state=42
    )
    quiz_validation = validate_quiz_features(
        X_scaled, labels, list(X.columns), quiz_features, random_state=42
    )

    # ========================================================================
    # 8. COMPUTE 2D EMBEDDING (PCA)
    # ========================================================================
    print("[8/12] Computing 2D PCA embedding...")
    pca = PCA(n_components=2, random_state=42)
    cluster_centroids_2d = pca.fit_transform(centroids)

    embedding_2d = {
        "method": "PCA",
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "clusters": {
            int(i): {
                "x": float(cluster_centroids_2d[i, 0]),
                "y": float(cluster_centroids_2d[i, 1])
            }
            for i in range(K)
        }
    }

    # ========================================================================
    # 9. BUILD CLUSTER PROFILES
    # ========================================================================
    print("[9/12] Building cluster profiles...")
    cluster_profiles = []

    for k in range(K):
        mask_k = (labels == k)
        n_unweighted = int(mask_k.sum())

        # Population share
        if weight_series is not None:
            pop_share = float(weight_series[mask_k].sum() / weight_series.sum())
        else:
            pop_share = float(mask_k.mean())

        # Party ID
        partyid_mean, partyid_p10, partyid_p90 = compute_party_id(df, mask_k, weight_series)

        # Vote shares (Harris, Trump, Other)
        vote_shares = compute_vote_shares(df, mask_k, weight_series)

        # Demographics
        demo = compute_demographics(df, mask_k, weight_series)

        # Region
        region = compute_region_composition(df, mask_k, weight_series)

        # Religion
        religion = compute_religion_composition(df, mask_k, weight_series)

        # Centroid in scaled space (z-scores) - for distance calculations
        centroid_scaled_dict = {var: float(centroids[k, i])
                                for i, var in enumerate(X.columns)}

        # Compute cluster means in ORIGINAL scale (for persona generation)
        # This is the actual average response value (1-7 etc), not z-scores
        original_means_dict = {}
        for var in X.columns:
            cluster_vals = X_imp_df.loc[mask_k, var]
            if cluster_vals.notna().any():
                # Weight if available
                if weight_series is not None:
                    cluster_mean = weighted_mean(cluster_vals, weight_series[mask_k])
                else:
                    cluster_mean = float(cluster_vals.mean())
                original_means_dict[var] = round(cluster_mean, 3)
            else:
                original_means_dict[var] = None

        # Cluster profile
        profile = {
            "cluster": int(k),
            "n_unweighted": n_unweighted,
            "pop_share": round(pop_share, 4),
            "partyid_mean": round(partyid_mean, 2) if not np.isnan(partyid_mean) else None,
            "partyid_p10": round(partyid_p10, 2) if not np.isnan(partyid_p10) else None,
            "partyid_p90": round(partyid_p90, 2) if not np.isnan(partyid_p90) else None,
            "vote_harris": round(vote_shares["harris"], 3) if not np.isnan(vote_shares["harris"]) else None,
            "vote_trump": round(vote_shares["trump"], 3) if not np.isnan(vote_shares["trump"]) else None,
            "vote_other": round(vote_shares["other"], 3) if not np.isnan(vote_shares["other"]) else None,
            "demographics": {k: round(v, 3) if not np.isnan(v) else None for k, v in demo.items()},
            "region": {k: round(v, 3) if not np.isnan(v) else None for k, v in region.items()},
            "religion": {k: round(v, 2) if not np.isnan(v) else None for k, v in religion.items()},
            **{f"centroid_{var}": val for var, val in centroid_scaled_dict.items()},  # z-scores
            **{f"mean_original_{var}": val for var, val in original_means_dict.items()}  # original scale
        }
        cluster_profiles.append(profile)

    # ========================================================================
    # REORDER CLUSTERS BY HARRIS VOTE (descending)
    # ========================================================================
    print("\n[9b/12] Reordering clusters by Harris vote share (descending)...")
    # Sort by Harris vote (highest first)
    cluster_profiles_sorted = sorted(cluster_profiles, key=lambda c: c.get("vote_harris", 0), reverse=True)

    # Create mapping: old cluster ID -> new cluster ID
    old_to_new = {}
    for new_id, profile in enumerate(cluster_profiles_sorted):
        old_id = profile["cluster"]
        old_to_new[old_id] = new_id
        profile["cluster"] = new_id  # Update cluster ID
        print(f"  Cluster {old_id} -> Cluster {new_id} (Harris: {profile['vote_harris']*100:.1f}%)")

    cluster_profiles = cluster_profiles_sorted

    # Remap labels array
    labels_remapped = np.array([old_to_new[label] for label in labels])
    labels = labels_remapped

    # Remap centroids array (reorder rows)
    centroids_remapped = np.zeros_like(centroids)
    for old_id, new_id in old_to_new.items():
        centroids_remapped[new_id] = centroids[old_id]
    centroids = centroids_remapped

    # Remap embedding_2d
    embedding_2d_remapped = {}
    for old_id, new_id in old_to_new.items():
        embedding_2d_remapped[new_id] = embedding_2d["clusters"][old_id]
    embedding_2d["clusters"] = embedding_2d_remapped

    # ========================================================================
    # 10. ML HOLDOUT PREDICTIONS
    # ========================================================================
    print("[10/12] Training ML models for holdout topics...")
    Y_holdout = clean_holdout_features(df)
    ml_models = train_holdout_predictors(X_scaled, Y_holdout, list(X.columns))
    ml_predictions = predict_cluster_stances(ml_models, centroids)
    print(f"  Trained {len(ml_models)} models: {list(ml_predictions.keys())}")

    # ========================================================================
    # 11. GENERATE PERSONAS & STORY
    # ========================================================================
    print("[11/12] Generating personas and story...")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    avatars = generate_avatars_json(cluster_profiles, ml_predictions, use_llm, api_key)
    metadata = {
        "analysis_universe": universe,
        "K": K,
        "silhouette": round(best_sil, 4),
        "stability_ari": round(stability_mean, 3),
        "weight_variable": weight_var,
        "n_respondents": len(df),
        "timestamp": datetime.now().isoformat(),
        "features": list(X.columns)
    }
    story = generate_story_json(cluster_profiles, metadata, use_llm, api_key)

    # ========================================================================
    # 11b. LLM VALIDATION EXPERIMENT (optional)
    # ========================================================================
    # LLM validation is now run separately: python run_individual_llm_validation.py
    print("[11b/12] LLM validation runs separately (see run_individual_llm_validation.py)")
    llm_validation_results = {"skipped": True, "reason": "Run separately via run_individual_llm_validation.py"}

    # ========================================================================
    # 12. SAVE ALL OUTPUTS
    # ========================================================================
    print("[12/12] Saving outputs...")

    # Metadata
    with open(OUT_DIR_DATA / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Cluster profiles
    with open(OUT_DIR_DATA / "cluster_profiles.json", "w") as f:
        json.dump({"clusters": cluster_profiles}, f, indent=2)

    # Preprocessing params
    with open(OUT_DIR_DATA / "preprocess.json", "w") as f:
        json.dump(preprocess_params, f, indent=2)

    # Centroids (scaled space)
    centroids_dict = {
        int(i): {var: float(centroids[i, j]) for j, var in enumerate(X.columns)}
        for i in range(K)
    }
    with open(OUT_DIR_DATA / "centroids.json", "w") as f:
        json.dump({"centroids": centroids_dict}, f, indent=2)

    # Cluster distances (pairwise Euclidean in scaled space)
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(centroids, metric='euclidean'))
    cluster_distances = {
        int(i): {int(j): round(float(dist_matrix[i, j]), 3) for j in range(K)}
        for i in range(K)
    }
    with open(OUT_DIR_DATA / "cluster_distances.json", "w") as f:
        json.dump({"distances": cluster_distances}, f, indent=2)

    # 2D embedding
    with open(OUT_DIR_DATA / "embedding_2d.json", "w") as f:
        json.dump(embedding_2d, f, indent=2)

    # ML predictions
    with open(OUT_DIR_DATA / "ml_holdout_predictions.json", "w") as f:
        json.dump(ml_predictions, f, indent=2)

    # Avatars
    with open(OUT_DIR_DATA / "avatars.json", "w") as f:
        json.dump(avatars, f, indent=2)

    # Story
    with open(OUT_DIR_DATA / "story.json", "w") as f:
        json.dump(story, f, indent=2)

    # Quiz features (top 10 by RF importance)
    quiz_feature_data = {
        "quiz_features": quiz_features,
        "all_feature_importance": {k: round(v, 6) for k, v in all_feature_importance.items()},
        "validation": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in quiz_validation.items()}
    }
    with open(OUT_DIR_DATA / "quiz_features.json", "w") as f:
        json.dump(quiz_feature_data, f, indent=2)

    # LLM validation results
    if llm_validation_results:
        with open(OUT_DIR_DATA / "llm_validation.json", "w") as f:
            json.dump(llm_validation_results, f, indent=2)

    # Welcome tour (static template)
    welcome_tour = {
        "tours": [
            {
                "id": "main_tour",
                "title": "Welcome to the ANES 2024 Ideological Clusters",
                "steps": [
                    {"target": "#explore-tab", "content": "Explore clusters in 3D and 2D visualizations", "placement": "bottom"},
                    {"target": "#quiz-tab", "content": "Take a 10-question quiz to find your cluster", "placement": "bottom"},
                    {"target": "#story-tab", "content": "Read a guided narrative about the findings", "placement": "bottom"},
                    {"target": "#persona-tab", "content": "Chat with simulated personas representing each cluster", "placement": "bottom"},
                    {"target": "#explore-content", "content": "Hover over points to see detailed cluster information", "placement": "top"}
                ]
            }
        ]
    }
    with open(OUT_DIR_DATA / "welcome_tour.json", "w") as f:
        json.dump(welcome_tour, f, indent=2)

    # ========================================================================
    # LEGACY OUTPUTS (backward compatibility)
    # ========================================================================
    print("  Saving legacy outputs...")

    # clusters_printable_summary.csv
    legacy_rows = []
    for p in cluster_profiles:
        row = {
            "cluster": p["cluster"],
            "n_unweighted": p["n_unweighted"],
            "pop_share_weighted_pct": round(p["pop_share"] * 100, 1),
            "partyid_mean_weighted": p["partyid_mean"],
            "vote_harris_pct_w": round(p["vote_harris"] * 100, 1) if p["vote_harris"] else None,
            "vote_trump_pct_w": round(p["vote_trump"] * 100, 1) if p["vote_trump"] else None,
            "vote_other_pct_w": round(p["vote_other"] * 100, 1) if p["vote_other"] else None,
        }
        legacy_rows.append(row)
    pd.DataFrame(legacy_rows).to_csv(OUT_DIR_LEGACY / "clusters_printable_summary.csv", index=False)

    # domain_means_by_cluster.csv (using centroid values)
    domain_rows = []
    for p in cluster_profiles:
        row = {"cluster": p["cluster"]}
        for var in X.columns:
            row[var] = p.get(f"centroid_{var}")
        domain_rows.append(row)
    pd.DataFrame(domain_rows).to_csv(OUT_DIR_LEGACY / "domain_means_by_cluster.csv", index=False)

    # clusters_3d_cube_select.html (using Plotly)
    print("  Generating 3D cube HTML...")
    generate_3d_cube_html(cluster_profiles, X.columns)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - {OUT_DIR_DATA.resolve()}")
    print(f"  - {OUT_DIR_LEGACY.resolve()}")
    print(f"\nNext steps:")
    print(f"  1. Review {OUT_DIR_DATA}/metadata.json")
    print(f"  2. Open {OUT_DIR_LEGACY}/clusters_3d_cube_select.html in browser")
    print(f"  3. Deploy docs/ folder to GitHub Pages")


# ============================================================================
# 3D CUBE HTML GENERATOR (backward compat with cube.txt)
# ============================================================================

def generate_3d_cube_html(cluster_profiles, feature_names):
    """Generate interactive 3D cube HTML with axis selectors."""
    df = pd.DataFrame(cluster_profiles)

    # Extract centroid values and normalize to [-1, 1] for better visualization
    arrays = {}
    for var in feature_names:
        col_name = f"centroid_{var}"
        if col_name in df.columns:
            values = df[col_name].to_numpy()
            # Normalize to [-1, 1] range so we can see left vs right positions clearly
            # This makes the axes comparable and easier to interpret
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                # Scale to [-1, 1]: normalized = 2 * (value - min) / (max - min) - 1
                values_normalized = 2 * (values - min_val) / (max_val - min_val) - 1
            else:
                # All values are the same, set to 0
                values_normalized = np.zeros_like(values)
            arrays[var] = values_normalized

    if len(arrays) < 3:
        print("  WARNING: Not enough features for 3D cube")
        return

    # Default axes
    dims = list(arrays.keys())
    x0, y0, z0 = dims[0], dims[1], dims[2]

    # Marker sizes (proportional to pop_share)
    sizes = df["pop_share"].fillna(0).to_numpy()
    marker_sizes = 6 + (sizes * 100) * 0.3

    # Color by party ID
    color_vals = df["partyid_mean"].values if "partyid_mean" in df.columns else None

    # Hover text
    hover_texts = []
    for _, row in df.iterrows():
        parts = [
            f"Cluster {int(row['cluster'])}",
            f"Pop share: {row['pop_share']*100:.1f}% (n={row['n_unweighted']})",
            f"Party ID: {row['partyid_mean']:.2f}" if row['partyid_mean'] else "Party ID: N/A",
            f"Harris: {row['vote_harris']*100:.1f}% | Trump: {row['vote_trump']*100:.1f}%" if row['vote_harris'] else "Vote: N/A"
        ]
        hover_texts.append("<br>".join(parts))

    # Create figure
    fig = go.Figure(
        data=[go.Scatter3d(
            x=arrays[x0], y=arrays[y0], z=arrays[z0],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=color_vals if color_vals is not None else "blue",
                showscale=True if color_vals is not None else False,
                colorbar=dict(title="Party ID") if color_vals is not None else None
            ),
            text=hover_texts,
            hovertemplate="%{text}",
            name="Clusters"
        )],
        layout=go.Layout(
            title=f"3D Cluster Map: {x0} × {y0} × {z0}",
            scene=dict(
                xaxis=dict(title=x0, zeroline=True, range=[-1.1, 1.1]),
                yaxis=dict(title=y0, zeroline=True, range=[-1.1, 1.1]),
                zaxis=dict(title=z0, zeroline=True, range=[-1.1, 1.1]),
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
    )

    # Generate HTML with selectors
    html_core = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="cube_plot")

    payload = {
        "arrays": {d: arrays[d].tolist() for d in arrays.keys()},
        "dims": dims,
        "start": {"x": x0, "y": y0, "z": z0}
    }
    payload_json = json.dumps(payload)

    controls_html = """
    <div id="cube_ctrl" style="position:absolute;top:10px;left:10px;background:white;border:1px solid #ccc;padding:10px;border-radius:5px;z-index:10;">
      <label>X: <select id="dimX"></select></label>
      <label>Y: <select id="dimY"></select></label>
      <label>Z: <select id="dimZ"></select></label>
    </div>
    """

    script = f"""
    <script>
    window.addEventListener('load', function() {{
      const gd = document.getElementById('cube_plot');
      const payload = {payload_json};
      const wrap = document.createElement('div');
      wrap.style.position='relative';
      gd.parentNode.insertBefore(wrap, gd);
      wrap.appendChild(gd);

      const ctrl = document.createElement('div');
      ctrl.innerHTML = `{controls_html}`;
      wrap.appendChild(ctrl);

      function fillSelect(id, defVal) {{
        const s = document.getElementById(id);
        payload.dims.forEach(d => {{
          const opt = document.createElement('option');
          opt.value = d;
          opt.text = d.replace(/V\\d+/, '').replace(/_/g, ' ');
          s.appendChild(opt);
        }});
        s.value = defVal;
      }}
      fillSelect('dimX', payload.start.x);
      fillSelect('dimY', payload.start.y);
      fillSelect('dimZ', payload.start.z);

      function apply() {{
        const dx = document.getElementById('dimX').value;
        const dy = document.getElementById('dimY').value;
        const dz = document.getElementById('dimZ').value;
        gd.data[0].x = payload.arrays[dx];
        gd.data[0].y = payload.arrays[dy];
        gd.data[0].z = payload.arrays[dz];
        gd.layout.scene.xaxis.title.text = dx;
        gd.layout.scene.yaxis.title.text = dy;
        gd.layout.scene.zaxis.title.text = dz;
        gd.layout.title.text = '3D Cluster Map: ' + dx + ' × ' + dy + ' × ' + dz;
        Plotly.react(gd, gd.data, gd.layout);
      }}

      document.getElementById('dimX').addEventListener('change', apply);
      document.getElementById('dimY').addEventListener('change', apply);
      document.getElementById('dimZ').addEventListener('change', apply);
    }});
    </script>
    """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>ANES 2024 Clusters - 3D Cube</title>
    </head>
    <body>
        {html_core}
        {script}
    </body>
    </html>
    """

    (OUT_DIR_LEGACY / "clusters_3d_cube_select.html").write_text(full_html, encoding="utf-8")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANES 2024 Clustering Pipeline")
    parser.add_argument("--universe", type=str, default="likely_voters",
                        choices=["likely_voters", "actual_voters", "all_respondents"],
                        help="Analysis universe (default: likely_voters)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LangChain for persona/story generation (requires API key)")

    args = parser.parse_args()

    main(args.universe, args.use_llm)

"""
ANES 2024 Preprocessing utilities for clustering pipeline - VERSION 2
Handles ~50 policy preference variables with variance weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================================
# EXPANDED CLUSTERING FEATURES (~50 PRE-CORE POLICY VARIABLES)
# ============================================================================
# All V241* (pre-election). Exclude presidential approval questions.
# Focus on policy preferences, not evaluations of president.

CLUSTERING_FEATURES_V2 = {
    # ========== IDEOLOGY ==========
    "V241228": {
        "desc": "Party identity importance (1=extremely, 4=not at all)",
        "flip": False,
        "domain": "ideology"
    },
    "V241234": {
        "desc": "How often can people be trusted (1=always, 5=never)",
        "flip": False,
        "domain": "ideology"
    },

    # ========== TRUST IN GOVERNMENT ==========
    "V241229": {
        "desc": "Trust government in Washington (1=always, 5=never)",
        "flip": False,
        "domain": "trust_gov"
    },
    "V241230": {
        "desc": "Trust court system (1=always, 5=never)",
        "flip": False,
        "domain": "trust_gov"
    },
    "V241231": {
        "desc": "Gov run by few big interests or benefit of all (1=few interests, 2=benefit all)",
        "flip": False,
        "domain": "trust_gov"
    },

    # ========== ABORTION & GENDER ==========
    "V241248": {
        "desc": "Abortion 7pt (1=always permit, 7=never permit)",
        "flip": False,
        "domain": "abortion_gender"
    },
    "V241290x": {
        "desc": "Approve/disapprove DEI (diversity, equity, inclusion)",
        "flip": False,  # Will need to check codebook for scale direction
        "domain": "abortion_gender"
    },
    "V241372x": {
        "desc": "Approve/disapprove transgender bathroom use matching identity",
        "flip": False,
        "domain": "abortion_gender"
    },
    "V241375x": {
        "desc": "Favor/oppose banning transgender girls from K-12 girls sports",
        "flip": False,
        "domain": "abortion_gender"
    },
    "V241378x": {
        "desc": "Favor/oppose laws protecting gays/lesbians from job discrimination",
        "flip": False,
        "domain": "abortion_gender"
    },
    "V241381x": {
        "desc": "Should gay/lesbian couples be allowed to adopt children",
        "flip": False,
        "domain": "abortion_gender"
    },
    "V241385x": {
        "desc": "Right of gay/lesbian couples to legally marry",
        "flip": False,
        "domain": "abortion_gender"
    },

    # ========== FISCAL ==========
    "V241232": {
        "desc": "Does government waste much tax money (1=waste lot, 4=don't waste much)",
        "flip": False,  # Already coded correctly: 1=waste lot (conservative)
        "domain": "fiscal"
    },
    "V241239": {
        "desc": "Gov services/spending 7pt (1=fewer services, 7=more services)",
        "flip": False,
        "domain": "fiscal"
    },
    "V241242": {
        "desc": "Defense spending 7pt (1=decrease, 7=increase)",
        "flip": False,
        "domain": "fiscal"
    },
    "V241245": {
        "desc": "Health insurance 7pt (1=gov plan, 7=private)",
        "flip": False,
        "domain": "fiscal"
    },

    # ========== IMMIGRATION ==========
    "V241269x": {
        "desc": "Federal budget spending: tightening border security",
        "flip": False,
        "domain": "immigration"
    },
    "V241389x": {
        "desc": "Favor/oppose ending birthright citizenship",
        "flip": False,
        "domain": "immigration"
    },
    "V241386": {
        "desc": "Policy toward unauthorized immigrants (1=felony/deport, 5=no penalty)",
        "flip": False,
        "domain": "immigration"
    },
    "V241392x": {
        "desc": "Children brought illegally: send back or allow to stay",
        "flip": False,
        "domain": "immigration"
    },
    "V241395x": {
        "desc": "Favor/oppose building wall on border with Mexico",
        "flip": False,
        "domain": "immigration"
    },
    "V241396": {
        "desc": "How important to speak English in US (1=extremely, 5=not at all)",
        "flip": False,
        "domain": "immigration"
    },

    # ========== INTERNATIONAL ==========
    "V241312x": {
        "desc": "Country better off if we just stayed home",
        "flip": False,
        "domain": "international"
    },
    "V241313": {
        "desc": "Use force to solve international problems (1=extremely willing, 7=extremely unwilling)",
        "flip": False,
        "domain": "international"
    },
    "V241400x": {
        "desc": "Favor/oppose US giving weapons to help Ukraine fight Russia",
        "flip": False,
        "domain": "international"
    },

    # ========== ENVIRONMENT ==========
    "V241366x": {
        "desc": "Government action about rising temperatures",
        "flip": False,
        "domain": "environment"
    },
    "V241284x": {
        "desc": "Federal budget spending: protecting the environment",
        "flip": False,
        "domain": "environment"
    },
    "V241258": {
        "desc": "Environment-business tradeoff 7pt (1=protect env, 7=business priority)",
        "flip": False,
        "domain": "environment"
    },

    # ========== EDUCATION ==========
    "V241287x": {
        "desc": "Approve/disapprove how colleges and universities are run",
        "flip": False,
        "domain": "education"
    },
    "V241266x": {
        "desc": "Federal budget spending: public schools",
        "flip": False,
        "domain": "education"
    },

    # ========== POLITICAL RIGHTS ==========
    "V241319x": {
        "desc": "Favor/oppose requiring ID when voting",
        "flip": False,
        "domain": "political_rights"
    },
    "V241322x": {
        "desc": "Favor/oppose allowing felons to vote",
        "flip": False,
        "domain": "political_rights"
    },
    "V241330x": {
        "desc": "Helpful/harmful if president didn't have to worry about Congress/courts",
        "flip": False,
        "domain": "political_rights"
    },

    # ========== CRIME ==========
    "V241397": {
        "desc": "Best way to deal with urban unrest (1=solve problems, 2=use force)",
        "flip": False,
        "domain": "crime"
    },
    "V241308x": {
        "desc": "Favor/oppose death penalty",
        "flip": False,
        "domain": "crime"
    },
    "V241272x": {
        "desc": "Federal budget spending: dealing with crime",
        "flip": False,
        "domain": "crime"
    },

    # ========== OTHER ==========
    "V241335": {
        "desc": "How much trust in news media (1=great deal, 5=none)",
        "flip": False,
        "domain": "other"
    },
    "V241341": {
        "desc": "Likelihood sexual harassment would keep you from voting for candidate (1=extremely, 5=not at all)",
        "flip": False,  # Already correct: 1=likely deterred (liberal), 5=not likely (conservative)
        "domain": "other"
    },
    "V241255": {
        "desc": "Gov assistance to Blacks 7pt (1=help, 7=no special help)",
        "flip": False,
        "domain": "other"
    },

    # ========== ECONOMY ==========
    "V241363x": {
        "desc": "How much larger is income gap today",
        "flip": False,
        "domain": "economy"
    },
    "V241369x": {
        "desc": "Require employers to offer paid leave to parents",
        "flip": False,
        "domain": "economy"
    },
    "V241252": {
        "desc": "Guaranteed job/income 7pt (1=gov should, 7=people on own)",
        "flip": False,
        "domain": "economy"
    },
    "V241263x": {
        "desc": "Federal budget spending: Social Security",
        "flip": False,
        "domain": "economy"
    },
    "V241278x": {
        "desc": "Federal budget spending: highways",
        "flip": False,
        "domain": "economy"
    },
    "V241281x": {
        "desc": "Federal budget spending: aid to the poor",
        "flip": False,
        "domain": "economy"
    },

    # ========== ISRAEL ==========
    "V241403x": {
        "desc": "Favor/oppose US giving military assistance to Israel",
        "flip": False,
        "domain": "israel"
    },
    "V241406x": {
        "desc": "Favor/oppose US giving humanitarian aid to Palestinians",
        "flip": False,
        "domain": "israel"
    },
    "V241409x": {
        "desc": "Side more with Israelis or Palestinians",
        "flip": False,
        "domain": "israel"
    },
    "V241412x": {
        "desc": "Approve/disapprove of protests against war in Gaza",
        "flip": False,
        "domain": "israel"
    },
}

# Democracy removed from clustering (not in user's list)
# Vaccines removed from clustering (not in user's list)

# PARTY ID - NOT USED FOR CLUSTERING (outcome variable)
# V241177 - Self ideology - NOT used (what we want to predict)
# V241227x - Party ID - NOT used (outcome variable)


# ============================================================================
# VARIANCE WEIGHTING
# ============================================================================

def compute_variance_weights(df: pd.DataFrame, features: Dict, min_valid_pct: float = 0.75) -> Dict[str, float]:
    """
    Compute variance for each feature. Exclude variables with >25% missing.

    Returns dict: {var: variance} for valid variables only
    """
    variances = {}

    for var in features.keys():
        if var not in df.columns:
            continue

        # Check missingness (after cleaning negative codes)
        series_clean = clean_anes_column(df[var])
        valid_pct = series_clean.notna().mean()

        if valid_pct < min_valid_pct:
            print(f"  Excluding {var}: only {valid_pct*100:.1f}% valid (need {min_valid_pct*100:.0f}%)")
            continue

        # Compute variance
        var_value = series_clean.var()
        if not np.isnan(var_value) and var_value > 0:
            variances[var] = var_value

    return variances


def apply_variance_weighting(X_scaled: np.ndarray, X_columns: List[str], variance_weights: Dict[str, float]) -> np.ndarray:
    """
    Apply variance weighting to scaled features.
    Higher variance features get higher weight in distance calculations.

    Formula: X_weighted = X_scaled * sqrt(variance_original)
    """
    X_weighted = X_scaled.copy()

    for i, var in enumerate(X_columns):
        if var in variance_weights:
            weight = np.sqrt(variance_weights[var])
            X_weighted[:, i] *= weight
            print(f"  {var}: variance={variance_weights[var]:.3f}, weight={weight:.3f}")

    return X_weighted


# ============================================================================
# PROFILE-ONLY VARIABLES (not used for clustering)
# ============================================================================

# Post-election variables (for defining ANALYSIS_UNIVERSE and outcomes)
VAR_TURNOUT_POST = "V242066"      # 1=Voted, 2=Did not vote
VAR_VOTE_POST_COMPUTER = "V242067"  # 1=Harris, 2=Trump, 4=West, 5=Stein, 6=Other
VAR_VOTE_POST_PAPER = "V242068"     # 1=Didn't vote, 2=Trump, 3=Harris, 4=West, 5=Stein, 7=Other

# Pre-election likelihood of voting (for "likely_voters" universe)
VAR_LIKELY_VOTE = "V241029"         # 1=Definitely will vote, 2=Probably, 3=50-50, 4=Probably won't, 5=Definitely won't

# Demographics
VAR_REGION = "V243007"             # Census region: 1=Northeast, 2=Midwest, 3=South, 4=West
VAR_GENDER = "V241551"             # 1=Man, 2=Woman, 3=Nonbinary, 4=Other
VAR_SEX = "V241550"                # 1=Male, 2=Female (fallback)
VAR_AGE = "V241458x"               # Age in years (80 = 80+)
VAR_EDU5 = "V241465x"              # Education 5-cat: 1=<HS, 2=HS, 3=Some college, 4=BA, 5=Grad
VAR_RACEETH = "V241501x"           # Race/ethnicity

# Weights (try in order)
PREFERRED_WEIGHTS = ["V240103a", "V240102a", "V240101a", "V240108a", "V240103b", "V240108b"]

# Party ID (post preferred, else pre)
VAR_PARTYID_PRE = "V241227x"
VAR_PARTYID_POST = "V242227x"


# ============================================================================
# CLEANING & TRANSFORMATION
# ============================================================================

def clean_anes_column(series: pd.Series) -> pd.Series:
    """Convert to numeric and set ANES missing codes to NaN.

    ANES missing codes:
    - Negative values: -1 (Inapplicable), -2 (DK/RF), -8 (Don't know), -9 (Refused), etc.
    - 8 = Special codes (e.g., "Side with neither" for some questions - not part of the scale)
    - Large positive values: 99 (Haven't thought much), 998 (Don't know), 999 (Refused)

    The scale values for clustering are strictly 1-7 (with many variables using 1-4, 1-5, or 1-7).
    Option 8 is excluded because it represents "neither" or special responses that are not on the
    ideological scale.
    """
    s = pd.to_numeric(series, errors="coerce")
    # Remove negative codes
    s = s.mask(s < 0, np.nan)  # Remove -1, -2, -8, -9, etc.
    # Remove 8 and higher (includes "Side with neither", "Haven't thought much", etc.)
    s = s.mask(s >= 8, np.nan)  # Remove 8, 99, 998, 999
    return s


def flip_series(s: pd.Series) -> pd.Series:
    """Flip scale: new = max - old + min (reverses direction)."""
    if s.notna().sum() == 0:
        return s
    return s.max() - s + s.min()


def apply_feature_transformations(df: pd.DataFrame, features: Dict) -> pd.DataFrame:
    """
    Apply coding transformations (map, flip) to clustering features.
    Returns transformed feature matrix.
    """
    X = pd.DataFrame(index=df.index)

    for var, spec in features.items():
        if var not in df.columns:
            continue

        s = df[var].copy()

        # Apply map if present
        if "map" in spec:
            s = s.map(spec["map"])

        # Apply flip if needed
        if spec.get("flip", False):
            s = flip_series(s)

        X[var] = s

    return X


# ============================================================================
# ANALYSIS_UNIVERSE FILTERING
# ============================================================================

def define_analysis_universe(df: pd.DataFrame, universe: str) -> pd.Series:
    """
    Define boolean mask for ANALYSIS_UNIVERSE.

    Parameters:
    - df: full ANES dataframe
    - universe: "actual_voters", "likely_voters", or "all_respondents"

    Returns:
    - Boolean Series indicating inclusion in analysis universe
    """
    if universe == "all_respondents":
        # Include everyone (no filter)
        return pd.Series(True, index=df.index)

    elif universe == "likely_voters":
        # Pre-election likelihood of voting
        # V241029: 1=Definitely will vote, 2=Probably will vote
        # Include only 1 and 2
        if VAR_LIKELY_VOTE not in df.columns:
            print(f"  WARNING: {VAR_LIKELY_VOTE} not found, falling back to actual_voters")
            return define_analysis_universe(df, "actual_voters")

        mask = df[VAR_LIKELY_VOTE].isin([1, 2])
        return mask

    elif universe == "actual_voters":
        # Must have: (1) reported voting, (2) valid presidential vote choice

        mask_voted = pd.Series(False, index=df.index)
        mask_valid_vote = pd.Series(False, index=df.index)

        # Check turnout
        if VAR_TURNOUT_POST in df.columns:
            mask_voted = df[VAR_TURNOUT_POST] == 1  # 1 = Voted

        # Check vote choice (computer OR paper)
        if VAR_VOTE_POST_COMPUTER in df.columns:
            # Valid codes: 1=Harris, 2=Trump, 4=West, 5=Stein, 6=Other
            mask_valid_vote |= df[VAR_VOTE_POST_COMPUTER].isin([1, 2, 4, 5, 6])

        if VAR_VOTE_POST_PAPER in df.columns:
            # Valid codes: 2=Trump, 3=Harris, 4=West, 5=Stein, 7=Other
            # NOTE: 1=Didn't vote is NOT a valid vote choice
            mask_valid_vote |= df[VAR_VOTE_POST_PAPER].isin([2, 3, 4, 5, 7])

        return mask_voted & mask_valid_vote

    else:
        raise ValueError(f"Invalid universe: {universe}. Must be 'actual_voters', 'likely_voters', or 'all_respondents'.")


# ============================================================================
# COMPUTE VOTE SHARES (within ANALYSIS_UNIVERSE only)
# ============================================================================

def compute_vote_shares(df: pd.DataFrame, mask: pd.Series, weight: Optional[pd.Series]) -> Dict[str, float]:
    """
    Compute vote shares ONLY from post-election reported vote.
    Does NOT fall back to pre-election intention.

    Returns dict with keys: harris, trump, other (sums to 1.0 or contains NaN).
    """
    if weight is None:
        denom = float(mask.sum())
    else:
        denom = float(weight[mask].sum())

    if denom <= 0:
        return {"harris": np.nan, "trump": np.nan, "other": np.nan}

    num_harris = 0.0
    num_trump = 0.0
    num_other = 0.0

    def weighted_count(condition):
        if weight is None:
            return float((mask & condition).sum())
        m = mask & condition & weight.notna()
        return float(weight[m].sum()) if m.any() else 0.0

    # Computer vote
    if VAR_VOTE_POST_COMPUTER in df.columns:
        v = df[VAR_VOTE_POST_COMPUTER]
        num_harris += weighted_count(v == 1)
        num_trump += weighted_count(v == 2)
        num_other += weighted_count(v.isin([4, 5, 6]))

    # Paper vote
    if VAR_VOTE_POST_PAPER in df.columns:
        v = df[VAR_VOTE_POST_PAPER]
        num_harris += weighted_count(v == 3)
        num_trump += weighted_count(v == 2)
        num_other += weighted_count(v.isin([4, 5, 7]))

    vec = np.array([num_harris, num_trump, num_other], dtype=float)
    vec = np.clip(vec, 0.0, None)
    s = vec.sum()

    if s > 0:
        vec = vec / s

    return {"harris": vec[0], "trump": vec[1], "other": vec[2]}


# ============================================================================
# COMPUTE REGION COMPOSITION
# ============================================================================

def compute_region_composition(df: pd.DataFrame, mask: pd.Series, weight: Optional[pd.Series]) -> Dict[str, float]:
    """
    Compute regional distribution (weighted).
    V243007: 1=Northeast, 2=Midwest, 3=South, 4=West
    """
    if VAR_REGION not in df.columns:
        return {"Northeast": np.nan, "Midwest": np.nan, "South": np.nan, "West": np.nan}

    region_labels = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
    composition = {}

    for code, label in region_labels.items():
        condition = (df[VAR_REGION] == code) & mask
        if weight is None:
            composition[label] = float(condition.sum()) / float(mask.sum()) if mask.sum() > 0 else np.nan
        else:
            num = float(weight[condition & weight.notna()].sum())
            denom = float(weight[mask & weight.notna()].sum())
            composition[label] = num / denom if denom > 0 else np.nan

    return composition


# ============================================================================
# COMPUTE RELIGION (V241725 not in clustering, but useful for profiles)
# ============================================================================

def compute_religion_composition(df: pd.DataFrame, mask: pd.Series, weight: Optional[pd.Series]) -> Dict[str, float]:
    """
    Compute religious attendance distribution (weighted).
    V241725: 1=Never, 2=Few times/year, 3=Once/twice/month, 4=Almost weekly, 5=Weekly, 6=More than weekly
    """
    if "V241725" not in df.columns:
        return {"mean_attendance": np.nan}

    vals = df.loc[mask, "V241725"]

    if weight is None:
        mean_att = vals.mean()
    else:
        w_vals = weight[mask]
        valid = vals.notna() & w_vals.notna()
        mean_att = np.average(vals[valid], weights=w_vals[valid]) if valid.any() else np.nan

    return {"mean_attendance": mean_att}

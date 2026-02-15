"""
LLM Validation: Individual-Level Crime Prediction
==================================================
Test whether LLM can predict individual respondents' crime policy positions
from their other policy positions (ideology-only vs ideology+demographics).

Samples 200 respondents and tests:
1. Ideology-only model: LLM gets only non-crime policy responses
2. Full model: LLM gets policy responses + demographics (gender, education, age, race)

Metrics:
- Correct prediction % (exact match after rounding to nearest integer)
- Within ±1 point %
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# LLM imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    print("ERROR: LangChain not installed. Run: pip install langchain-openai")
    sys.exit(1)

# Add helpers to path
sys.path.insert(0, str(Path(__file__).parent))
from helpers.preprocessing_v2 import (
    CLUSTERING_FEATURES_V2,
    clean_anes_column,
    VAR_GENDER, VAR_AGE, VAR_EDU5, VAR_RACEETH
)

# Crime variables to predict (holdout)
CRIME_VARS = {
    "V241397": "Urban unrest response (1=solve problems of racism/police violence, 7=use all available force)",
    "V241308x": "Death penalty (1=favor strongly, 2=favor not strongly, 3=oppose not strongly, 4=oppose strongly)",
    "V241272x": "Federal crime spending (1=increased a lot, 2=increased a little, 3=kept the same, 4=decreased a little, 5=decreased a lot)"
}

# Paths
CSV_PATH = Path("anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv")
CODEBOOK_PATH = Path("anes_timeseries_2024_csv_20250808/anes_variable_scales.csv")
OUTPUT_JSON = Path("docs/data/llm_validation_individuals.json")


def load_codebook():
    """Load ANES codebook with full scale labels for each variable."""
    codebook = {}
    try:
        cb_df = pd.read_csv(CODEBOOK_PATH)
        for _, row in cb_df.iterrows():
            var = row['variable']
            full_scale = str(row.get('full_scale', ''))
            # Extract only positive-numbered scale options (skip negative codes)
            parts = [s.strip() for s in full_scale.split(';')]
            valid_parts = [s for s in parts if s and s[0].isdigit()]
            codebook[var] = '; '.join(valid_parts)
    except Exception as e:
        print(f"  WARNING: Could not load codebook: {e}")
    return codebook


def load_and_sample_respondents(n_sample=200, random_state=42):
    """
    Load ANES data and sample respondents with valid crime variable responses.
    """
    print(f"\n[1/5] Loading ANES data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} total respondents")

    # Clean crime variables
    for var in CRIME_VARS.keys():
        df[var] = clean_anes_column(df[var])

    # Keep only respondents with ALL crime variables answered
    valid_mask = pd.Series(True, index=df.index)
    for var in CRIME_VARS.keys():
        valid_mask &= df[var].notna()

    valid_df = df[valid_mask].copy()
    print(f"  {len(valid_df)} respondents have all 3 crime variables answered")

    # Sample n_sample respondents
    if len(valid_df) < n_sample:
        print(f"  WARNING: Only {len(valid_df)} valid respondents, sampling all")
        sample_df = valid_df
    else:
        sample_df = valid_df.sample(n=n_sample, random_state=random_state)
        print(f"  Sampled {n_sample} respondents (random_state={random_state})")

    return sample_df


def extract_respondent_features(row, include_demographics=False, codebook=None):
    """
    Extract policy features (and optionally demographics) for a respondent.
    """
    features = {}

    # Policy features (exclude crime variables)
    for var, spec in CLUSTERING_FEATURES_V2.items():
        if var in CRIME_VARS:
            continue  # Skip crime variables (holdout)

        val = clean_anes_column(pd.Series([row.get(var, np.nan)])).iloc[0]
        if pd.notna(val):
            scale = codebook.get(var, "") if codebook else ""
            features[var] = {
                "value": float(val),
                "desc": spec.get("desc", var),
                "scale": scale
            }

    # Demographics (optional)
    if include_demographics:
        # Gender
        if VAR_GENDER in row and pd.notna(row[VAR_GENDER]):
            gender_code = int(row[VAR_GENDER])
            gender_map = {1: "Man", 2: "Woman", 3: "Nonbinary", 4: "Other"}
            features["_gender"] = {"value": gender_map.get(gender_code, "Unknown"), "desc": "Gender"}

        # Age
        if VAR_AGE in row and pd.notna(row[VAR_AGE]):
            features["_age"] = {"value": int(row[VAR_AGE]), "desc": "Age in years"}

        # Education
        if VAR_EDU5 in row and pd.notna(row[VAR_EDU5]):
            edu_code = int(row[VAR_EDU5])
            edu_map = {1: "Less than HS", 2: "HS grad", 3: "Some college", 4: "Bachelor's", 5: "Graduate degree"}
            features["_education"] = {"value": edu_map.get(edu_code, "Unknown"), "desc": "Education"}

        # Race/ethnicity
        if VAR_RACEETH in row and pd.notna(row[VAR_RACEETH]):
            race_code = int(row[VAR_RACEETH])
            race_map = {1: "White NH", 2: "Black NH", 3: "Hispanic", 4: "Asian/NHPI NH",
                       5: "Native/Other NH", 6: "Multiracial NH"}
            features["_race"] = {"value": race_map.get(race_code, "Unknown"), "desc": "Race/ethnicity"}

    return features


def build_llm_prompt(features: Dict, respondent_id: int):
    """
    Build prompt for LLM to predict crime policy positions.
    Includes full scale labels from ANES codebook for each policy position.
    """
    prompt = f"You are analyzing respondent #{respondent_id} from a political survey. "
    prompt += "Based on their responses below, predict their positions on 3 crime-related questions.\n\n"

    # Demographics (if included)
    demo_features = {k: v for k, v in features.items() if k.startswith("_")}
    if demo_features:
        prompt += "DEMOGRAPHICS:\n"
        for key, info in demo_features.items():
            prompt += f"  - {info['desc']}: {info['value']}\n"
        prompt += "\n"

    # Policy positions (non-crime) with scale labels
    policy_features = {k: v for k, v in features.items() if not k.startswith("_")}
    prompt += "POLICY POSITIONS:\n\n"

    for var, info in sorted(policy_features.items()):
        scale = info.get("scale", "")
        prompt += f"  - {info['desc']}\n"
        if scale:
            prompt += f"    Scale: {scale}\n"
        prompt += f"    Response: {info['value']:.0f}\n"

    prompt += "\n\nBased on the above, predict this person's responses to these 3 crime questions:\n\n"
    prompt += "1. Urban unrest: Best way to deal with urban unrest and rioting?\n"
    prompt += "   Scale: 1 Solve problems of racism and police violence; 2-6 Intermediate; 7 Use all available force to maintain law and order\n\n"
    prompt += "2. Death penalty: Do you favor or oppose the death penalty for persons convicted of murder?\n"
    prompt += "   Scale: 1 Favor strongly; 2 Favor not strongly; 3 Oppose not strongly; 4 Oppose strongly\n\n"
    prompt += "3. Crime spending: Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
    prompt += "   Scale: 1 Increased a lot; 2 Increased a little; 3 Kept the same; 4 Decreased a little; 5 Decreased a lot\n"
    prompt += "\nRespond with ONLY a JSON object: {\"urban_unrest\": X, \"death_penalty\": Y, \"crime_spending\": Z}\n"
    prompt += "where X, Y, Z are integers on the scales indicated above."

    return prompt


def query_llm(prompt: str, api_key: str) -> Dict[str, float]:
    """
    Query GPT-4o-mini to predict crime positions.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0
        )

        messages = [
            SystemMessage(content="You are a political analyst predicting survey responses. Respond ONLY with the requested JSON, no explanation."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            predictions = json.loads(json_match.group())
            return {
                "urban_unrest": float(predictions.get("urban_unrest", np.nan)),
                "death_penalty": float(predictions.get("death_penalty", np.nan)),
                "crime_spending": float(predictions.get("crime_spending", np.nan))
            }
        else:
            print(f"    WARNING: Could not parse response: {response_text[:100]}")
            return {"urban_unrest": np.nan, "death_penalty": np.nan, "crime_spending": np.nan}

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"urban_unrest": np.nan, "death_penalty": np.nan, "crime_spending": np.nan}


def evaluate_predictions(pred: Dict[str, float], truth: Dict[str, float]) -> Dict:
    """
    Evaluate prediction accuracy per question.

    Returns dict with per-question metrics:
    - For each question: correct (0/1), within_1 (0/1), error
    """
    var_map = {
        "V241397": "urban_unrest",
        "V241308x": "death_penalty",
        "V241272x": "crime_spending"
    }

    per_question = {}
    errors = []

    for anes_var, pred_key in var_map.items():
        if anes_var in truth and pred_key in pred:
            t = truth[anes_var]
            p = pred[pred_key]
            if pd.notna(t) and pd.notna(p):
                t_round = round(t)
                p_round = round(p)
                error = abs(p - t)
                errors.append(error)

                per_question[pred_key] = {
                    "correct": 1 if t_round == p_round else 0,
                    "within_1": 1 if error <= 1.0 else 0,
                    "error": float(error)
                }

    return {
        "per_question": per_question,
        "mean_error": float(np.mean(errors)) if errors else np.nan
    }


def run_validation(sample_df: pd.DataFrame, api_key: str, include_demographics: bool, model_name: str, codebook: dict = None):
    """
    Run validation experiment on sampled respondents.
    """
    results = []

    print(f"\n[Running {model_name}]")
    print(f"  Demographics included: {include_demographics}")

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(sample_df)} respondents...")

        # Extract features
        features = extract_respondent_features(row, include_demographics=include_demographics, codebook=codebook)

        # Extract ground truth
        truth = {var: row[var] for var in CRIME_VARS.keys() if pd.notna(row[var])}

        # Build prompt and query LLM
        prompt = build_llm_prompt(features, idx + 1)
        predictions = query_llm(prompt, api_key)

        # Evaluate
        metrics = evaluate_predictions(predictions, truth)

        results.append({
            "respondent_id": idx + 1,
            "predictions": predictions,
            "ground_truth": truth,
            "metrics": metrics
        })

    # Aggregate per-question metrics
    question_keys = ["urban_unrest", "death_penalty", "crime_spending"]
    per_question_agg = {}

    for qk in question_keys:
        corrects = [r["metrics"]["per_question"][qk]["correct"]
                    for r in results if qk in r["metrics"]["per_question"]]
        within_1s = [r["metrics"]["per_question"][qk]["within_1"]
                     for r in results if qk in r["metrics"]["per_question"]]
        errs = [r["metrics"]["per_question"][qk]["error"]
                for r in results if qk in r["metrics"]["per_question"]]

        n = len(corrects)
        per_question_agg[qk] = {
            "correct_pct": (sum(corrects) / n * 100) if n > 0 else 0,
            "within_1_pct": (sum(within_1s) / n * 100) if n > 0 else 0,
            "mean_error": float(np.mean(errs)) if errs else 0,
            "n": n
        }

    mean_error = np.mean([r["metrics"]["mean_error"] for r in results
                          if not np.isnan(r["metrics"]["mean_error"])])

    print(f"\n  RESULTS for {model_name}:")
    for qk in question_keys:
        q = per_question_agg[qk]
        print(f"    {qk}: Correct={q['correct_pct']:.1f}%, Within±1={q['within_1_pct']:.1f}%, MAE={q['mean_error']:.2f}")
    print(f"    Overall mean error: {mean_error:.2f}")

    return {
        "model_name": model_name,
        "include_demographics": include_demographics,
        "n_respondents": len(results),
        "per_question": per_question_agg,
        "mean_error": mean_error,
        "individual_results": results
    }


def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY not set in environment")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("="*70)
    print("LLM VALIDATION: Individual-Level Crime Prediction")
    print("="*70)

    # Load codebook and sample data
    print("\n[1.5/5] Loading ANES codebook...")
    codebook = load_codebook()
    print(f"  Loaded scales for {len(codebook)} variables")

    sample_df = load_and_sample_respondents(n_sample=200, random_state=42)

    # Run two experiments
    print("\n[2/5] Running ideology-only model...")
    ideology_results = run_validation(sample_df, api_key, include_demographics=False,
                                     model_name="Ideology Only", codebook=codebook)

    print("\n[3/5] Running full model (ideology + demographics)...")
    full_results = run_validation(sample_df, api_key, include_demographics=True,
                                  model_name="Full Model (Ideology + Demographics)", codebook=codebook)

    # Save results
    print("\n[4/5] Saving results...")
    output = {
        "experiment": "individual_crime_prediction",
        "description": "Predict individual crime policy responses from other policy positions",
        "n_respondents": len(sample_df),
        "crime_variables": CRIME_VARS,
        "models": {
            "ideology_only": {
                "per_question": ideology_results["per_question"],
                "mean_error": ideology_results["mean_error"]
            },
            "full_model": {
                "per_question": full_results["per_question"],
                "mean_error": full_results["mean_error"]
            }
        },
        "detailed_results": {
            "ideology_only": ideology_results["individual_results"],
            "full_model": full_results["individual_results"]
        }
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to: {OUTPUT_JSON}")

    # Summary
    question_labels = {
        "urban_unrest": "Urban Unrest",
        "death_penalty": "Death Penalty",
        "crime_spending": "Crime Spending"
    }

    print("\n" + "="*80)
    print("SUMMARY: Per-Question Results")
    print("="*80)

    print(f"\n{'Question':<20} {'Model':<35} {'Correct %':<12} {'Within ±1 %':<12}")
    print("-"*80)
    for qk, qlabel in question_labels.items():
        iq = ideology_results["per_question"][qk]
        fq = full_results["per_question"][qk]
        print(f"{qlabel:<20} {'Ideology Only':<35} {iq['correct_pct']:>8.1f}%    {iq['within_1_pct']:>8.1f}%")
        print(f"{'':<20} {'Ideology + Demographics':<35} {fq['correct_pct']:>8.1f}%    {fq['within_1_pct']:>8.1f}%")
        print()

    print("\n[5/5] Done! Run generate_individual_validation_report.py to create HTML report with charts.")


if __name__ == "__main__":
    main()

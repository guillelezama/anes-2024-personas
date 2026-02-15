"""
LLM Validation Experiment: Crime Prediction on Individual Respondents
=====================================================================
Holdout crime policy variables and use LLM to predict responses based on
all other policy positions for INDIVIDUAL survey respondents.

This validates whether the LLM can infer consistent policy positions from
a subset of questions.

Crime variables (from preprocessing_v2.py):
- V241397: Police funding (urban unrest - solve problems vs use force)
- V241308x: Death penalty
- V241272x: Gun background checks (actually crime spending)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Try modern LangChain imports first, fallback to older versions
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI, ChatAnthropic
        from langchain.schema import SystemMessage, HumanMessage
    except ImportError:
        ChatOpenAI = None
        ChatAnthropic = None
        SystemMessage = None
        HumanMessage = None


# Crime variables to holdout and predict
CRIME_VARIABLES = {
    "V241397": "Police funding",
    "V241308x": "Death penalty",
    "V241272x": "Crime spending"
}


def load_codebook(codebook_path: str = "anes_timeseries_2024_csv_20250808/anes_variable_scales.csv") -> Dict[str, Dict]:
    """
    Load ANES codebook with full question text and scale labels.

    Returns:
    --------
    codebook : Dict[str, Dict]
        Dictionary mapping variable names to {question, full_scale}
    """
    import pandas as pd

    codebook = {}
    try:
        df = pd.read_csv(codebook_path)
        for _, row in df.iterrows():
            codebook[row['variable']] = {
                'question': row['question'],
                'full_scale': row['full_scale']
            }
    except Exception as e:
        print(f"  WARNING: Could not load codebook from {codebook_path}: {e}")

    return codebook


def load_and_sample_respondents(
    data_file: str,
    feature_list: List[str],
    n_samples: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load ANES data and sample respondents who have answered crime questions.

    OUTPUT filtering: Only select respondents with valid scale responses (1-7) on crime variables.
    This excludes:
    - 8: Special codes like "Side with neither" (not part of ideological scale)
    - 99: Haven't thought much about this
    - -9: Refused
    - -8: Don't know
    - -1: Inapplicable
    - -2: DK/RF in previous questions

    INPUT features will be kept as-is (including 8, 99, negatives) since the LLM can handle
    voters who don't know much about certain topics.

    Parameters:
    -----------
    data_file : str
        Path to ANES CSV file
    feature_list : List[str]
        List of all clustering features
    n_samples : int
        Number of respondents to sample
    random_state : int
        Random seed

    Returns:
    --------
    sample_df : pd.DataFrame
        Sample of respondents with all features
    """
    print(f"\n  Loading ANES data from {data_file}...")
    df = pd.read_csv(data_file, low_memory=False)
    print(f"  Total respondents: {len(df)}")

    # Filter to respondents who answered all crime variables with VALID SCALE responses (1-7)
    # This ensures we have ground truth OUTPUT to validate against
    # Excludes: option 8 (neither/special), 99 (haven't thought), negatives (DK/RF/inapplicable)
    crime_vars = list(CRIME_VARIABLES.keys())
    for var in crime_vars:
        if var in df.columns:
            # Keep only valid scale responses (1-7)
            df = df[df[var].between(1, 7)]

    print(f"  Respondents with valid crime variables: {len(df)}")

    # Sample respondents
    np.random.seed(random_state)
    if len(df) > n_samples:
        sample_df = df.sample(n=n_samples, random_state=random_state)
    else:
        sample_df = df

    print(f"  Sampled {len(sample_df)} respondents for validation")

    return sample_df


def build_llm_prompt_individual(
    respondent_data: Dict[str, float],
    feature_specs: Dict[str, Dict],
    codebook: Optional[Dict[str, Dict]] = None
) -> str:
    """
    Build prompt for LLM to predict crime policy positions for an individual.

    Parameters:
    -----------
    respondent_data : Dict[str, float]
        Dictionary of variable -> value for non-crime features
    feature_specs : Dict[str, Dict]
        Feature specifications with descriptions
    codebook : Optional[Dict[str, Dict]]
        Codebook with full question text and scale labels

    Returns:
    --------
    prompt : str
        Formatted prompt for LLM
    """
    prompt = "You are analyzing a voter's policy positions. Based on their positions below, predict their stance on crime-related policies.\n\n"
    prompt += "KNOWN POLICY POSITIONS:\n\n"

    # Group by domain
    by_domain = {}
    for var, value in respondent_data.items():
        if var in feature_specs:
            spec = feature_specs[var]
            domain = spec.get("domain", "other")
            if domain not in by_domain:
                by_domain[domain] = []

            # Get full question and scale from codebook
            question = spec.get("desc", var)
            scale = None
            if codebook and var in codebook:
                q = codebook[var].get('question')
                if q and isinstance(q, str):
                    question = q.strip().replace('\n', ' ')
                s = codebook[var].get('full_scale')
                if s and isinstance(s, str):
                    scale = s

            by_domain[domain].append((var, value, question, scale))

    for domain, items in sorted(by_domain.items()):
        prompt += f"\n{domain.upper().replace('_', ' ')}:\n"
        for var, value, question, scale in items:
            # Clean up question text
            question_clean = question.replace('\n', ' ').strip()
            prompt += f"  - Q: {question_clean}\n"
            if scale:
                # Extract only valid response options (positive numbers)
                scale_parts = [s.strip() for s in scale.split(';')]
                valid_options = [s for s in scale_parts if s and s[0].isdigit() and not s.startswith('-')]
                scale_clean = '; '.join(valid_options[:7])  # Limit to first 7 to avoid clutter
                prompt += f"    Scale: {scale_clean}\n"

            # Display response with interpretation for special codes
            if value < 0:
                prompt += f"    Your response: {value:.0f} (Don't know/Refused/Not applicable)\n"
            elif value == 8:
                prompt += f"    Your response: {value:.0f} (Neither/No strong opinion on this scale)\n"
            elif value >= 99:
                prompt += f"    Your response: {value:.0f} (Haven't thought much about this)\n"
            else:
                prompt += f"    Your response: {value:.0f}\n"

    # Add crime prediction questions with full ANES text
    prompt += "\n\nBased on these positions, predict the following:\n\n"

    prompt += "1. URBAN UNREST:\n"
    prompt += "   What is the best way to deal with the problem of urban unrest and rioting? "
    prompt += "Some say it is more important to use all available force to maintain law and order, no matter what results. "
    prompt += "Others say it is more important to correct the problems of racism and police violence that give rise to the disturbances. "
    prompt += "And, of course, other people have opinions in between.\n"
    prompt += "   On this scale from 1 to 7, where 1 means solve problems of racism and police violence, "
    prompt += "and 7 means use all available force to maintain law and order, where would you place yourself on this scale?\n\n"

    prompt += "2. DEATH PENALTY:\n"
    prompt += "   Do you favor or oppose the death penalty for persons convicted of murder?\n"
    prompt += "   Scale: 1 = Favor strongly; 2 = Favor not strongly; 3 = Oppose not strongly; 4 = Oppose strongly\n\n"

    prompt += "3. CRIME SPENDING:\n"
    prompt += "   Should federal spending on dealing with crime be increased, decreased, or kept the same?\n"
    prompt += "   Scale: 1 = Increased a lot; 2 = Increased a little; 3 = Kept the same; 4 = Decreased a little; 5 = Decreased a lot\n"

    prompt += '\n\nRespond in JSON format with ONLY the numeric values: {"urban_unrest": X, "death_penalty": Y, "crime_spending": Z}'

    return prompt


def query_llm_for_individual_predictions(
    prompt: str,
    api_key: str,
    use_anthropic: bool = False
) -> Dict[str, float]:
    """
    Query LLM to predict crime policy positions for an individual.

    Returns dict with keys: urban_unrest, death_penalty, crime_spending
    """
    if ChatOpenAI is None or ChatAnthropic is None:
        print("  ERROR: LangChain not installed")
        return {
            "urban_unrest": np.nan,
            "death_penalty": np.nan,
            "crime_spending": np.nan
        }

    try:
        if use_anthropic:
            llm = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                anthropic_api_key=api_key,
                temperature=0
            )
        else:
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use mini for cost efficiency
                openai_api_key=api_key,
                temperature=0
            )

        messages = [
            SystemMessage(content="You are a political analyst expert at inferring policy positions from known stances."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Parse JSON response
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
            print(f"  WARNING: Could not parse LLM response: {response_text[:200]}")
            return {
                "urban_unrest": np.nan,
                "death_penalty": np.nan,
                "crime_spending": np.nan
            }

    except Exception as e:
        print(f"  ERROR querying LLM: {e}")
        return {
            "urban_unrest": np.nan,
            "death_penalty": np.nan,
            "crime_spending": np.nan
        }


def compute_individual_accuracy(
    predictions: Dict[str, float],
    ground_truth: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute accuracy metrics for individual predictions.

    Returns dict with:
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - r_squared: R-squared (coefficient of determination)
    - within_1: Fraction of predictions within 1 point
    """
    # Map variable names to prediction keys
    VAR_TO_KEY = {
        "V241397": "urban_unrest",
        "V241308x": "death_penalty",
        "V241272x": "crime_spending"
    }

    errors = []
    within_1 = []
    pred_values = []
    gt_values = []

    for var, key in VAR_TO_KEY.items():
        if var in ground_truth and key in predictions:
            gt = ground_truth[var]
            pred = predictions[key]
            if not np.isnan(gt) and not np.isnan(pred):
                error = abs(pred - gt)
                errors.append(error)
                within_1.append(1 if error <= 1.0 else 0)
                pred_values.append(pred)
                gt_values.append(gt)

    if not errors:
        return {"mae": np.nan, "rmse": np.nan, "r_squared": np.nan, "within_1": np.nan}

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    within_1_pct = np.mean(within_1)

    # Compute R-squared
    ss_res = np.sum([(gt - pred)**2 for gt, pred in zip(gt_values, pred_values)])
    ss_tot = np.sum([(gt - np.mean(gt_values))**2 for gt in gt_values])
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "within_1": float(within_1_pct)
    }


def create_distribution_plots(results: Dict, output_dir: str = "docs/data") -> str:
    """
    Create distribution plots comparing predicted vs actual values.

    Parameters:
    -----------
    results : Dict
        Validation results with respondent predictions and ground truth
    output_dir : str
        Directory to save plots

    Returns:
    --------
    plot_path : str
        Path to saved plot
    """
    # Extract predictions and ground truth for each variable
    VAR_TO_KEY = {
        "V241397": ("urban_unrest", "Urban Unrest (1=solve problems, 7=use force)"),
        "V241308x": ("death_penalty", "Death Penalty (1=favor, 7=oppose)"),
        "V241272x": ("crime_spending", "Crime Spending (1=increase, 5=decrease)")
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LLM Predictions vs Ground Truth - Crime Policy Variables', fontsize=14, fontweight='bold')

    for idx, (var, (key, title)) in enumerate(VAR_TO_KEY.items()):
        predicted = []
        actual = []

        for resp in results["respondents"]:
            if key in resp["predictions"] and var in resp["ground_truth"]:
                pred = resp["predictions"][key]
                gt = resp["ground_truth"][var]
                if not np.isnan(pred) and not np.isnan(gt):
                    predicted.append(pred)
                    actual.append(gt)

        ax = axes[idx]

        # Create overlapping histograms with correct bins for each variable
        if var == "V241397":  # Urban unrest: 1-7
            bins = np.arange(0.5, 8.5, 1)
        elif var == "V241308x":  # Death penalty: 1-4
            bins = np.arange(0.5, 5.5, 1)
        else:  # V241272x - Crime spending: 1-5
            bins = np.arange(0.5, 6.5, 1)

        ax.hist(actual, bins=bins, alpha=0.6, label='Actual', color='blue', edgecolor='black')
        ax.hist(predicted, bins=bins, alpha=0.6, label='Predicted', color='red', edgecolor='black')

        ax.set_xlabel('Response Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation info
        if len(predicted) > 0 and len(actual) > 0:
            corr = np.corrcoef(predicted, actual)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "llm_validation_distributions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def run_individual_llm_validation(
    data_file: str,
    feature_specs: Dict[str, Dict],
    api_key: str,
    use_anthropic: bool = False,
    n_samples: int = 30,
    random_state: int = 42
) -> Dict:
    """
    Run LLM validation on individual respondents.

    Parameters:
    -----------
    data_file : str
        Path to ANES CSV file
    feature_specs : Dict[str, Dict]
        Feature specifications from CLUSTERING_FEATURES_V2
    api_key : str
        LLM API key
    use_anthropic : bool
        Whether to use Anthropic (Claude) instead of OpenAI
    n_samples : int
        Number of respondents to test (default 30)
    random_state : int
        Random seed

    Returns:
    --------
    results : Dict
        Full validation results including predictions and accuracy metrics
    """
    print("\n[LLM Validation - Individual Respondents]")
    print(f"  Model: {'claude-sonnet-4-5' if use_anthropic else 'gpt-4o-mini'}")
    print(f"  Samples: {n_samples}")

    # Load codebook with full question text and scales
    print("  Loading ANES codebook...")
    codebook = load_codebook()
    print(f"  Loaded codebook with {len(codebook)} variables")

    # Get all features
    all_features = list(feature_specs.keys())

    # Load and sample respondents
    sample_df = load_and_sample_respondents(data_file, all_features, n_samples, random_state)

    # Run predictions
    results = {
        "experiment": "individual_crime_prediction",
        "description": "Predict crime policy positions from all other policy positions for individual respondents",
        "model": "claude-sonnet-4-5" if use_anthropic else "gpt-4o-mini",
        "n_samples": len(sample_df),
        "respondents": []
    }

    all_accuracies = []

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\n  Respondent {idx}/{len(sample_df)}:")

        # Extract non-crime features (INPUT for LLM)
        # Keep ALL values including 8, 99, and negatives - the LLM can handle voters who don't know much
        non_crime_data = {}
        for var in all_features:
            if var not in CRIME_VARIABLES and var in row.index:
                value = row[var]
                # Keep all values - don't filter out 8, 99, or negatives
                if pd.notna(value):  # Only skip actual NaN values
                    non_crime_data[var] = float(value)

        # Extract crime ground truth (OUTPUT for validation)
        # These are already guaranteed to be valid (1-7) due to filtering in load_and_sample_respondents
        crime_truth = {}
        for var in CRIME_VARIABLES.keys():
            if var in row.index:
                value = row[var]
                if pd.notna(value):
                    crime_truth[var] = float(value)

        # Build prompt and get predictions
        prompt = build_llm_prompt_individual(non_crime_data, feature_specs, codebook)
        predictions = query_llm_for_individual_predictions(prompt, api_key, use_anthropic)

        # Compute accuracy
        accuracy = compute_individual_accuracy(predictions, crime_truth)

        print(f"    Predictions: {predictions}")
        print(f"    Ground truth: {crime_truth}")
        print(f"    MAE: {accuracy['mae']:.2f}, RMSE: {accuracy['rmse']:.2f}, R²: {accuracy['r_squared']:.3f}, Within 1pt: {accuracy['within_1']*100:.0f}%")

        all_accuracies.append(accuracy)

        results["respondents"].append({
            "respondent_id": idx,
            "predictions": predictions,
            "ground_truth": crime_truth,
            "accuracy": accuracy
        })

    # Compute aggregate metrics
    valid_mae = [a["mae"] for a in all_accuracies if not np.isnan(a["mae"])]
    valid_rmse = [a["rmse"] for a in all_accuracies if not np.isnan(a["rmse"])]
    valid_r_squared = [a["r_squared"] for a in all_accuracies if not np.isnan(a["r_squared"])]
    valid_within_1 = [a["within_1"] for a in all_accuracies if not np.isnan(a["within_1"])]

    results["aggregate"] = {
        "mean_mae": float(np.mean(valid_mae)) if valid_mae else np.nan,
        "mean_rmse": float(np.mean(valid_rmse)) if valid_rmse else np.nan,
        "mean_r_squared": float(np.mean(valid_r_squared)) if valid_r_squared else np.nan,
        "mean_within_1": float(np.mean(valid_within_1)) if valid_within_1 else np.nan,
        "n_respondents_tested": len(sample_df)
    }

    print(f"\n  AGGREGATE RESULTS:")
    print(f"    Mean MAE: {results['aggregate']['mean_mae']:.2f}")
    print(f"    Mean RMSE: {results['aggregate']['mean_rmse']:.2f}")
    print(f"    Mean R-squared: {results['aggregate']['mean_r_squared']:.3f}")
    print(f"    Mean Within 1pt: {results['aggregate']['mean_within_1']*100:.0f}%")

    # Create distribution plots
    print(f"\n  Creating distribution plots...")
    plot_path = create_distribution_plots(results)
    results["plot_path"] = plot_path
    print(f"    Saved to: {plot_path}")

    return results

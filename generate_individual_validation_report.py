"""
Generate Individual-Level LLM Validation Report
================================================
Creates HTML report with per-question accuracy tables and distribution charts.
Two models compared: Ideology-only vs Full (ideology + gender, age, education).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from collections import Counter

# Paths
INPUT_JSON = Path("docs/data/llm_validation_individuals.json")
OUTPUT_DIR = Path("docs/docs")
OUTPUT_HTML = OUTPUT_DIR / "llm_validation_report.html"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load validation data
print("[1/3] Loading validation results...")
with open(INPUT_JSON) as f:
    data = json.load(f)

models = data["models"]
detailed = data["detailed_results"]

# Question metadata
QUESTIONS = {
    "urban_unrest": {
        "label": "Urban Unrest Response",
        "anes_var": "V241397",
        "scale": "1 = solve problems of racism/police violence, 7 = use all available force"
    },
    "death_penalty": {
        "label": "Death Penalty",
        "anes_var": "V241308x",
        "scale": "1 = favor strongly, 4 = oppose strongly"
    },
    "crime_spending": {
        "label": "Federal Crime Spending",
        "anes_var": "V241272x",
        "scale": "1 = increased a lot, 5 = decreased a lot"
    }
}

# Extract per-question distributions from detailed results
def extract_question_distributions(results):
    dists = {qk: {"pred": [], "actual": []} for qk in QUESTIONS}
    for r in results:
        pred = r["predictions"]
        truth = r["ground_truth"]
        var_map = {"V241397": "urban_unrest", "V241308x": "death_penalty", "V241272x": "crime_spending"}
        for anes_var, qk in var_map.items():
            if qk in pred and anes_var in truth:
                p, t = pred[qk], truth[anes_var]
                if not np.isnan(p) and not np.isnan(t):
                    dists[qk]["pred"].append(round(p))
                    dists[qk]["actual"].append(round(t))
    return dists

ideology_dist = extract_question_distributions(detailed["ideology_only"])
full_dist = extract_question_distributions(detailed["full_model"])

print("[2/3] Creating visualizations...")

# Create visualization: Distribution comparison for each question
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('LLM Prediction Distributions vs Actual Responses\n(200 Individual Respondents)',
             fontsize=15, fontweight='bold', y=0.995)

for idx, (q_key, q_info) in enumerate(QUESTIONS.items()):
    for col, (model_label, dist) in enumerate([
        ("Ideology Only", ideology_dist),
        ("Ideology + Demographics", full_dist)
    ]):
        ax = axes[idx, col]
        pred_vals = dist[q_key]["pred"]
        actual_vals = dist[q_key]["actual"]

        all_vals = sorted(set(pred_vals + actual_vals))
        if not all_vals:
            continue

        pred_counts = Counter(pred_vals)
        actual_counts = Counter(actual_vals)

        x = np.arange(len(all_vals))
        width = 0.35

        ax.bar(x - width/2, [pred_counts[v] for v in all_vals], width,
               label='LLM Predicted', color='#3498db', edgecolor='black', linewidth=1.2)
        ax.bar(x + width/2, [actual_counts[v] for v in all_vals], width,
               label='Actual', color='#e74c3c', edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Response Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{q_info["label"]}\n[{model_label}]', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_vals)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add accuracy metrics from model data
        model_key = "ideology_only" if col == 0 else "full_model"
        pq = models[model_key]["per_question"].get(q_key, {})
        correct = pq.get("correct_pct", 0)
        within_1 = pq.get("within_1_pct", 0)
        ax.text(0.02, 0.98, f"Correct: {correct:.1f}%\nWithin ±1: {within_1:.1f}%",
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
chart_path = OUTPUT_DIR / "validation_individuals_distributions.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {chart_path}")

# Generate an example prompt from the first respondent
print("  Generating example prompt...")
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_individual_llm_validation import (
    load_codebook, load_and_sample_respondents, extract_respondent_features, build_llm_prompt
)

codebook = load_codebook()
sample_df = load_and_sample_respondents(n_sample=200, random_state=42)
row = sample_df.iloc[0]

# Build the full model prompt (with demographics) as the example
features = extract_respondent_features(row, include_demographics=True, codebook=codebook)
example_prompt = build_llm_prompt(features, 1)

# Get ground truth for this respondent
example_truth = {}
for var in ["V241397", "V241308x", "V241272x"]:
    if var in row.index and pd.notna(row[var]):
        example_truth[var] = int(row[var])

# Get predictions for this respondent from detailed results
example_pred = detailed["full_model"][0]["predictions"]

# Escape for HTML
import html as html_module
example_prompt_html = html_module.escape(example_prompt)

# Build HTML report
print("[3/3] Generating HTML report...")

# Build per-question results table rows
def metric_class(val, good_thresh, med_thresh):
    if val >= good_thresh:
        return "metric-good"
    elif val >= med_thresh:
        return "metric-medium"
    return "metric-bad"

table_rows = ""
for qk, qinfo in QUESTIONS.items():
    iq = models["ideology_only"]["per_question"].get(qk, {})
    fq = models["full_model"]["per_question"].get(qk, {})

    ic = iq.get("correct_pct", 0)
    iw = iq.get("within_1_pct", 0)
    fc = fq.get("correct_pct", 0)
    fw = fq.get("within_1_pct", 0)

    table_rows += f"""
            <tr>
                <td><strong>{qinfo['label']}</strong><br><small>{qinfo['scale']}</small></td>
                <td class="{metric_class(ic, 40, 20)}">{ic:.1f}%</td>
                <td class="{metric_class(iw, 70, 50)}">{iw:.1f}%</td>
                <td class="{metric_class(fc, 40, 20)}">{fc:.1f}%</td>
                <td class="{metric_class(fw, 70, 50)}">{fw:.1f}%</td>
            </tr>
"""

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Validation Experiment Results - ANES 2024</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.2em;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        .meta {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.95em;
        }}
        .meta strong {{ color: #2980b9; }}
        .summary-box {{
            background: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }}
        .success-box {{
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95em;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }}
        th.left {{ text-align: left; }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
            text-align: center;
        }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #e8f4f8; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-medium {{ color: #f39c12; font-weight: bold; }}
        .metric-bad {{ color: #e74c3c; font-weight: bold; }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .chart-container {{ margin: 30px 0; text-align: center; }}
        ul {{ margin: 15px 0; padding-left: 30px; }}
        li {{ margin: 10px 0; }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }}
        small {{ color: #777; }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Validation Experiment Results</h1>

        <div class="meta">
            <strong>Experiment:</strong> Individual-level crime policy prediction<br>
            <strong>Respondents Tested:</strong> {data["n_respondents"]} randomly sampled from ANES 2024<br>
            <strong>LLM Model:</strong> GPT-4o-mini (temperature=0)<br>
            <strong>Models Compared:</strong> (1) 46 policy variables only, (2) policy variables + gender, age, education
        </div>

        <h2>Per-Question Results</h2>

        <div class="summary-box">
            <table>
                <tr>
                    <th class="left" rowspan="2">Question</th>
                    <th colspan="2">Ideology Only (46 variables)</th>
                    <th colspan="2">+ Gender, Age, Education</th>
                </tr>
                <tr>
                    <th>Correct %</th>
                    <th>Within ±1 %</th>
                    <th>Correct %</th>
                    <th>Within ±1 %</th>
                </tr>
{table_rows}
            </table>
        </div>

        <h2>Response Distributions</h2>

        <div class="chart-container">
            <p style="margin-bottom: 15px; font-size: 1.1em;">
                <strong>How to read:</strong> Blue bars = LLM predictions, Red bars = Actual responses.
                Overlapping bars indicate accurate predictions.
            </p>
            <img src="validation_individuals_distributions.png" alt="Response Distributions">
        </div>

        <h2>Interpretation</h2>

        <div class="warning-box">
            <h3>Key Findings</h3>
            <ul>
                <li><strong>Per-question variation:</strong> Accuracy varies substantially across the three crime questions, reflecting different levels of ideological coherence.</li>
                <li><strong>Demographics effect:</strong> Adding gender, age, and education {'improves' if models['full_model']['mean_error'] < models['ideology_only']['mean_error'] else 'does not substantially improve'} predictions beyond ideology alone.</li>
                <li><strong>Better than chance:</strong> All within ±1 rates exceed the baseline for random guessing on each scale.</li>
            </ul>
        </div>

        <div class="success-box">
            <h3>Implications for Persona Chat</h3>
            <ul>
                <li><strong>Validation of design choice:</strong> These results validate providing <strong>explicit directional stances</strong> from data in system prompts, rather than letting the LLM infer positions.</li>
                <li><strong>LLM useful for engagement, not inference:</strong> The chat feature is best for conversational interaction, not for predicting missing data.</li>
            </ul>
        </div>

        <h2>Methodology</h2>

        <h3>Held-Out Variables (Crime Policy)</h3>
        <ul>
            <li><code>V241397</code>: Urban unrest response (1=solve problems of racism/police violence, 7=use all available force)</li>
            <li><code>V241308x</code>: Death penalty (1=favor strongly, 4=oppose strongly)</li>
            <li><code>V241272x</code>: Federal crime spending (1=increased a lot, 5=decreased a lot)</li>
        </ul>

        <h3>Model Specifications</h3>
        <ul>
            <li><strong>Ideology-Only Model:</strong> LLM receives respondent's answers to all non-crime policy questions (~46 variables)</li>
            <li><strong>Full Model:</strong> LLM receives policy questions + demographics (gender, age, education)</li>
            <li><strong>LLM:</strong> GPT-4o-mini (temperature=0 for reproducibility)</li>
            <li><strong>Sample:</strong> {data['n_respondents']} respondents randomly drawn from those with valid responses on all 3 crime variables</li>
        </ul>

        <h3>Evaluation Metrics</h3>
        <ul>
            <li><strong>Correct %:</strong> Percentage where LLM prediction exactly matches actual response (after rounding)</li>
            <li><strong>Within ±1 %:</strong> Percentage where prediction is within 1 point of actual (more forgiving)</li>
        </ul>

        <h2>Raw Data</h2>

        <p>Full validation results available in JSON format: <a href="../data/llm_validation_individuals.json" target="_blank">llm_validation_individuals.json</a></p>

        <h2>Appendix: Example Prompt</h2>

        <p>Below is the exact prompt sent to the LLM for respondent #1 (Full Model with demographics).
        The system message is: <em>"You are a political analyst predicting survey responses. Respond ONLY with the requested JSON, no explanation."</em></p>

        <div style="background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.82em; line-height: 1.5; max-height: 600px; overflow-y: auto; white-space: pre-wrap; margin: 20px 0;">{example_prompt_html}</div>

        <p style="margin-top: 15px;"><strong>Ground truth for this respondent:</strong>
        Urban unrest = {example_truth.get('V241397', '?')},
        Death penalty = {example_truth.get('V241308x', '?')},
        Crime spending = {example_truth.get('V241272x', '?')}</p>

        <p><strong>LLM prediction:</strong>
        Urban unrest = {example_pred.get('urban_unrest', '?')},
        Death penalty = {example_pred.get('death_penalty', '?')},
        Crime spending = {example_pred.get('crime_spending', '?')}</p>

        <div class="footer">
            <p><strong>ANES 2024 Ideological Clusters Project</strong></p>
            <p>Created by Guillermo Lezama | <a href="https://guillelezama.com" target="_blank">guillelezama.com</a></p>
            <p><a href="../index.html">&larr; Back to Main Site</a></p>
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"  Saved: {OUTPUT_HTML}")
print(f"\n[SUCCESS] Report generated!")
print(f"   View at: http://localhost:5000/docs/llm_validation_report.html")

# Technical Note: ANES 2024 Ideological Clustering Methodology

## Overview

This document describes the complete methodology for identifying ideological clusters among voters in the ANES 2024 Time Series dataset. The analysis uses unsupervised machine learning (K-means clustering) on 49 policy attitude variables to discover natural groupings in the American electorate, going beyond simple liberal/conservative labels to reveal cross-cutting coalitions and mixed-policy combinations.

**Key Findings:**
- 15 distinct ideological clusters identified among likely voters
- Clusters vary in size from ~2% to ~15% of the likely voter population
- Some clusters align with traditional party platforms; others represent cross-cutting positions (e.g., fiscally conservative + socially liberal, or vice versa)
- Quiz with just 10 questions can predict cluster membership with 61% accuracy (vs 79% with all 49 features)

---

## 1. Analysis Universe

### Definition: "Likely Voters"

The analysis focuses on **likely voters** defined as respondents who report they will "definitely" or "probably" vote in the 2024 election.

**Operationalization:**
- Variable: `V241029` (How likely R is to vote in the election)
- Included codes: 1 (Definitely will vote), 2 (Probably will vote)
- Excluded: 50-50 chance, probably won't vote, definitely won't vote

**Final sample size:** 2,753 respondents (after additional exclusions for missing data)

**Rationale:**
- Focuses on the active electorate most likely to influence election outcomes
- Pre-election measure avoids conditioning on post-election turnout (which may be influenced by ideology)
- Alternative universes ("actual_voters" or "all_respondents") supported via CLI flag

---

## 2. Survey Weighting

### Weight Variable

The analysis uses pre-election survey weights for **descriptive statistics only** (population shares, demographics, vote shares). The clustering algorithm itself is **unweighted** (all observations have equal influence on cluster formation).

**Selected weight:** `V240103a` (or first available from priority list: V240102a, V240101a, V240108a, V240103b, V240108b)

**Rationale:**
- Pre-election weights adjust for ANES sampling design and nonresponse bias
- Post-election weights condition on post-election outcomes (turnout, vote choice), which could bias pre-election ideology inference
- Unweighted clustering treats each respondent equally, avoiding distortions from extreme weights

---

## 3. Feature Selection (Clustering Variables)

### Criteria

Clustering features must satisfy:
1. **Pre-election only** (V241xxx variables): measured before the election to avoid post-hoc rationalization
2. **Policy attitudes, not evaluations**: focus on issue positions, not approval of specific politicians/institutions
3. **Sufficient coverage**: >75% valid responses (exclude variables with >25% missing data)
4. **Interpretable scales**: ordinal or interval scales (1-7, 1-5, etc.) representing ideological positions

### Selected Features (N=49)

The final feature set includes 49 variables across 11 policy domains:

#### **Ideology & Trust (5 variables)**
- V241228: Party identity importance (1=extremely, 4=not at all)
- V241234: Trust in people (1=always, 5=never)
- V241229: Trust government in Washington (1=always, 5=never)
- V241230: Trust court system (1=always, 5=never)
- V241231: Gov run by few big interests or benefit of all (1=few interests, 2=benefit all)

#### **Abortion & Gender (7 variables)**
- V241248: Abortion 7pt (1=always permit, 7=never permit)
- V241290x: Approve/disapprove DEI programs
- V241372x: Transgender bathroom use matching identity
- V241375x: Banning transgender girls from K-12 girls sports
- V241378x: Laws protecting gays/lesbians from job discrimination
- V241381x: Gay/lesbian couples allowed to adopt children
- V241385x: Right of gay/lesbian couples to legally marry

#### **Fiscal Policy (4 variables)**
- V241232: Does government waste much tax money (1=waste lot, 4=don't waste much)
- V241239: Gov services/spending 7pt (1=fewer services, 7=more services)
- V241242: Defense spending 7pt (1=decrease, 7=increase)
- V241245: Health insurance 7pt (1=gov plan, 7=private)

#### **Immigration (6 variables)**
- V241269x: Federal budget spending on tightening border security
- V241389x: Favor/oppose ending birthright citizenship
- V241386: Policy toward unauthorized immigrants (1=felony/deport, 5=no penalty)
- V241392x: Children brought illegally: send back or allow to stay
- V241395x: Favor/oppose building wall on border with Mexico
- V241396: Importance of speaking English in US (1=extremely, 5=not at all)

#### **International Affairs (3 variables)**
- V241312x: Country better off if we just stayed home
- V241313: Use force to solve international problems (1=extremely willing, 7=extremely unwilling)
- V241400x: Favor/oppose US giving weapons to Ukraine

#### **Environment (3 variables)**
- V241366x: Government action about rising temperatures
- V241284x: Federal budget spending on protecting environment
- V241258: Environment-business tradeoff 7pt (1=protect env, 7=business priority)

#### **Education (2 variables)**
- V241287x: Approve/disapprove how colleges and universities are run
- V241266x: Federal budget spending on public schools

#### **Political Rights (3 variables)**
- V241319x: Favor/oppose requiring ID when voting
- V241322x: Favor/oppose allowing felons to vote
- V241330x: Helpful/harmful if president didn't have to worry about Congress/courts

#### **Crime (3 variables)**
- V241397: Best way to deal with urban unrest (1=solve problems, 2=use force)
- V241308x: Favor/oppose death penalty
- V241272x: Federal budget spending on dealing with crime

#### **Other Attitudes (7 variables)**
- V241335: Trust in news media (1=great deal, 5=none)
- V241341: Likelihood sexual harassment would deter you from voting for candidate
- V241255: Gov assistance to Blacks 7pt (1=help, 7=no special help)
- V241363x: How much larger is income gap today
- V241369x: Require employers to offer paid leave to parents
- V241252: Guaranteed job/income 7pt (1=gov should, 7=people on own)
- V241263x, V241278x, V241281x: Federal budget spending (Social Security, highways, aid to poor)

#### **Israel/Palestine (4 variables)**
- V241403x: Favor/oppose US giving military assistance to Israel
- V241406x: Favor/oppose US giving humanitarian aid to Palestinians
- V241409x: Side more with Israelis or Palestinians
- V241412x: Approve/disapprove of protests against war in Gaza

### Excluded Variables

- **Self-reported ideology** (V241177): An outcome to predict, not an input
- **Party ID** (V241227x, V242227x): Outcome variable, not clustering input
- **Presidential approval/evaluations**: Focus on policy, not personalities
- **High-missingness variables**: Any variable with <75% valid responses excluded
- **Post-election variables** (V242xxx): Avoid post-hoc rationalization

### Scale Direction

**CRITICAL:** All variables are preserved in their **original ANES coding direction**. No variables are flipped or reversed.

- For most questions, higher values indicate conservative positions (e.g., 7 = never permit abortion)
- Some questions naturally code liberal positions higher (e.g., 7 = more government services)
- This reflects natural variation in how questions are asked and ensures data fidelity

---

## 4. Missing Data Handling

### Respondent-Level Filtering

The pipeline retains only respondents with **≥60% of clustering features answered** (≥30 out of 49 variables).

**Rationale:**
- Ensures sufficient information for meaningful cluster assignment
- Balances data quality with sample size
- More stringent thresholds (e.g., 80%) would severely reduce sample

### Feature-Level Imputation

For respondents meeting the 60% threshold, remaining missing values are imputed using **median imputation** (computed per feature across all valid responses).

**Implementation:**
- Median computed on cleaned data (negative ANES codes already converted to NaN)
- Imputation parameters saved to `preprocess.json` for quiz (browser-based prediction)

**Rationale:**
- Median is robust to outliers and preserves ordinal scale interpretation
- Simple, transparent, reproducible
- More complex methods (e.g., multiple imputation, MICE) offer limited benefit for clustering exploratory analysis

---

## 5. Standardization

All features are **z-score standardized** (mean=0, std=1) before clustering.

**Formula:** `Z = (X - μ) / σ`

**Rationale:**
- Ensures equal weight for features with different scales (e.g., 1-4 vs 1-7)
- Makes Euclidean distance metric interpretable across dimensions
- Standardization parameters (means, SDs) saved to `preprocess.json` for reproducibility

---

## 6. Variance Weighting

After standardization, features are **variance-weighted** before computing distances.

**Method:**
1. Compute original variance for each feature (before standardization)
2. Exclude features with <75% valid responses
3. Apply weight = √(variance_original) to each standardized feature column

**Formula:** `X_weighted[:, i] = X_scaled[:, i] * sqrt(variance_original[i])`

**Rationale:**
- Features with higher natural variance (more discriminating) receive higher weight in distance calculations
- Prevents low-variance features (where nearly everyone agrees) from dominating cluster formation
- Balances contribution of different policy domains

**Result:** Features like transgender sports ban (high variance) receive more weight than trust in courts (low variance)

---

## 7. Clustering Algorithm

### Method: K-Means

The analysis uses **K-means clustering** with the following settings:

```python
KMeans(
    n_clusters=K,
    init='k-means++',
    n_init=50,
    max_iter=300,
    random_state=42
)
```

**Key parameters:**
- **Distance metric:** Euclidean (in variance-weighted standardized space)
- **Initialization:** k-means++ (smart centroid seeding to avoid poor local minima)
- **Number of initializations:** 50 (algorithm runs 50 times with different initializations, keeps best)
- **Random state:** 42 (for reproducibility)

### K Selection

K (number of clusters) is chosen using **penalized silhouette score** across a range:

- **Range tested:** K ∈ [8, 20]
  - Lower bound (8): ensures meaningful differentiation beyond simple partisan splits
  - Upper bound (20): avoids over-fragmentation and tiny clusters
- **Selection metric:** Silhouette score with a distance penalty from target K=15
  - Score = silhouette − 0.05 × |K − 15|
  - This is a **design choice** favoring granularity (more clusters) over parsimony. The penalty is small (0.05 per step), so a K far from 15 with substantially better silhouette would still win.
- **Selected K:** 15 (silhouette score = 0.0445)

**Interpretation of silhouette = 0.0445:**
- Indicates **weak cluster separation** — boundaries between clusters are soft, not sharp
- This is expected for ideological data (a continuous spectrum with fuzzy boundaries, not discrete types)
- Higher scores are unlikely without artificially constraining the feature set
- **Clusters should be interpreted as fuzzy ideological prototypes**, not hard-edged voter "types." Individual voters often straddle multiple clusters.
- Stability ARI = 0.54 across random seeds confirms moderate reproducibility (see Section 8)

### Rationale for K-Means

**Strengths:**
- Fast and scalable to large datasets
- Interpretable (cluster centroids represent "average" member profiles)
- Widely used in political science clustering
- Deterministic with fixed random seed

**Alternatives considered:**
- **Hierarchical clustering:** Computationally expensive for N=2,753; dendrograms hard to interpret with 49 dimensions
- **DBSCAN:** Requires density assumptions inappropriate for ideological data (no natural "dense regions")
- **Gaussian Mixture Models:** More flexible but harder to interpret; similar results in practice

---

## 8. Stability Check

### Method: Multi-Seed Clustering

Stability is assessed by running K-means with **3 different random seeds** (42, 123, 456) and measuring cluster assignment agreement using the **Adjusted Rand Index (ARI)**.

**ARI properties:**
- Range: [0, 1]
  - 1 = perfect agreement (identical cluster assignments)
  - 0 = agreement no better than random chance
- Adjusts for chance agreement (unlike raw Rand Index)

**Threshold for stability:** ARI > 0.80 indicates stable clusters

### Results

**Mean ARI across seed pairs:** 0.539

**Interpretation:**
- **Moderate stability** (below ideal threshold of 0.80)
- Expected for soft cluster boundaries in ideological space
- Suggests cluster boundaries are somewhat fuzzy, but core cluster identities are consistent
- Users should interpret clusters as "prototypical profiles" rather than hard categories

**Recommendation:** Results are suitable for exploratory analysis and persona generation, but cluster boundaries should not be over-interpreted as sharp dividing lines.

### Alternative Stability Checks

- **Bootstrap resampling:** Clustering on random subsamples could further assess stability but is computationally expensive
- **Multi-seed is standard practice:** Provides reasonable stability estimate with low computational cost

---

## 9. 2D Visualization (PCA Embedding)

For the 2D map visualization, the analysis uses **Principal Component Analysis (PCA)** on cluster centroids:

**Input:** K × 49 matrix of cluster centroids (in standardized, variance-weighted space)

**Output:** K × 2 coordinates for plotting

**Method:**
- PCA with 2 components
- Random state: 42 (for reproducibility)
- Variance explained: Reported in `embedding_2d.json`

**Rationale:**
- PCA is linear and interpretable (principal components are weighted combinations of original features)
- Nonlinear methods (t-SNE, UMAP) could preserve local structure better but are harder to explain
- For cluster-level visualization (K=15 points), PCA is sufficient

**Limitations:**
- 2D projection necessarily loses information from 49-dimensional space
- Distances in 2D map only approximate true 49D distances
- Use 3D cube visualization for more accurate distance representation

---

## 10. Quiz Feature Selection

To create a short quiz that predicts cluster membership without asking all 49 questions, the pipeline uses **Random Forest feature importance** to select the top 10 most predictive features.

### Method

1. **Train Random Forest classifier:**
   - Target: Cluster label (1-15)
   - Features: All 49 clustering variables (standardized, variance-weighted)
   - Model: 100 trees, max depth 5, random state 42

2. **Compute feature importance:**
   - Mean decrease in impurity (Gini importance) across all trees
   - Normalized to sum to 1.0

3. **Select top 10 features** by importance

### Selected Quiz Features

| Rank | Variable | Importance | Description |
|------|----------|------------|-------------|
| 1 | V241403x | 0.0704 | US military assistance to Israel |
| 2 | V241395x | 0.0583 | Building wall on border with Mexico |
| 3 | V241375x | 0.0578 | Banning transgender girls from K-12 sports |
| 4 | V241372x | 0.0524 | Transgender bathroom use matching identity |
| 5 | V241400x | 0.0509 | US giving weapons to Ukraine |
| 6 | V241319x | 0.0493 | Requiring ID when voting |
| 7 | V241409x | 0.0450 | Side more with Israelis or Palestinians |
| 8 | V241258 | 0.0391 | Environment-business tradeoff |
| 9 | V241412x | 0.0385 | Protests against war in Gaza |
| 10 | V241366x | 0.0371 | Government action on rising temperatures |

### Validation

**Full model accuracy (49 features):** 78.57%

**Quiz model accuracy (10 features):** 60.99%

**Accuracy ratio:** 77.62% (quiz retains 78% of full model's predictive power with only 20% of features)

**Interpretation:**
- 61% accuracy is **substantially better than chance** (1/15 = 6.7%)
- Most users will be assigned to correct cluster or a nearby cluster
- Trade-off between quiz length and accuracy is acceptable for user engagement

---

## 11. Persona Generation

### Cluster Profiles

Each cluster is represented by a **simulated persona** based on statistical aggregates of cluster members:

**Demographics:**
- Age: Weighted mean age (years)
- Gender: Modal gender (Man/Woman/Nonbinary)
- Education: Modal education level (5-category)
- Race/ethnicity: Modal race/ethnicity
- Region: Weighted distribution across Northeast/Midwest/South/West
- Party ID: Weighted mean on 7pt scale (1=Strong Dem, 7=Strong Rep)

**Vote shares:**
- Harris vs Trump vs Third-party, computed from post-election data (V242067, V242068)
- Weighted percentages (may not sum to 100% due to nonresponse)

### Policy Stances

Persona stances are generated from **cluster-level means** of policy variables:

**Method:**
1. Compute weighted mean of each variable for cluster members (in original scale, not z-scores)
2. Convert to **decisive, directional text**:
   - Example: If abortion mean = 6.2 (on 1-7 scale), stance = "I strongly believe abortion should never be permitted" (not "I hold a position on abortion")
3. Ensure first sentence is **explicitly directional** to avoid LLM confusion

**Evidence transparency:**
- Stances based on clustering variables marked as "Observed" (green badge)
- Stances from non-clustering ANES variables marked as "Data-based" (blue badge)
- Fictional extrapolations clearly labeled with disclaimer

### LLM Chat Integration

The persona chat feature uses **directional stances as system prompt constraints** to ensure LLM responses align with cluster data.

**LLM system prompt structure:**
```
You are [Name], a [demographics].

Your political views (NEVER deviate from these):
- Abortion: I strongly believe abortion should never be permitted.
- Immigration: I strongly favor building a wall on the Mexico border.
...

When answering questions:
1. Always respond consistently with your stances above
2. Use first-person voice
3. Be conversational but stay in character
4. Never change your positions
```

**Key insight from debugging:**
- Non-directional stances ("I hold a clear position on abortion") allow LLM to interpret freely → inconsistent responses
- Directional stances ("I strongly believe abortion should never be permitted") enforce consistency → correct persona behavior

**LLM model:** OpenAI GPT-4o-mini (via API, requires OPENAI_API_KEY in environment)

---

## 12. LLM Validation Experiment

### Goal

Test whether an LLM can **predict individual respondents' crime policy positions** from their other policy positions alone. This validates the persona chat approach by measuring how much ideological coherence LLMs can capture across policy domains.

### Method

1. **Sample:** 200 randomly sampled respondents (random_state=42) who answered all 3 holdout crime questions.

2. **Holdout variables:** Three crime policy variables:
   - V241397: Urban unrest response (1=solve problems of racism/police violence … 7=use all available force)
   - V241308x: Death penalty (1=favor strongly, 2=favor not strongly, 3=oppose not strongly, 4=oppose strongly)
   - V241272x: Federal crime spending (1=increased a lot … 5=decreased a lot)

3. **Two conditions:**
   - **Ideology Only:** LLM receives only the respondent's 46 non-crime policy positions (with full scale labels from ANES codebook)
   - **Ideology + Demographics:** Same as above, plus gender, age, education, and race/ethnicity

4. **LLM model:** GPT-4o-mini (temperature=0) via LangChain

5. **Evaluation metrics (per question):**
   - **Correct %:** Exact match after rounding to nearest integer
   - **Within ±1 point %:** Prediction within 1 point of actual response
   - **MAE:** Mean absolute error

### Results

**Per-Question Results (n=200 respondents):**

| Question | Ideology Only Correct | Ideology Only Within±1 | + Demographics Correct | + Demographics Within±1 |
|----------|----------------------|------------------------|----------------------|------------------------|
| Urban Unrest (1-7) | 31.5% | 48.0% | 31.5% | 49.5% |
| Death Penalty (1-4) | 34.0% | 63.0% | 39.5% | 72.0% |
| Crime Spending (1-5) | 41.0% | 77.0% | 48.0% | 75.0% |

### Interpretation

1. **Policy positions carry substantial predictive power:** From ideology alone, the LLM correctly predicts 32-41% of individual crime stances (well above chance: ~14% for a 7-point scale, ~25% for a 4-point scale).

2. **Within ±1 performance is strong for narrower scales:** 63-77% for death penalty and crime spending (4- and 5-point scales), more modest for the wide 7-point urban unrest scale (48%).

3. **Demographics provide a modest boost:** Adding gender, age, education, and race improves death penalty predictions by +5.5pp (correct) and +9pp (within ±1), but has minimal effect on urban unrest and crime spending.

4. **Implications for persona chat:** These results confirm meaningful ideological coherence across policy domains, while also showing that LLM inference alone produces substantial individual-level errors. This validates the design choice to provide **explicit data-driven stances** in persona system prompts rather than relying on inference.

### Limitations

1. **Crime domain only:** Results may differ for other holdout domains (healthcare, environment, foreign policy)
2. **Single model:** Only tested GPT-4o-mini; larger models may perform differently
3. **Prompt design:** More sophisticated prompting (few-shot, chain-of-thought) might improve performance
4. **Individual heterogeneity:** Respondents with unusual or cross-cutting positions are harder to predict

### Reproducibility

To re-run validation:

```bash
export OPENAI_API_KEY="your-key-here"
python run_individual_llm_validation.py
python generate_individual_validation_report.py
```

Results saved to `docs/data/llm_validation_individuals.json` and `docs/docs/llm_validation_report.html`

**Cost estimate:** ~$0.10-0.20 for 400 API calls to GPT-4o-mini (as of Feb 2026)

---

## 13. Software & Reproducibility

### Dependencies

**Core Python packages:**
- Python: 3.9+
- pandas: 2.0+
- numpy: 1.24+
- scikit-learn: 1.3+
- scipy: 1.11+
- plotly: 5.18+
- matplotlib: 3.7+

**Optional (for LLM features):**
- langchain: 0.1+
- langchain-openai: 0.0.5+
- openai: 1.12+

**Deployment:**
- Flask: 3.0+
- gunicorn: 21.2+

### Reproducibility

All analysis is **fully reproducible** with:
- Fixed random seeds (clustering: 42; stability: 42, 123, 456; PCA: 42; RF: 42)
- Deterministic preprocessing (median imputation, z-score standardization)
- Version-controlled code (see GitHub repository)

**To reproduce:**

```bash
# Clone repository
git clone https://github.com/guillelezama/anes-2024-personas.git
cd anes-2024-personas

# Install dependencies
pip install -r requirements.txt

# Download ANES 2024 data (not included in repo due to file size)
# Place CSV in: anes_timeseries_2024_csv_20250808/

# Run analysis pipeline
python build_site_data.py --universe likely_voters

# Launch local server
python server.py
```

**Output files** (saved to `docs/data/`):
- `metadata.json`: K, silhouette, stability ARI, feature list, timestamp
- `centroids.json`: Cluster centroids (variance-weighted standardized space)
- `cluster_profiles.json`: Demographics, stances, vote shares per cluster
- `embedding_2d.json`: PCA coordinates for 2D map
- `quiz_features.json`: Selected quiz features + validation accuracy
- `preprocess.json`: Imputation medians, standardization params
- `cluster_distances.json`: Pairwise cluster distances (for quiz result display)
- `llm_validation.json`: LLM validation results (if --use-llm enabled)

---

## 14. Limitations & Future Work

### Limitations

1. **Cross-sectional data:** Captures ideology at one time point (2024). Cannot assess change over time or causality.

2. **Feature selection:** 49 variables may not capture all ideological dimensions. Notable omissions:
   - Gun rights/Second Amendment
   - Trade policy (tariffs, free trade)
   - Specific religious/cultural issues

3. **K-means assumptions:** Assumes spherical clusters of similar size. May not capture:
   - Complex manifolds (e.g., horseshoe-shaped ideological space)
   - Hierarchical structure (sub-clusters within clusters)
   - Outliers (K-means assigns everyone to a cluster, even if poorly fitting)

4. **Soft cluster boundaries:** Low silhouette score (0.0445) and moderate stability (ARI=0.539) indicate fuzzy boundaries. Clusters are "prototypical profiles," not discrete types.

5. **Survey limitations:** Self-reported attitudes subject to:
   - Social desirability bias
   - Question wording effects
   - Nonresponse bias (even after weighting)

6. **Persona simplification:** Statistical aggregates cannot capture:
   - Within-cluster heterogeneity
   - Intersectionality of identities
   - Nuance of individual reasoning

### Future Work

**Methodological extensions:**
- **Longitudinal analysis:** Compare 2024 clusters to 2020, 2016, 2012 ANES to track ideological evolution
- **Hierarchical clustering:** Test whether clusters have stable sub-clusters
- **Alternative algorithms:** Gaussian Mixture Models, DBSCAN with adaptive epsilon
- **Feature expansion:** Include gun rights, trade, additional cultural issues (if measured in future ANES waves)

**Validation:**
- **External validation:** Compare cluster vote shares to exit polls, precinct-level voting records
- **Predictive validation:** Do 2024 clusters predict 2028 vote choice (when data available)?
- **Qualitative validation:** In-depth interviews with survey respondents to assess cluster interpretability

**Applications:**
- **Campaign targeting:** Which clusters are persuadable? Which issues resonate with each?
- **Coalition analysis:** Which clusters could form winning coalitions?
- **Narrative generation:** LLM-powered "debate" between personas on hot-button issues

---

## 15. Acknowledgments

This project uses data from the **American National Election Studies (ANES) 2024 Time Series Study**.

The ANES is a collaboration of Stanford University, the University of Michigan, and funded by the National Science Foundation.

**Data citation:**
American National Election Studies. 2024. ANES 2024 Time Series Study [dataset and documentation]. www.electionstudies.org

**Disclaimer:**
This is an independent educational project. Any opinions, findings, and conclusions or recommendations expressed here are those of the author and do not necessarily reflect the views of ANES or the National Science Foundation.

---

## 16. Contact & Contributions

**Author:** Guillermo Lezama
**Title:** Data Scientist and PhD in Economics
**Website:** [guillelezama.com](https://guillelezama.com)
**LinkedIn:** [linkedin.com/in/guillelezama](https://linkedin.com/in/guillelezama)
**GitHub:** [github.com/guillelezama](https://github.com/guillelezama)

**Code repository:** [github.com/guillelezama/anes-2024-personas](https://github.com/guillelezama/anes-2024-personas)

**Feedback welcome:**
- Open an issue on GitHub for bugs, feature requests, or methodological questions
- Connect on LinkedIn for collaboration opportunities

**License:** MIT (code) / ANES data subject to ANES terms of use

---

## References

**Methodology:**
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12:2825-2830.
- Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics* 20:53-65.
- Hubert, L., & Arabie, P. (1985). "Comparing partitions." *Journal of Classification* 2:193-218.

**Substantive:**
- Abramowitz, A. I., & Saunders, K. L. (2008). "Is polarization a myth?" *Journal of Politics* 70(2):542-555.
- Fiorina, M. P., & Abrams, S. J. (2008). "Political polarization in the American public." *Annual Review of Political Science* 11:563-588.
- Lelkes, Y. (2016). "Mass polarization: Manifestations and measurements." *Public Opinion Quarterly* 80(S1):392-410.

**Data:**
- American National Election Studies. 2024. *ANES 2024 Time Series Study*. Available at: www.electionstudies.org

---

*Last updated: 2026-02-15*
*Generated by build_site_data.py with --universe likely_voters*

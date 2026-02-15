"""
Persona generation using LangChain.
Creates avatars.json (persona definitions, names, stance wording) and story.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Import clustering features
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.preprocessing_v2 import CLUSTERING_FEATURES_V2

# Load question mappings for scale endpoint information
QUESTION_MAPPINGS = {}
question_mappings_path = Path(__file__).parent.parent / "site" / "data" / "question_mappings.json"
if question_mappings_path.exists():
    with open(question_mappings_path, 'r', encoding='utf-8') as f:
        QUESTION_MAPPINGS = json.load(f)


def generate_persona_name(cluster_id: int, cluster_stats: Dict) -> str:
    """
    Generate a fictitious persona name based on cluster characteristics.
    Deterministic fallback if no LLM available.
    """
    # Deterministic names (no protected demographics)
    names = [
        "Taylor", "Jordan", "Morgan", "Casey", "Alex",
        "Riley", "Quinn", "Avery", "Cameron", "Sage",
        "Skylar", "Dakota", "Reese", "Parker", "Blake",
        "River", "Finley", "Emerson", "Rowan", "Harper"
    ]
    return names[cluster_id % len(names)]


def generate_stance_wording_deterministic(topic: str, value: float, evidence_type: str, cluster_stats: Dict) -> Dict:
    """
    Generate decisive first-sentence stance wording (deterministic).
    """
    # Abortion
    if topic == "abortion":
        if value >= 5.0:
            decisive = "I am against abortion."
            detail = "I believe life begins at conception and abortion should be heavily restricted or banned."
        elif value <= 3.0:
            decisive = "I support abortion rights."
            detail = "I believe women should have the right to choose and abortion should remain legal and accessible."
        else:
            decisive = "I have mixed views on abortion."
            detail = "I support some restrictions but believe abortion should be available in certain circumstances."

    # Immigration
    elif topic == "immigration":
        if value >= 4.0:
            decisive = "I support strict immigration enforcement."
            detail = "I believe we need strong borders and should prioritize deporting unauthorized immigrants."
        elif value <= 2.0:
            decisive = "I support compassionate immigration reform."
            detail = "I believe we should provide paths to citizenship and treat immigrants humanely."
        else:
            decisive = "I support balanced immigration policy."
            detail = "I believe we need border security but also humane treatment of immigrants."

    # Redistribution
    elif topic == "redistribution":
        if value >= 5.0:
            decisive = "I oppose big government."
            detail = "I believe government should be small and people should rely on themselves, not handouts."
        elif value <= 3.0:
            decisive = "I support government services."
            detail = "I believe government should provide more services and help people who need it."
        else:
            decisive = "I support moderate government."
            detail = "I believe we need some government services but also fiscal responsibility."

    # Environment
    elif topic == "environment":
        if value >= 5.0:
            decisive = "I oppose burdensome environmental regulations."
            detail = "I believe environmental rules hurt businesses and cost jobs."
        elif value <= 3.0:
            decisive = "I support strong environmental protections."
            detail = "I believe we must act on climate change and protect our planet for future generations."
        else:
            decisive = "I support balanced environmental policy."
            detail = "I believe we can protect the environment while supporting business growth."

    # Health
    elif topic == "health":
        if value >= 5.0:
            decisive = "I oppose government-run healthcare."
            detail = "I believe healthcare should be handled by private insurance, not government."
        elif value <= 3.0:
            decisive = "I support government healthcare."
            detail = "I believe healthcare is a right and government should ensure everyone has coverage."
        else:
            decisive = "I support a mixed healthcare system."
            detail = "I believe we need both private and public options for healthcare."

    # Self ideology
    elif topic == "ideology":
        if value >= 5.0:
            decisive = "I am conservative."
            detail = "I believe in traditional values, limited government, and personal responsibility."
        elif value <= 3.0:
            decisive = "I am liberal."
            detail = "I believe in progressive values, government action, and social justice."
        else:
            decisive = "I am moderate."
            detail = "I hold views from both sides and don't fit neatly into liberal or conservative labels."

    # Religion
    elif topic == "religion":
        if value >= 4.5:
            decisive = "I am religiously observant."
            detail = "My faith is very important to me and I attend services regularly."
        elif value <= 2.5:
            decisive = "I am not religiously observant."
            detail = "Religion is not a major part of my life and I rarely or never attend services."
        else:
            decisive = "I am moderately religious."
            detail = "I have some religious beliefs but don't attend services very regularly."

    # Efficacy/trust
    elif topic == "efficacy_trust":
        if value >= 4.0:
            decisive = "I don't trust politicians."
            detail = "I believe officials don't care what ordinary people think and the system is broken."
        elif value <= 2.0:
            decisive = "I have faith in democratic institutions."
            detail = "I believe officials generally try to do what's right and the system works."
        else:
            decisive = "I am skeptical but hopeful."
            detail = "I have concerns about politicians but believe the system can improve."

    # Democracy
    elif topic == "democracy":
        if value >= 3.0:
            decisive = "I think democracy is overrated."
            detail = "I believe strong leadership is more important than democratic processes sometimes."
        elif value <= 2.0:
            decisive = "I strongly value democracy."
            detail = "I believe democracy is extremely important and must be protected at all costs."
        else:
            decisive = "I value democracy with reservations."
            detail = "I think democracy is important but has its flaws."

    # Vaccines
    elif topic == "vaccines":
        if value >= 4.0:
            decisive = "I oppose vaccine mandates."
            detail = "I believe requiring vaccines for school infringes on parental rights and personal freedom."
        elif value <= 2.0:
            decisive = "I support vaccine requirements."
            detail = "I believe vaccines should be required for public school to protect public health."
        else:
            decisive = "I am unsure about vaccine mandates."
            detail = "I see both sides of the vaccine debate and have mixed feelings."

    # Defense spending (clustering feature)
    elif topic == "defense":
        if value >= 5.0:
            decisive = "I support increased defense spending."
            detail = "I believe we need a strong military and should invest more in defense."
        elif value <= 3.0:
            decisive = "I support reduced defense spending."
            detail = "I believe we spend too much on defense and should redirect funds to other priorities."
        else:
            decisive = "I support current defense spending levels."
            detail = "I believe our current military budget is about right."

    # Welfare spending (clustering feature)
    elif topic == "welfare":
        if value >= 5.0:
            decisive = "I oppose welfare spending."
            detail = "I believe welfare spending should be decreased and people should be more self-reliant."
        elif value <= 3.0:
            decisive = "I support welfare spending."
            detail = "I believe we should increase welfare spending to help those in need."
        else:
            decisive = "I support moderate welfare spending."
            detail = "I believe we need some welfare programs but should be careful about costs."

    # Defense spending (holdout ML - duplicate topic name, won't occur)
    elif topic == "defense_spending":
        if value >= 5.0:
            decisive = "I support increased defense spending."
            detail = "I believe we need a strong military and should invest more in defense."
        elif value <= 3.0:
            decisive = "I support reduced defense spending."
            detail = "I believe we spend too much on defense and should redirect funds to other priorities."
        else:
            decisive = "I support current defense spending levels."
            detail = "I believe our current military budget is about right."

    # Guaranteed jobs (holdout ML)
    elif topic == "guaranteed_jobs":
        if value >= 5.0:
            decisive = "I oppose government job guarantees."
            detail = "I believe people should find jobs on their own, not rely on government."
        elif value <= 3.0:
            decisive = "I support government job guarantees."
            detail = "I believe government should ensure everyone has a job and income."
        else:
            decisive = "I have mixed views on job guarantees."
            detail = "I see merit in both individual responsibility and government support."

    # Gay marriage (holdout ML)
    elif topic == "gay_marriage":
        if value >= 4.0:
            decisive = "I oppose same-sex marriage."
            detail = "I believe marriage should be between a man and a woman."
        elif value <= 2.0:
            decisive = "I support same-sex marriage."
            detail = "I believe everyone should have the right to marry whom they love."
        else:
            decisive = "I am ambivalent about same-sex marriage."
            detail = "I don't have strong feelings either way on this issue."

    # Israel aid (holdout ML)
    elif topic == "israel_aid":
        if value >= 4.0:
            decisive = "I support military aid to Israel."
            detail = "I believe the U.S. should continue providing strong support to Israel."
        elif value <= 2.0:
            decisive = "I oppose military aid to Israel."
            detail = "I believe the U.S. should stop or reduce military assistance to Israel."
        else:
            decisive = "I am uncertain about Israel aid."
            detail = "I have mixed feelings about U.S. military support for Israel."

    # Ukraine aid (holdout ML)
    elif topic == "ukraine_aid":
        if value >= 4.0:
            decisive = "I support weapons for Ukraine."
            detail = "I believe the U.S. should help Ukraine defend itself against Russian aggression."
        elif value <= 2.0:
            decisive = "I oppose weapons for Ukraine."
            detail = "I believe the U.S. should not be involved in the Ukraine conflict."
        else:
            decisive = "I am uncertain about Ukraine aid."
            detail = "I have mixed feelings about U.S. support for Ukraine."

    # Fictional extrapolation example
    else:
        decisive = "I have not formed an opinion on this topic."
        detail = "This is not something I have thought deeply about."

    return {
        "decisive_stance": decisive,
        "detail": detail,
        "evidence_type": evidence_type
    }


def generate_stance_wording_generic(var: str, spec: Dict, value: float, evidence_type: str) -> Dict:
    """
    Generate generic stance wording for variables without custom templates.
    Uses the full question text from question_mappings.json and generates
    DIRECTIONAL stance based on value and scale endpoints.
    """
    import re

    desc = spec.get("desc", var)
    domain = spec.get("domain", "policy")

    # Get full question text with scale endpoints from question_mappings.json
    full_question = QUESTION_MAPPINGS.get("quiz_questions", {}).get(var, "")

    if full_question:
        # Parse out the scale endpoints to determine direction
        # Example: "Do you favor... (1 = Favor a great deal, 7 = Oppose a great deal)"

        # Extract the main topic/question (before the scale info)
        topic_match = re.match(r'^(.+?)\s*\(1\s*=', full_question)
        if topic_match:
            topic = topic_match.group(1).strip()
        else:
            topic = full_question.split('(')[0].strip() if '(' in full_question else full_question

        # Extract scale endpoints
        scale_match = re.search(r'\(1\s*=\s*([^,)]+),\s*(\d+)\s*=\s*([^)]+)\)', full_question)

        if scale_match:
            low_endpoint = scale_match.group(1).strip()
            high_value = int(scale_match.group(2))
            high_endpoint = scale_match.group(3).strip()

            # Determine position strength based on value
            # For a 1-7 scale: 1-2 = strong low, 3-5 = moderate, 6-7 = strong high
            # For a 1-4/5 scale: adjust thresholds proportionally
            midpoint = (1 + high_value) / 2.0

            if value <= midpoint - 1.5:
                # Strong position toward low end
                decisive = f"{topic}: I am strongly {low_endpoint.lower()}"
            elif value >= midpoint + 1.5:
                # Strong position toward high end
                decisive = f"{topic}: I am strongly {high_endpoint.lower()}"
            elif value < midpoint:
                # Moderate position toward low end
                decisive = f"{topic}: I lean toward {low_endpoint.lower()}"
            elif value > midpoint:
                # Moderate position toward high end
                decisive = f"{topic}: I lean toward {high_endpoint.lower()}"
            else:
                # Exactly at midpoint
                decisive = f"{topic}: I am moderate/neutral"

            detail = f"Position: {value:.1f} on a 1-{high_value} scale (1={low_endpoint}, {high_value}={high_endpoint})"
        else:
            # Fallback if we can't parse endpoints
            decisive = f"{topic}: My position is {value:.1f}"
            detail = f"Based on: {full_question}"
    else:
        # Fallback to basic description
        if value >= 5.0:
            decisive = f"On {desc.lower()}, I hold a strong position."
            detail = f"I score {value:.1f} on the scale."
        elif value <= 3.0:
            decisive = f"On {desc.lower()}, I hold a clear position."
            detail = f"I score {value:.1f} on the scale."
        else:
            decisive = f"On {desc.lower()}, I'm moderate."
            detail = f"I score {value:.1f}, placing me in the middle."

    return {
        "decisive_stance": decisive,
        "detail": detail,
        "evidence_type": evidence_type
    }


def generate_avatars_json(cluster_profiles: List[Dict], ml_predictions: Optional[Dict] = None, use_llm: bool = False, api_key: Optional[str] = None) -> Dict:
    """
    Generate avatars.json with persona definitions and stance tables.

    Parameters:
    - cluster_profiles: List of cluster profile dicts
    - ml_predictions: Optional ML predictions dict
    - use_llm: Whether to use LangChain for enhanced wording (optional)
    - api_key: LLM API key (optional)

    Returns:
    - avatars dict
    """
    # Map old variable names to topic names for backwards compatibility
    CUSTOM_TOPIC_MAP = {
        "V241248": "abortion",
        "V241386": "immigration",
        "V241239": "redistribution",
        "V241258": "environment",
        "V241245": "health",
        "V241242": "defense",
        "V241275x": "welfare",
        "V241726": "efficacy_trust",
        "V241731": "democracy",
        "V241732": "vaccines"
    }

    avatars = {}

    for profile in cluster_profiles:
        cluster_id = profile["cluster"]
        persona_name = generate_persona_name(cluster_id, profile)

        # Build stance table
        stances = {}

        # Observed stances (from ALL clustering features)
        # ALWAYS use generic wording with scale endpoints (no custom templates)
        for var, spec in CLUSTERING_FEATURES_V2.items():
            # Use mean_original_* (original scale 1-7) instead of centroid_* (z-scores)
            value = profile.get(f"mean_original_{var}", np.nan)
            if not np.isnan(value):
                # Use generic wording with scale endpoints for all variables
                topic = f"{spec.get('domain', 'policy')}_{var}"
                wording = generate_stance_wording_generic(var, spec, value, "observed")

                stances[topic] = {
                    **wording,
                    "value": round(float(value), 2),
                    "variable": var
                }

        # ML-inferred stances: REMOVED per user request
        # All ML predictions had incorrect liberal/conservative assumptions
        # and cluttered the "All Policy Positions" display

        avatars[cluster_id] = {
            "name": persona_name,
            "cluster": cluster_id,
            "description": f"A simulated voter representing Cluster {cluster_id}",
            "voice_rules": [
                "Speak in first person",
                "Be decisive and clear",
                "Start with your stance, then explain",
                "Do not claim to be a real person",
                "Do not roleplay protected demographics"
            ],
            "stances": stances
        }

    return {"personas": avatars}


def generate_story_json(cluster_profiles: List[Dict], metadata: Dict, use_llm: bool = False, api_key: Optional[str] = None) -> Dict:
    """
    Generate story.json with a guided narrative including statistical insights.

    Uses deterministic template or LangChain if enabled.
    """
    K = metadata["K"]
    universe = metadata["analysis_universe"]

    # Compute statistical insights from cluster data
    partyid_values = [c.get("partyid_mean") for c in cluster_profiles if c.get("partyid_mean") is not None]
    trump_shares = [c.get("vote_trump", 0) for c in cluster_profiles]
    harris_shares = [c.get("vote_harris", 0) for c in cluster_profiles]
    sizes = [c.get("size", 0) for c in cluster_profiles]

    # Find most/least Republican clusters
    if partyid_values:
        most_rep_idx = partyid_values.index(max(partyid_values))
        most_dem_idx = partyid_values.index(min(partyid_values))
        most_rep_cluster = cluster_profiles[most_rep_idx]
        most_dem_cluster = cluster_profiles[most_dem_idx]

        # Calculate weighted averages
        total_size = sum(sizes)
        avg_trump = sum(t*s for t,s in zip(trump_shares, sizes)) / total_size if total_size > 0 else 0
        avg_harris = sum(h*s for h,s in zip(harris_shares, sizes)) / total_size if total_size > 0 else 0

        # Find swing clusters (Party ID between 3.5 and 4.5)
        swing_clusters = [c for c in cluster_profiles if c.get("partyid_mean") and 3.5 <= c["partyid_mean"] <= 4.5]
        num_swing = len(swing_clusters)

        # Find largest cluster
        largest_cluster = max(cluster_profiles, key=lambda c: c.get("size", 0))
        largest_pct = (largest_cluster["size"] / total_size * 100) if total_size > 0 else 0

    # Build story with statistical insights
    story = {
        "title": "Statistical Insights: The 2024 Political Landscape",
        "steps": [
            {
                "step": 1,
                "title": "Overview",
                "content": f"We analyzed {universe.replace('_', ' ')} from the 2024 ANES survey and discovered {K} distinct ideological clusters using K-means clustering on ~50 policy dimensions. These clusters represent real patterns in how Americans combine different political beliefs across topics including abortion, immigration, fiscal policy, environment, healthcare, crime, Israel, and more.",
                "highlight": None
            },
            {
                "step": 2,
                "title": "Partisan Distribution",
                "content": f"The most Republican-leaning cluster (Cluster {most_rep_cluster['cluster']}) has an average Party ID of {most_rep_cluster['partyid_mean']:.1f} (7 = Strong Republican) with {most_rep_cluster['vote_trump']*100:.0f}% Trump support. The most Democratic cluster (Cluster {most_dem_cluster['cluster']}) has Party ID {most_dem_cluster['partyid_mean']:.1f} (1 = Strong Democrat) with {most_dem_cluster['vote_harris']*100:.0f}% Harris support.",
                "highlight": "explore_tab"
            },
            {
                "step": 3,
                "title": "The Middle Ground",
                "content": f"We found {num_swing} clusters with moderate Party ID (between 3.5 and 4.5), suggesting that {num_swing}/{K} clusters represent swing voters or independents. These groups show more varied combinations of policy preferences that don't align with traditional party platforms.",
                "highlight": None
            },
            {
                "step": 4,
                "title": "Dominant Cluster",
                "content": f"The largest cluster (Cluster {largest_cluster['cluster']}) represents {largest_pct:.1f}% of {universe.replace('_', ' ')}. This cluster has Party ID {largest_cluster.get('partyid_mean', 0):.1f}, voted {largest_cluster.get('vote_trump', 0)*100:.0f}% Trump / {largest_cluster.get('vote_harris', 0)*100:.0f}% Harris, and is {largest_cluster.get('pct_college', 0)*100:.0f}% college-educated.",
                "highlight": "quiz_tab"
            },
            {
                "step": 5,
                "title": "Vote Shares Across All Clusters",
                "content": f"Across all {K} clusters weighted by size, Trump received {avg_trump*100:.1f}% and Harris {avg_harris*100:.1f}%. However, individual clusters vary dramatically—from nearly 100% Trump in some clusters to nearly 100% Harris in others.",
                "highlight": None
            },
            {
                "step": 6,
                "title": "Evidence-Based Personas",
                "content": f"We created {K} simulated personas that represent each cluster's statistical profile. Each persona's stances are derived from: (1) observed survey responses on the ~50 clustering dimensions, (2) ML predictions for related attitudes, or (3) fictional extrapolation for unmeasured topics. All evidence levels are clearly labeled.",
                "highlight": "persona_tab"
            },
            {
                "step": 7,
                "title": "Explore the Data",
                "content": "Use the 3D cube to see how clusters differ across all policy dimensions. The 2D map uses PCA to project all dimensions onto a plane. Hover over any cluster to see full demographics: gender, age, education, region, and more. Take the 10-question quiz to find your nearest cluster!",
                "highlight": "explore_tab"
            }
        ]
    }

    return story

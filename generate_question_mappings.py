"""
Generate JavaScript mappings for quiz questions and 3D cube axis labels
from the ANES codebook.
"""

import pandas as pd
import json
import re
import sys

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load the codebook
codebook = pd.read_csv('anes_timeseries_2024_csv_20250808/anes_variable_scales.csv')

# Load quiz features
with open('docs/data/quiz_features.json', 'r') as f:
    quiz_data = json.load(f)
    all_features = list(quiz_data['all_feature_importance'].keys())

print(f"Generating mappings for {len(all_features)} features")

# Create mappings
quiz_questions = {}
axis_labels = {}

for var in all_features:
    # Find in codebook
    row = codebook[codebook['variable'] == var]

    if len(row) == 0:
        print(f"WARNING: Variable {var} not found in codebook")
        quiz_questions[var] = var
        axis_labels[var] = var
        continue

    row = row.iloc[0]

    # Extract question text (first line, clean up)
    question_full = str(row['question'])
    if question_full == 'nan' or pd.isna(row['question']):
        # Use label as fallback
        question_text = str(row['label'])
    else:
        question_lines = question_full.split('\n')
        question_text = question_lines[0].strip()

    # Simplify overly long questions (increased limit for LLM context)
    if len(question_text) > 400:
        # Take first sentence or first 300 chars
        first_sentence = question_text.split('.')[0]
        if len(first_sentence) < 400:
            question_text = first_sentence
        else:
            question_text = question_text[:300] + '...'

    # Replace special Unicode characters that cause Windows encoding issues
    question_text = question_text.replace('\ufb01', 'fi')  # ligature fi
    question_text = question_text.replace('\u2019', "'")  # right single quote
    question_text = question_text.replace('\u2018', "'")  # left single quote
    question_text = question_text.replace('\u201c', '"')  # left double quote
    question_text = question_text.replace('\u201d', '"')  # right double quote
    question_text = question_text.replace('\u2013', '-')  # en dash
    question_text = question_text.replace('\u2014', '-')  # em dash

    # Extract scale endpoints (from full_scale)
    scale_text = str(row['full_scale'])

    # Parse scale endpoints (e.g., "1 Always; ... 7 Never")
    # Find all numeric values and their labels (including negative numbers like -1, -2)
    scale_matches = re.findall(r'(-?\d+)\s+([^;]+)', scale_text)

    if len(scale_matches) >= 2:
        # Filter out invalid scale values
        valid_matches = []
        for match in scale_matches:
            val_str, label = match
            val = int(val_str)
            label_clean = label.strip()

            # Only include values 1-7 as valid scale points
            # Exclude:
            # - negative values (-1, -2, -8, -9)
            # - 8+ (special codes like "Side with neither")
            # - 99+ (DK/RF codes like "Haven't thought much")
            # - "Intermediate positions" labels (placeholder for 2-6 range)
            if 1 <= val <= 7 and 'Intermediate' not in label_clean:
                valid_matches.append((val_str, label))

        if len(valid_matches) >= 2:
            first_val, first_label = valid_matches[0]
            last_val, last_label = valid_matches[-1]

            # Clean labels
            first_label = first_label.strip().rstrip(';').strip()
            last_label = last_label.strip().rstrip(';').strip()

            # Create quiz question text with endpoints only
            quiz_questions[var] = f"{question_text} ({first_val} = {first_label}, {last_val} = {last_label})"
        elif len(valid_matches) == 1:
            # Only one valid value (unusual but handle it)
            first_val, first_label = valid_matches[0]
            first_label = first_label.strip().rstrip(';').strip()
            quiz_questions[var] = f"{question_text} ({first_val} = {first_label})"
        else:
            # No valid matches, fallback to just question
            quiz_questions[var] = question_text
    else:
        # Fallback: just use question
        quiz_questions[var] = question_text

    # Create descriptive but concise axis label
    # First check for specific variable mappings
    VARIABLE_MAPPINGS = {
        'V241239': 'Government Services Level',
        'V241242': 'Defense Spending',
        'V241245': 'Healthcare: Government vs Private',
        'V241248': 'Abortion Policy',
        'V241252': 'Government Jobs Guarantee',
        'V241255': 'Government Help for Black Americans',
        'V241258': 'Environmental Regulations',
        'V241231': 'Government Run by Elites',
        'V241330x': 'Voter Turnout Impact',
        'V241341': 'Candidate Rejection Likelihood',
        'V241386': 'Unauthorized Immigrants Policy',
        'V241396': 'English Language Importance',
    }

    if var in VARIABLE_MAPPINGS:
        axis_labels[var] = VARIABLE_MAPPINGS[var]
        continue

    # Use pattern matching to extract topic and create clean title
    label_lower = question_text.lower()
    axis_label = None

    # Pattern 1: Trust questions
    if 'trust' in label_lower and 'federal government' in label_lower:
        axis_label = 'Trust in Federal Government'
    elif 'trust' in label_lower and 'court' in label_lower:
        axis_label = 'Trust in Court System'
    elif 'trust' in label_lower and 'other people' in label_lower:
        axis_label = 'Trust in Other People'
    elif 'trust' in label_lower and 'news' in label_lower:
        axis_label = 'Trust in News Media'

    # Pattern 2: Party/ideology
    elif 'party' in label_lower or 'democrat' in label_lower or 'republican' in label_lower:
        if 'important' in label_lower:
            axis_label = 'Party Identity Importance'

    # Pattern 3: Abortion
    elif 'abortion' in label_lower:
        axis_label = 'Abortion Policy'

    # Pattern 4: LGBTQ issues
    elif 'transgender' in label_lower and 'bathroom' in label_lower:
        axis_label = 'Transgender Bathroom Rights'
    elif 'transgender' in label_lower and 'sports' in label_lower:
        axis_label = 'Transgender Sports Bans'
    elif ('gay' in label_lower or 'lesbian' in label_lower) and 'marry' in label_lower:
        axis_label = 'Same-Sex Marriage'
    elif ('gay' in label_lower or 'lesbian' in label_lower) and 'adopt' in label_lower:
        axis_label = 'Same-Sex Couple Adoption'
    elif ('gay' in label_lower or 'lesbian' in label_lower) and 'discrimination' in label_lower:
        axis_label = 'LGBTQ Job Discrimination'

    # Pattern 5: DEI
    elif 'dei' in label_lower or 'diversity, equity' in label_lower:
        axis_label = 'DEI Policies on Campus'

    # Pattern 6: Immigration
    elif 'unauthorized immigrants' in label_lower or 'illegal immigrants' in label_lower:
        axis_label = 'Unauthorized Immigrants Policy'
    elif 'birthright citizenship' in label_lower:
        axis_label = 'Birthright Citizenship'
    elif 'children brought illegally' in label_lower:
        axis_label = 'DACA / Dreamers Policy'
    elif 'wall' in label_lower and 'mexico' in label_lower:
        axis_label = 'Border Wall'
    elif 'border security' in label_lower:
        axis_label = 'Border Security Spending'
    elif 'speak english' in label_lower or 'everyone in the united states' in label_lower:
        axis_label = 'English Language Importance'

    # Pattern 7: Government services/spending
    elif 'government' in label_lower and 'services' in label_lower:
        axis_label = 'Government Services Level'
    elif 'waste' in label_lower and 'money' in label_lower:
        axis_label = 'Government Waste'
    elif 'run by' in label_lower and 'big interests' in label_lower:
        axis_label = 'Government Run by Elites'

    # Pattern 8: Spending questions
    elif 'defense spending' in label_lower:
        axis_label = 'Defense Spending'
    elif 'social security' in label_lower:
        axis_label = 'Social Security Spending'
    elif 'public schools' in label_lower:
        axis_label = 'Public School Spending'
    elif 'dealing with crime' in label_lower:
        axis_label = 'Crime Prevention Spending'
    elif 'highways' in label_lower:
        axis_label = 'Highway Infrastructure Spending'
    elif 'aid to the poor' in label_lower:
        axis_label = 'Aid to Poor Spending'
    elif 'protecting the environment' in label_lower:
        axis_label = 'Environmental Protection Spending'

    # Pattern 9: Healthcare
    elif 'insurance' in label_lower and ('government' in label_lower or 'private' in label_lower):
        axis_label = 'Healthcare: Government vs Private'
    elif 'paid leave' in label_lower:
        axis_label = 'Paid Parental Leave'

    # Pattern 10: Environment/climate
    elif 'rising temperatures' in label_lower or 'climate' in label_lower:
        axis_label = 'Climate Change Action'
    elif 'environment' in label_lower and 'business' in label_lower:
        axis_label = 'Environmental Regulations'

    # Pattern 11: Foreign policy
    elif 'ukraine' in label_lower:
        axis_label = 'Aid to Ukraine'
    elif 'israel' in label_lower and 'military' in label_lower:
        axis_label = 'Military Aid to Israel'
    elif 'palestinians' in label_lower or 'gaza' in label_lower:
        if 'humanitarian' in label_lower:
            axis_label = 'Humanitarian Aid to Gaza'
        elif 'protests' in label_lower:
            axis_label = 'Gaza Protest Approval'
        elif 'side' in label_lower:
            axis_label = 'Israel-Palestine Conflict'
    elif 'military force' in label_lower:
        axis_label = 'Use of Military Force'
    elif 'stayed home' in label_lower:
        axis_label = 'Isolationism vs Intervention'

    # Pattern 12: Education/colleges
    elif 'colleges' in label_lower and 'universities' in label_lower:
        axis_label = 'College Administration Approval'

    # Pattern 13: Voting/democracy
    elif 'photo id' in label_lower and 'vote' in label_lower:
        axis_label = 'Voter ID Laws'
    elif 'felons' in label_lower and 'vote' in label_lower:
        axis_label = 'Felon Voting Rights'

    # Pattern 14: Race
    elif 'blacks' in label_lower or 'african american' in label_lower:
        if 'government help' in label_lower or 'help themselves' in label_lower:
            axis_label = 'Government Help for Black Americans'
    elif 'gap' in label_lower and ('income' in label_lower or 'wealth' in label_lower):
        axis_label = 'Income/Wealth Gap'

    # Pattern 15: Crime/police
    elif 'urban unrest' in label_lower:
        axis_label = 'Urban Unrest Response'
    elif 'death penalty' in label_lower:
        axis_label = 'Death Penalty'

    # Pattern 16: Jobs/economy
    elif 'jobs' in label_lower and 'standard of living' in label_lower:
        axis_label = 'Government Jobs Guarantee'

    # Pattern 17: 7-point scale questions (specific handling)
    elif '7-point' in label_lower or 'where would you place yourself' in label_lower:
        if 'services' in label_lower:
            axis_label = 'Government Services Level'
        elif 'defense spending' in label_lower:
            axis_label = 'Defense Spending'
        elif 'insurance' in label_lower:
            axis_label = 'Healthcare: Government vs Private'
        elif 'abortion' in label_lower:
            axis_label = 'Abortion Policy'
        elif 'blacks' in label_lower or 'help themselves' in label_lower:
            axis_label = 'Government Help for Black Americans'
        elif 'environment' in label_lower and 'business' in label_lower:
            axis_label = 'Environmental Regulations'
        elif 'jobs' in label_lower:
            axis_label = 'Government Jobs Guarantee'

    # Pattern 18: Misc
    elif 'candidate' in label_lower and 'voting' in label_lower:
        axis_label = 'Likelihood to Vote Against'
    elif 'run by' in label_lower:
        axis_label = 'Government Run by Elites'

    # Fallback: clean up the question text
    if not axis_label:
        axis_label = question_text
        # Remove scale info
        axis_label = re.sub(r'\s*\(1\s*=.*\).*', '', axis_label)
        # Limit to first 50 chars at word boundary
        if len(axis_label) > 50:
            axis_label = axis_label[:50].rsplit(' ', 1)[0] + '...'

    axis_labels[var] = axis_label

# Generate JavaScript code
print("\n// For getQuizQuestionText():")
print("const questions = {")
for var in sorted(quiz_questions.keys()):
    # Escape quotes
    text = quiz_questions[var].replace("'", "\\'")
    print(f"    '{var}': '{text}',")
print("};")

print("\n\n// For getReadableFeatureName():")
print("const labels = {")
for var in sorted(axis_labels.keys()):
    text = axis_labels[var].replace("'", "\\'")
    print(f"    '{var}': '{text}',")
print("};")

# Save to JSON for reference
output = {
    'quiz_questions': quiz_questions,
    'axis_labels': axis_labels
}
with open('docs/data/question_mappings.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n\nSaved to docs/data/question_mappings.json")
